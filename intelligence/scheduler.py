"""
Intelligence Scheduler

Runs background jobs during market hours:
  - Every 15 min (09:30–15:15): full F&O universe scan + auto execute
  - Every 15 min:               refresh global market context
  - Every 30 min:               pre-warm news sentiment for watchlist
  - 09:10 IST:                  morning briefing (pre-market summary)
  - 15:35 IST:                  EOD summary (P&L + movers)

Uses APScheduler (AsyncIOScheduler) timezone-aware for IST.
"""
from __future__ import annotations

import asyncio
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

log = structlog.get_logger(__name__)

# Symbols to pre-warm intelligence for (loaded from strategies.yaml + NIFTY 50)
_WATCHLIST: list[str] = []


def _get_watchlist() -> list[str]:
    global _WATCHLIST
    if _WATCHLIST:
        return _WATCHLIST
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).parent.parent / "config" / "strategies.yaml"
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        symbols = []
        for strat in cfg.get("strategies", {}).values():
            symbols.extend(strat.get("symbols", []))
        _WATCHLIST = list(set(symbols)) or ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK"]
    except Exception:
        _WATCHLIST = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK"]
    return _WATCHLIST


async def _refresh_market_context() -> None:
    """Refresh global market context cache."""
    from intelligence.market_context import MarketContextTracker
    tracker = MarketContextTracker()
    # Force fresh fetch by clearing cache
    r = tracker._redis
    if r:
        await r.delete("market:context")
    ctx = await tracker.get_context()
    log.info("scheduler.context_refreshed", bias=ctx.market_bias, score=ctx.context_score)


async def _prewarm_sentiment() -> None:
    """Pre-warm sentiment cache for all watchlist symbols."""
    from intelligence.news import NewsSentimentAnalyzer
    analyzer = NewsSentimentAnalyzer()
    symbols = _get_watchlist()
    for sym in symbols:
        try:
            await analyzer.get_symbol_sentiment(sym)
        except Exception as e:
            log.warning("scheduler.sentiment_prewarm_failed", symbol=sym, error=str(e))
        await asyncio.sleep(1.0)  # respect rate limits


async def _morning_briefing() -> None:
    """Send pre-market intelligence briefing to Telegram at 09:10 IST."""
    from intelligence.market_context import MarketContextTracker
    from intelligence.news import NewsSentimentAnalyzer
    from utils.notifications import Notifier

    tracker = MarketContextTracker()
    ctx = await tracker.get_context()

    analyzer = NewsSentimentAnalyzer()
    mood = await analyzer.get_market_mood()

    summary = tracker.summarize(ctx)
    text = (
        "📊 <b>Morning Market Briefing</b>\n\n"
        f"{summary}\n\n"
        f"News Mood: {mood['mood'].upper()} ({mood['score']:+.2f})\n\n"
        f"⚡ AutoTrader is {'PAPER' if True else 'LIVE'} trading today."
    )

    notifier = Notifier()
    await notifier._send(text)
    log.info("scheduler.morning_briefing_sent")


async def _run_ai_options_analysis() -> None:
    """Run AI options analysis every 30 min during market hours."""
    from signals.filters import MarketHoursFilter
    if not MarketHoursFilter.is_open():
        return
    from intelligence.ai_analyst import run_ai_analysis_and_notify
    await run_ai_analysis_and_notify()


async def _eod_summary() -> None:
    """Send end-of-day summary at 15:35 IST."""
    from risk.manager import RiskManager
    from utils.notifications import Notifier
    from brokers.factory import get_broker

    rm = RiskManager()
    pnl = await rm.get_daily_pnl()
    open_count = await rm.get_open_position_count()

    try:
        broker = get_broker()
        positions = broker.get_positions()
        trades_count = open_count
    except Exception:
        positions = []
        trades_count = 0

    notifier = Notifier()
    await notifier.send_daily_summary(pnl, trades_count, len(positions))
    log.info("scheduler.eod_summary_sent", pnl=pnl)


async def _run_stock_scan_and_execute() -> None:
    """
    Full F&O universe scan + auto execution.

    Runs every 15 minutes from 09:30 to 15:15 IST on trading days.
    First 15 min (09:15–09:30) skipped — price discovery, too noisy.
    Last 15 min (15:15–15:30) skipped — MIS auto square-off zone.
    """
    import pytz
    from datetime import datetime
    from signals.filters import MarketHoursFilter

    if not MarketHoursFilter.is_open():
        return

    # Skip first 15 min (09:15–09:30) and last 15 min (15:15–15:30)
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).time()
    from datetime import time as dtime
    if now < dtime(9, 30) or now >= dtime(15, 15):
        log.info("scheduler.scan_skipped", reason="Outside trading window (09:30–15:15)")
        return

    log.info("scheduler.scan_start")

    from scanner.stock_scanner import StockScanner
    from execution.auto_trader import AutoTrader
    from config.settings import settings

    try:
        scanner = StockScanner()
        report = await scanner.run_scan(top_n=10)

        if not report.top_picks:
            log.info("scheduler.scan_no_picks",
                     scanned=report.symbols_scanned,
                     blocked=report.symbols_blocked)
            return

        log.info(
            "scheduler.scan_picks",
            count=len(report.top_picks),
            top=[f"{p.symbol}({p.score})" for p in report.top_picks[:3]],
        )

        # Execute top picks (paper trading if settings.paper_trading=True)
        trader = AutoTrader()
        results = await trader.execute_scan_results(
            picks=report.top_picks,
            dry_run=False,  # Controlled by settings.paper_trading inside
        )

        executed = [r for r in results if r.status in ("executed", "paper")]
        log.info(
            "scheduler.scan_executed",
            executed=len(executed),
            skipped=len(results) - len(executed),
        )

    except Exception as e:
        log.error("scheduler.scan_failed", error=str(e), exc_info=True)


def build_scheduler() -> AsyncIOScheduler:
    """Build and return the APScheduler instance (not yet started)."""
    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")

    # ── Stock scan + auto execute every 15 min (09:30–15:15 IST, Mon–Fri) ──
    scheduler.add_job(
        _run_stock_scan_and_execute,
        CronTrigger(
            minute="*/15",
            hour="9,10,11,12,13,14,15",
            day_of_week="mon-fri",
            timezone="Asia/Kolkata",
        ),
        id="stock_scan_execute",
        name="Full F&O scan + auto execute",
        replace_existing=True,
        misfire_grace_time=120,
        max_instances=1,  # Never run two scans simultaneously
    )

    # ── Refresh context every 15 min ──────────────────────────────────────
    scheduler.add_job(
        _refresh_market_context,
        IntervalTrigger(minutes=15),
        id="refresh_context",
        name="Refresh global market context",
        replace_existing=True,
        misfire_grace_time=120,
    )

    # ── Pre-warm sentiment every 30 min ───────────────────────────────────
    scheduler.add_job(
        _prewarm_sentiment,
        IntervalTrigger(minutes=30),
        id="prewarm_sentiment",
        name="Pre-warm news sentiment cache",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # ── Morning briefing 09:10 IST, Mon–Fri ──────────────────────────────
    scheduler.add_job(
        _morning_briefing,
        CronTrigger(hour=9, minute=10, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="morning_briefing",
        name="Morning market briefing",
        replace_existing=True,
    )

    # ── EOD summary 15:35 IST, Mon–Fri ───────────────────────────────────
    scheduler.add_job(
        _eod_summary,
        CronTrigger(hour=15, minute=35, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="eod_summary",
        name="End-of-day P&L summary",
        replace_existing=True,
    )

    # ── AI options analysis every 30 min ─────────────────────────────────
    scheduler.add_job(
        _run_ai_options_analysis,
        IntervalTrigger(minutes=30),
        id="ai_options_analysis",
        name="AI options signal analysis",
        replace_existing=True,
        misfire_grace_time=300,
        max_instances=1,
    )

    log.info("scheduler.built", job_count=len(scheduler.get_jobs()))
    return scheduler
