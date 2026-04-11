from __future__ import annotations

import hmac
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from config.settings import settings
from webhook.parser import TradingViewAlert

log = structlog.get_logger(__name__)


def _jsonify(obj):
    """Recursively convert numpy/pandas scalars to Python-native types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(i) for i in obj]
    return obj

# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    log.info("autotrader.startup", paper=settings.paper_trading, broker=settings.active_broker)

    # Start intelligence scheduler
    from intelligence.scheduler import build_scheduler
    scheduler = build_scheduler()
    scheduler.start()
    log.info("autotrader.scheduler.started")

    yield

    scheduler.shutdown(wait=False)
    log.info("autotrader.shutdown")


# ── App factory ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="India AutoTrader",
        description="TradingView webhook → Indian broker order execution",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
    )

    from fastapi.staticfiles import StaticFiles
    from pathlib import Path
    static_dir = Path(__file__).parent.parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(_build_router())

    @app.get("/ui")
    async def dashboard_ui():
        from fastapi.responses import FileResponse
        from pathlib import Path
        return FileResponse(Path(__file__).parent.parent / "static" / "index.html")

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_ERROR", "message": "An internal error occurred"}},
        )

    return app


# ── Auth dependency ───────────────────────────────────────────────────────────


def _verify_token(request: Request) -> None:
    """Constant-time comparison of webhook token to prevent timing attacks."""
    path_token = request.path_params.get("token", "")
    if not hmac.compare_digest(path_token, settings.webhook_secret):
        log.warning("webhook.auth.failed", ip=request.client.host if request.client else "unknown")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def _verify_ip(request: Request) -> None:
    """Optional IP whitelist check for TradingView sender IPs."""
    allowed = settings.tv_allowed_ips
    if not allowed:
        return
    client_ip = request.client.host if request.client else ""
    # Check X-Forwarded-For if behind a proxy
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    actual_ip = forwarded or client_ip
    if actual_ip not in allowed:
        log.warning("webhook.ip.blocked", ip=actual_ip)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="IP not allowed")


# ── Background task ───────────────────────────────────────────────────────────


async def _process_alert(alert: TradingViewAlert, received_at: float) -> None:
    """
    Full signal processing pipeline — runs in background so webhook returns fast.
    """
    from signals.processor import SignalProcessor
    processor = SignalProcessor()
    await processor.process(alert, received_at=received_at)


# ── Routes ────────────────────────────────────────────────────────────────────


def _build_router():
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/health")
    async def health():
        return {
            "status": "ok",
            "paper_trading": settings.paper_trading,
            "broker": settings.active_broker,
            "ts": int(time.time()),
        }

    @router.get("/broker/status")
    async def broker_status():
        from brokers.factory import get_broker
        broker = get_broker()
        return broker.health_check()

    @router.post(
        "/webhook/{token}",
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(_verify_token), Depends(_verify_ip)],
    )
    async def receive_alert(
        token: str,  # noqa: ARG001 — consumed by auth dep
        alert: TradingViewAlert,
        background_tasks: BackgroundTasks,
        request: Request,
    ):
        received_at = time.time()
        log.info(
            "webhook.received",
            symbol=alert.parsed_symbol,
            action=alert.action,
            strategy=alert.strategy,
            ip=request.client.host if request.client else "unknown",
        )
        # Respond immediately; process in background
        background_tasks.add_task(_process_alert, alert, received_at)
        return {"status": "queued", "symbol": alert.parsed_symbol, "action": alert.action}

    @router.post("/webhook/{token}/test")
    async def test_alert(
        token: str,  # noqa: ARG001
        alert: TradingViewAlert,
        _: None = Depends(_verify_token),
    ):
        """Dry-run endpoint: validate payload and return parsed signal without placing order."""
        return {
            "status": "dry_run",
            "parsed": {
                "symbol": alert.parsed_symbol,
                "exchange": alert.parsed_exchange,
                "action": alert.action,
                "qty": alert.qty,
                "price": alert.price,
                "strategy": alert.strategy,
                "order_type": "MARKET" if alert.price == 0 else "LIMIT",
            },
        }

    @router.get("/intelligence/context")
    async def get_market_context():
        """Global market context: SGX NIFTY, VIX, FII/DII, US markets."""
        from intelligence.market_context import MarketContextTracker
        tracker = MarketContextTracker()
        ctx = await tracker.get_context()
        return {
            "bias": ctx.market_bias,
            "score": ctx.context_score,
            "gift_nifty_change_pct": ctx.gift_nifty_change_pct,
            "sp500_change_pct": ctx.sp500_change_pct,
            "nasdaq_change_pct": ctx.nasdaq_change_pct,
            "india_vix": ctx.india_vix,
            "fii_net_crore": ctx.fii_net_crore,
            "dii_net_crore": ctx.dii_net_crore,
            "usdinr": ctx.usdinr,
            "crude_oil_usd": ctx.crude_oil_usd,
            "data_quality": ctx.data_quality,
            "summary": tracker.summarize(ctx),
        }

    @router.get("/intelligence/sentiment/{symbol}")
    async def get_news_sentiment(symbol: str):
        """News sentiment for a given NSE symbol."""
        from intelligence.news import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer()
        sentiment = await analyzer.get_symbol_sentiment(symbol.upper())
        label = (
            "STRONGLY BULLISH" if sentiment.score > 0.5
            else "BULLISH" if sentiment.score > 0.15
            else "BEARISH" if sentiment.score < -0.15
            else "STRONGLY BEARISH" if sentiment.score < -0.5
            else "NEUTRAL"
        )
        return {
            "symbol": sentiment.symbol,
            "score": sentiment.score,
            "label": label,
            "article_count": sentiment.article_count,
            "bullish_articles": sentiment.bullish_count,
            "bearish_articles": sentiment.bearish_count,
            "top_headlines": sentiment.top_headlines,
        }

    @router.get("/intelligence/score/{symbol}/{action}")
    async def get_signal_score(symbol: str, action: str):
        """Full multi-factor confidence score for a BUY or SELL signal."""
        if action.upper() not in {"BUY", "SELL"}:
            raise HTTPException(status_code=400, detail="action must be BUY or SELL")
        from intelligence.scorer import SignalScorer
        scorer = SignalScorer()
        result = await scorer.score(symbol=symbol.upper(), action=action.upper())
        return {
            "symbol": result.symbol,
            "action": result.action,
            "score": result.score,
            "strength": result.strength.value,
            "size_multiplier": result.size_multiplier,
            "skip": result.skip,
            "rationale": result.rationale,
            "breakdown": {
                "technical_pts": result.breakdown.technical_pts,
                "news_pts": result.breakdown.news_pts,
                "context_pts": result.breakdown.context_pts,
                "momentum_pts": result.breakdown.momentum_pts,
                "ema_aligned": result.breakdown.ema_aligned,
                "rsi_value": result.breakdown.rsi_value,
                "macd_aligned": result.breakdown.macd_aligned,
                "supertrend_aligned": result.breakdown.supertrend_aligned,
                "volume_above_avg": result.breakdown.volume_above_avg,
                "news_score": result.breakdown.news_score,
                "vix_level": result.breakdown.vix_level,
                "momentum_5d_pct": result.breakdown.momentum_5d,
                "warnings": result.breakdown.warnings,
            },
        }

    @router.get("/scanner/latest")
    async def get_scanner_results():
        """Return the latest stock scan results (cached, updates every 15 min)."""
        from scanner.stock_scanner import StockScanner
        scanner = StockScanner()
        cached = await scanner.get_cached_results()
        if cached:
            return cached
        return {"message": "No scan results yet. Scanner runs every 15 min from 09:30 IST."}

    @router.post("/scanner/run")
    async def trigger_scan(background_tasks: BackgroundTasks):
        """Manually trigger a stock scan (runs in background, check /scanner/latest)."""
        async def _do_scan():
            from scanner.stock_scanner import StockScanner
            from execution.auto_trader import AutoTrader
            scanner = StockScanner()
            report = await scanner.run_scan()
            if report.top_picks:
                trader = AutoTrader()
                await trader.execute_scan_results(report.top_picks)
        background_tasks.add_task(_do_scan)
        return {"message": "Scan triggered. Results available at /scanner/latest in ~2 min."}

    @router.get("/scanner/analyze/{symbol}")
    async def analyze_symbol(symbol: str):
        """Full analysis for a single symbol: S/R, breakout, Wyckoff, SMC, manipulation."""
        from data.historical import HistoricalDataFetcher
        from datetime import datetime, timedelta
        from intelligence.manipulation import ManipulationDetector
        from intelligence.wyckoff import WyckoffAnalyzer
        from intelligence.smart_money import SMCAnalyzer
        from scanner.support_resistance import SRFinder
        from scanner.breakout_detector import BreakoutDetector
        from intelligence.scorer import SignalScorer

        fetcher = HistoricalDataFetcher()
        to_d = datetime.now().strftime("%Y-%m-%d")
        from_d = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")

        import asyncio
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None, fetcher.get_with_indicators, symbol.upper(), from_d, to_d, None
        )
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        manip = await loop.run_in_executor(None, ManipulationDetector().detect, df)
        wyckoff = await loop.run_in_executor(None, WyckoffAnalyzer().analyze, df)
        smc = await loop.run_in_executor(None, SMCAnalyzer().analyze, df)
        sr = await loop.run_in_executor(None, SRFinder().find, df)
        breakout = await loop.run_in_executor(
            None,
            lambda: BreakoutDetector().detect(df, sr, manip.manipulation_score, wyckoff.trade_bias)
        )

        return _jsonify({
            "symbol": symbol.upper(),
            "price": float(df["close"].iloc[-1]),
            "manipulation": {
                "score": manip.manipulation_score,
                "block_trade": manip.block_trade,
                "operator_direction": manip.operator_direction,
                "warnings": manip.warnings,
            },
            "wyckoff": {
                "phase": wyckoff.phase.value,
                "trade_bias": wyckoff.trade_bias,
                "confidence": wyckoff.confidence,
                "spring": wyckoff.spring_detected,
                "upthrust": wyckoff.upthrust_detected,
                "demand_dominates": wyckoff.demand_dominates,
                "supply_dominates": wyckoff.supply_dominates,
                "pv_divergence": wyckoff.pv_divergence,
                "reasoning": wyckoff.reasoning,
            },
            "smart_money": {
                "bos_direction": smc.bos_direction,
                "choch_detected": smc.choch_detected,
                "price_at_bullish_ob": smc.price_at_bullish_ob,
                "price_at_bearish_ob": smc.price_at_bearish_ob,
                "liquidity_swept": smc.liquidity_swept,
                "sweep_confirmed": smc.sweep_confirmed,
                "in_discount_zone": smc.in_discount_zone,
                "in_premium_zone": smc.in_premium_zone,
                "bullish_signals": smc.bullish_confluence,
                "bearish_signals": smc.bearish_confluence,
                "evidence": smc.evidence_summary,
            },
            "support_resistance": {
                "zone": sr.current_zone,
                "pdh": sr.pdh,
                "pdl": sr.pdl,
                "pivot": sr.pivot,
                "r1": sr.r1, "s1": sr.s1,
                "r2": sr.r2, "s2": sr.s2,
                "poc": sr.poc,
                "nearest_support": {
                    "price": sr.nearest_support.price,
                    "strength": sr.nearest_support.strength,
                    "sources": sr.nearest_support.sources,
                } if sr.nearest_support else None,
                "nearest_resistance": {
                    "price": sr.nearest_resistance.price,
                    "strength": sr.nearest_resistance.strength,
                    "sources": sr.nearest_resistance.sources,
                } if sr.nearest_resistance else None,
                "all_supports": [{"price": s.price, "strength": s.strength} for s in sr.support_levels[:5]],
                "all_resistances": [{"price": r.price, "strength": r.strength} for r in sr.resistance_levels[:5]],
            },
            "breakout": {
                "detected": breakout.detected,
                "direction": breakout.direction,
                "type": breakout.breakout_type,
                "level": breakout.breakout_level,
                "level_strength": breakout.level_strength,
                "volume_confirmed": breakout.volume_confirmed,
                "volume_ratio": breakout.volume_ratio,
                "confidence": breakout.confidence,
                "false_breakout_risk": breakout.false_breakout_risk,
                "false_breakout_reasons": breakout.false_breakout_reasons,
                "retest_entry": breakout.retest_entry,
                "reasons": breakout.reasons,
            },
        })

    @router.get("/positions")
    async def get_positions():
        from brokers.factory import get_broker
        broker = get_broker()
        positions = broker.get_positions()
        return {
            "positions": [
                {
                    "symbol": p.symbol,
                    "exchange": p.exchange,
                    "qty": p.qty,
                    "avg_price": p.avg_price,
                    "ltp": p.ltp,
                    "pnl": p.pnl,
                }
                for p in positions
            ]
        }

    @router.get("/api/ohlcv/{symbol}")
    async def get_ohlcv(symbol: str, days: int = 90):
        """OHLCV candlestick data for Lightweight Charts (via yfinance fallback)."""
        from data.nse_data import NSEDataProvider
        from datetime import datetime, timedelta
        import asyncio

        provider = NSEDataProvider()
        to_d = datetime.now().strftime("%Y-%m-%d")
        from_d = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None, provider.get_equity_ohlcv, symbol.upper(), from_d, to_d
        )
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        candles = []
        for _, row in df.iterrows():
            ts = int(row["date"].timestamp()) if hasattr(row["date"], "timestamp") else int(row["date"])
            candles.append({
                "time": ts,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]),
            })
        return {"symbol": symbol.upper(), "candles": candles}

    @router.get("/api/ai/signals")
    async def get_ai_signals():
        """Latest AI options signals from background analysis."""
        import redis.asyncio as aioredis
        import json as _json
        r = await aioredis.from_url(settings.redis_url, decode_responses=True)
        data = await r.get("ai:options:latest")
        if data:
            return _json.loads(data)
        return {"signals": [], "message": "AI analysis runs every 30 min during market hours"}

    @router.get("/api/dashboard")
    async def get_dashboard_data():
        """Aggregated data for the UI dashboard."""
        import redis.asyncio as aioredis
        import json as _json
        r = await aioredis.from_url(settings.redis_url, decode_responses=True)

        results = {}

        # Market context — use cache or fetch live
        ctx_raw = await r.get("market:context")
        if ctx_raw:
            results["market_context"] = _json.loads(ctx_raw)
        else:
            try:
                from intelligence.market_context import MarketContextTracker
                tracker = MarketContextTracker()
                ctx = await tracker.get_context()
                results["market_context"] = {
                    "bias": ctx.market_bias, "score": ctx.context_score,
                    "india_vix": ctx.india_vix,
                    "gift_nifty_change_pct": ctx.gift_nifty_change_pct,
                    "sp500_change_pct": ctx.sp500_change_pct,
                    "nasdaq_change_pct": ctx.nasdaq_change_pct,
                    "fii_net_crore": ctx.fii_net_crore, "dii_net_crore": ctx.dii_net_crore,
                    "usdinr": ctx.usdinr, "crude_oil_usd": ctx.crude_oil_usd,
                    "summary": tracker.summarize(ctx),
                }
            except Exception:
                results["market_context"] = {}

        # AI signals
        ai_raw = await r.get("ai:options:latest")
        results["ai_signals"] = _json.loads(ai_raw) if ai_raw else {"signals": []}

        # Scanner picks
        scan_raw = await r.get("scanner:latest")
        results["scanner"] = _json.loads(scan_raw) if scan_raw else {}

        return results

    return router
