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

    @app.get("/")
    async def root_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/ui", status_code=302)

    @app.get("/ui")
    async def dashboard_ui():
        from fastapi.responses import FileResponse
        from pathlib import Path
        return FileResponse(Path(__file__).parent.parent / "static" / "index.html")

    @app.get("/debug/ai")
    async def debug_ai():
        """Debug: test OpenRouter connectivity via httpx."""
        import os, traceback, httpx
        key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        result = {"key_set": bool(key), "key_prefix": key[:12] + "…" if key else "MISSING"}
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "openai/gpt-oss-20b:free",
                          "messages": [{"role": "user", "content": 'Reply: {"ok":true}'}],
                          "max_tokens": 20},
                )
            result["status_code"] = resp.status_code
            result["response"] = resp.text[:300]
            result["status"] = "ok" if resp.status_code == 200 else "http_error"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()[-600:]
        return result

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
        if settings.paper_trading:
            return {"positions": [], "mode": "paper", "note": "Paper trading mode — no live broker positions"}
        try:
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
        except Exception as e:
            log.warning("positions.error", error=str(e))
            return {"positions": [], "note": f"Broker unavailable: {e}"}

    @router.post("/api/ai/run")
    async def run_ai_analysis():
        """Single-turn AI options analysis — fast enough for Vercel's 60s limit."""
        import time, json as _json

        FREE_MODELS = [
            "openai/gpt-oss-20b:free",
            "openai/gpt-oss-120b:free",
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "google/gemma-3-27b-it:free",
            "meta-llama/llama-3.2-3b-instruct:free",
        ]

        # Quick market context via yfinance
        def _get_quick_context():
            try:
                import yfinance as yf
                nifty = yf.download("^NSEI", period="5d", interval="1d", auto_adjust=True, progress=False)
                vix   = yf.download("^INDIAVIX", period="2d", interval="1d", auto_adjust=True, progress=False)
                import pandas as pd
                if isinstance(nifty.columns, pd.MultiIndex):
                    nifty.columns = [c[0].lower() for c in nifty.columns]
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = [c[0].lower() for c in vix.columns]
                n_last = float(nifty["close"].iloc[-1]) if not nifty.empty else 0
                n_prev = float(nifty["close"].iloc[-2]) if len(nifty) > 1 else n_last
                n_chg  = round((n_last - n_prev) / n_prev * 100, 2) if n_prev else 0
                v_last = float(vix["close"].iloc[-1]) if not vix.empty else 15.0
                return {"nifty_last": round(n_last, 0), "nifty_chg_pct": n_chg, "india_vix": round(v_last, 1)}
            except Exception:
                return {"nifty_last": 0, "nifty_chg_pct": 0, "india_vix": 15.0}

        import asyncio
        loop = asyncio.get_event_loop()
        ctx = await loop.run_in_executor(None, _get_quick_context)

        prompt = f"""You are an expert NSE options trader. Based on today's market:

NIFTY: {ctx['nifty_last']} ({ctx['nifty_chg_pct']:+.2f}% today)
India VIX: {ctx['india_vix']}

Analyse the market and suggest 1-2 high-conviction options trades for this week.
Reply ONLY with valid JSON (no markdown, no explanation):

{{"signals": [{{"symbol": "NIFTY", "action": "BUY_CE", "strike": 24000, "expiry": "weekly", "premium_est": 150, "target_pct": 40, "sl_pct": 20, "confidence": 0.72, "rationale": "brief reason", "risk_factors": ["VIX elevated"]}}]}}

action must be one of: BUY_CE, BUY_PE, SELL_CE, SELL_PE, HOLD
Use HOLD with empty signals if market is unclear. Max 2 signals."""

        import re, httpx
        api_key = settings.openrouter_api_key or ""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://india-autotrader.vercel.app",
            "X-Title": "India AutoTrader",
        }

        last_error = "No models available"
        async with httpx.AsyncClient(timeout=40) as client:
            for model in FREE_MODELS:
                try:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.3,
                            "max_tokens": 800,
                        },
                    )
                    if resp.status_code == 429:
                        log.warning("ai.rate_limited", model=model)
                        continue
                    if resp.status_code != 200:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        log.warning("ai.http_error", model=model, error=last_error)
                        continue

                    data = resp.json()
                    content = data["choices"][0]["message"]["content"] or ""

                    # Extract JSON block
                    m = re.search(r'\{.*\}', content, re.DOTALL)
                    if not m:
                        raise ValueError(f"No JSON in response: {content[:200]}")
                    parsed = _json.loads(m.group())
                    sigs = parsed.get("signals", [])

                    result = {
                        "signals": sigs,
                        "generated_at": int(time.time()),
                        "model": model.split("/")[-1].replace(":free", ""),
                        "nifty": ctx,
                    }

                    # Cache to Redis if available
                    try:
                        import redis.asyncio as aioredis
                        r = await aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
                        await r.setex("ai:options:latest", 2700, _json.dumps(result))
                        await r.aclose()
                    except Exception:
                        pass

                    log.info("ai.on_demand.complete", signals=len(sigs), model=model)
                    return result

                except Exception as e:
                    last_error = str(e)
                    log.warning("ai.model_failed", model=model, error=last_error)
                    continue

        return {"signals": [], "error": last_error, "generated_at": int(time.time())}

    @router.post("/api/scanner/run")
    async def run_scanner_now():
        """Synchronous on-demand scanner — returns picks directly using yfinance."""
        import time
        import asyncio

        WATCHLIST = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                     "SBIN", "BHARTIARTL", "KOTAKBANK", "WIPRO", "AXISBANK"]

        def _quick_scan():
            import yfinance as yf
            import pandas as pd
            picks = []
            for sym in WATCHLIST:
                try:
                    df = yf.download(f"{sym}.NS", period="60d", interval="1d",
                                     auto_adjust=True, progress=False)
                    if df.empty or len(df) < 20:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0].lower() for c in df.columns]
                    else:
                        df.columns = [c.lower() for c in df.columns]

                    close = df["close"]
                    sma20 = close.rolling(20).mean().iloc[-1]
                    sma5  = close.rolling(5).mean().iloc[-1]
                    last  = close.iloc[-1]
                    prev  = close.iloc[-2]
                    vol   = df["volume"].iloc[-1]
                    avg_vol = df["volume"].rolling(20).mean().iloc[-1]

                    delta = close.diff()
                    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
                    rsi = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] or 1e-9)))

                    chg_pct = round((last - prev) / prev * 100, 2)
                    vol_surge = round(vol / avg_vol, 1) if avg_vol > 0 else 1.0

                    action = "NEUTRAL"
                    score  = 0.0
                    if last > sma20 and sma5 > sma20 and rsi < 70:
                        action = "BUY"; score = round(min(rsi / 100 + vol_surge * 0.1, 1.0), 2)
                    elif last < sma20 and sma5 < sma20 and rsi > 30:
                        action = "SELL"; score = round(-min((100 - rsi) / 100 + vol_surge * 0.1, 1.0), 2)

                    picks.append({
                        "symbol": sym, "action": action, "score": score,
                        "ltp": round(float(last), 2), "change_pct": chg_pct,
                        "rsi": round(rsi, 1), "vol_surge": vol_surge,
                        "above_sma20": bool(last > sma20),
                        "wyckoff_phase": "Markup" if action == "BUY" else "Markdown" if action == "SELL" else "Distribution",
                        "breakout_type": "Volume Surge" if vol_surge > 1.5 else "",
                    })
                except Exception:
                    continue
            picks.sort(key=lambda p: abs(p["score"]), reverse=True)
            return picks

        try:
            loop = asyncio.get_event_loop()
            picks = await loop.run_in_executor(None, _quick_scan)
            result = {"top_picks": picks, "picks": picks,
                      "scanned_at": int(time.time()), "symbols_scanned": len(WATCHLIST)}

            # Cache to Redis if available
            try:
                import redis.asyncio as aioredis, json as _j
                r = await aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
                await r.setex("scanner:latest", 900, _j.dumps(result))
                await r.aclose()
            except Exception:
                pass

            log.info("scanner.on_demand.complete", picks=len(picks))
            return result
        except Exception as e:
            log.error("scanner.on_demand.error", error=str(e))
            return {"top_picks": [], "picks": [], "error": str(e), "scanned_at": int(time.time())}

    @router.get("/api/ohlcv/{symbol}")
    async def get_ohlcv(symbol: str, days: int = 90):
        """OHLCV candlestick data for Lightweight Charts (via yfinance fallback)."""
        try:
            import yfinance as yf
            import pandas as pd
            from datetime import datetime, timedelta

            ticker = symbol.upper()
            # Try NSE suffix first, fallback to BSE
            for suffix in [".NS", ".BO", ""]:
                try:
                    df = yf.download(
                        f"{ticker}{suffix}",
                        period=f"{days}d",
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                    )
                    if not df.empty:
                        break
                except Exception:
                    continue

            if df is None or df.empty:
                return {"symbol": ticker, "candles": [], "error": "No data available"}

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            candles = []
            for ts, row in df.iterrows():
                try:
                    t = int(ts.timestamp())
                    candles.append({
                        "time": t,
                        "open": round(float(row.get("open", 0)), 2),
                        "high": round(float(row.get("high", 0)), 2),
                        "low": round(float(row.get("low", 0)), 2),
                        "close": round(float(row.get("close", 0)), 2),
                        "volume": int(row.get("volume", 0)),
                    })
                except Exception:
                    continue
            return {"symbol": ticker, "candles": candles}
        except Exception as e:
            log.warning("ohlcv.error", symbol=symbol, error=str(e))
            return {"symbol": symbol.upper(), "candles": [], "error": str(e)}

    async def _redis_get(key: str):
        """Safely get a Redis key; returns None if Redis is unavailable."""
        try:
            import redis.asyncio as aioredis
            import json as _json
            r = await aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
            raw = await r.get(key)
            await r.aclose()
            return _json.loads(raw) if raw else None
        except Exception:
            return None

    @router.get("/api/ai/signals")
    async def get_ai_signals():
        """Latest AI options signals from background analysis."""
        data = await _redis_get("ai:options:latest")
        if data:
            return data
        return {"signals": [], "message": "AI analysis runs every 30 min during market hours. Connect Redis to persist signals."}

    @router.get("/api/market-context")
    async def get_market_context():
        """Live market context via yfinance — no heavy imports needed."""
        import asyncio

        def _fetch():
            import yfinance as yf
            import pandas as pd

            tickers = {
                "nifty":    "^NSEI",
                "vix":      "^INDIAVIX",
                "sensex":   "^BSESN",
                "sp500":    "^GSPC",
                "nasdaq":   "^IXIC",
                "usdinr":   "INR=X",
                "crude":    "CL=F",
                "giftnifty":"GC=F",   # gold as proxy if gift nifty unavailable
            }

            def _safe_pct(df, col="close"):
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0].lower() for c in df.columns]
                    else:
                        df.columns = [c.lower() for c in df.columns]
                    s = df[col].dropna()
                    if len(s) < 2: return 0.0, float(s.iloc[-1]) if len(s) else 0.0
                    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
                    return round((last - prev) / prev * 100, 2), round(last, 2)
                except Exception:
                    return 0.0, 0.0

            results = {}
            for name, sym in tickers.items():
                try:
                    df = yf.download(sym, period="5d", interval="1d",
                                     auto_adjust=True, progress=False)
                    pct, last = _safe_pct(df)
                    results[name] = {"last": last, "chg_pct": pct}
                except Exception:
                    results[name] = {"last": 0.0, "chg_pct": 0.0}

            n = results.get("nifty", {})
            v = results.get("vix",   {})
            sp = results.get("sp500", {})
            nq = results.get("nasdaq", {})
            fx = results.get("usdinr", {})
            cl = results.get("crude",  {})

            vix_val = v.get("last", 15.0)
            n_chg   = n.get("chg_pct", 0.0)
            sp_chg  = sp.get("chg_pct", 0.0)

            # Simple bias
            bull_pts = sum([
                n_chg > 0.3,
                vix_val < 15,
                sp_chg > 0.2,
                nq.get("chg_pct", 0) > 0.2,
            ])
            bear_pts = sum([
                n_chg < -0.3,
                vix_val > 20,
                sp_chg < -0.2,
            ])
            bias = "BULLISH" if bull_pts >= 3 else "BEARISH" if bear_pts >= 2 else "NEUTRAL"

            return {
                "bias": bias,
                "nifty_last": n.get("last", 0),
                "nifty_chg_pct": n_chg,
                "india_vix": round(vix_val, 1),
                "sensex_chg_pct": results.get("sensex", {}).get("chg_pct", 0),
                "sp500_change_pct": sp_chg,
                "nasdaq_change_pct": nq.get("chg_pct", 0),
                "usdinr": fx.get("last", 0),
                "crude_oil_usd": cl.get("last", 0),
                "gift_nifty_change_pct": n_chg,   # best proxy without direct feed
                "fii_net_crore": None,
                "dii_net_crore": None,
                "score": round((bull_pts - bear_pts) / 4.0, 2),
                "summary": (
                    f"NIFTY {n.get('last',0):.0f} ({n_chg:+.2f}%) | "
                    f"VIX {vix_val:.1f} | "
                    f"S&P {sp_chg:+.2f}% | "
                    f"USD/INR {fx.get('last',0):.2f} | "
                    f"Crude ${cl.get('last',0):.1f}"
                ),
            }

        try:
            loop = asyncio.get_event_loop()
            ctx = await loop.run_in_executor(None, _fetch)
            return ctx
        except Exception as e:
            log.warning("market_context.error", error=str(e))
            return {"bias": "NEUTRAL", "summary": "Market data unavailable", "error": str(e)}

    @router.get("/api/dashboard")
    async def get_dashboard_data():
        """Aggregated data for the UI dashboard."""
        import asyncio
        results = {}

        # Market context — fast yfinance fetch (no heavy imports)
        async def _ctx():
            try:
                return await get_market_context()
            except Exception:
                return {}

        # Run market context fetch concurrently with Redis lookups
        ctx_task = asyncio.create_task(_ctx())

        # AI signals and scanner from Redis (instant if available)
        ai_data  = await _redis_get("ai:options:latest") or {"signals": []}
        scan_data = await _redis_get("scanner:latest") or {}

        results["market_context"] = await ctx_task
        results["ai_signals"]     = ai_data
        results["scanner"]        = scan_data

        return results

    return router
