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
    """Recursively convert scalars to Python-native types for JSON serialization."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj
    # Handle numpy types if numpy happens to be available
    try:
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(i) for i in obj]
    return obj

# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    log.info("autotrader.startup", paper=settings.paper_trading, broker=settings.active_broker)

    scheduler = None
    if not settings.is_production:
        # Scheduler only runs in local/dev — serverless functions have no persistent process
        try:
            from intelligence.scheduler import build_scheduler
            scheduler = build_scheduler()
            scheduler.start()
            log.info("autotrader.scheduler.started")
        except Exception as e:
            log.warning("autotrader.scheduler.skipped", reason=str(e))

    yield

    if scheduler:
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
    async def get_positions(request: Request):
        import httpx as _httpx
        token = _kite_token_from_request(request) or await _redis_get("zerodha:access_token")

        # Use live Kite positions if token is available
        if token:
            try:
                async with _httpx.AsyncClient(timeout=10) as client:
                    r = await client.get("https://api.kite.trade/portfolio/positions",
                                         headers=_kite_headers(token))
                if r.status_code == 200:
                    raw = r.json().get("data", {}).get("day", [])
                    positions = [
                        {
                            "symbol": p["tradingsymbol"],
                            "exchange": p["exchange"],
                            "qty": p["quantity"],
                            "avg_price": round(p.get("average_price", 0), 2),
                            "ltp": round(p.get("last_price", 0), 2),
                            "pnl": round(p.get("pnl", 0), 2),
                            "product": p.get("product", ""),
                        }
                        for p in raw if p.get("quantity", 0) != 0
                    ]
                    return {"positions": positions, "mode": "live", "source": "kite"}
                log.warning("positions.kite_error", status=r.status_code)
            except Exception as e:
                log.warning("positions.kite_error", error=str(e))

        if settings.paper_trading:
            return {"positions": [], "mode": "paper", "note": "Paper trading — connect Zerodha for live positions"}
        return {"positions": [], "note": "Zerodha not connected. Visit /broker/zerodha/login"}

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

        # Quick market context via Yahoo Finance v8 API (no pandas/yfinance needed)
        async def _get_quick_context():
            import httpx as _httpx
            _headers = {"User-Agent": "Mozilla/5.0"}
            _base = "https://query1.finance.yahoo.com/v8/finance/chart"

            async def _yf_meta(sym):
                """Fetch (last_price, prev_close) from Yahoo Finance meta — no close lag."""
                try:
                    async with _httpx.AsyncClient(timeout=10) as c:
                        r = await c.get(f"{_base}/{sym}",
                                        params={"interval": "1d", "range": "5d"},
                                        headers=_headers)
                    if r.status_code != 200:
                        return 0.0, 0.0
                    res = r.json().get("chart", {}).get("result", [])
                    if not res:
                        return 0.0, 0.0
                    meta = res[0].get("meta", {})
                    last = meta.get("regularMarketPrice") or 0.0
                    prev = meta.get("chartPreviousClose") or meta.get("previousClose") or last
                    return float(last), float(prev)
                except Exception:
                    return 0.0, 0.0

            (n_last, n_prev), (v_last, _) = await asyncio.gather(
                _yf_meta("^NSEI"), _yf_meta("^INDIAVIX"))
            n_chg  = round((n_last - n_prev) / n_prev * 100, 2) if n_prev else 0
            n_last = round(n_last, 0)
            v_last = round(v_last, 1) if v_last else 15.0
            return {"nifty_last": n_last, "nifty_chg_pct": n_chg, "india_vix": v_last}

        import asyncio
        ctx = await _get_quick_context()

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
        """Full-universe scanner: NIFTY 50 stocks, RSI+MACD+BB+EMA+ATR+Volume."""
        import time, asyncio, httpx as _httpx

        # ── NIFTY 50 universe — all have F&O, high liquidity ─────────────
        UNIVERSE = [
            "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
            "BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BHARTIARTL","BPCL",
            "BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY",
            "EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE",
            "HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK",
            "INFY","ITC","JSWSTEEL","KOTAKBANK","LT",
            "M&M","MARUTI","NTPC","ONGC","POWERGRID",
            "RELIANCE","SBIN","SUNPHARMA","TATACONSUM","TATAMOTORS",
            "TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","WIPRO",
        ]

        # ── Pure-Python technical indicators ─────────────────────────────
        def _ema(xs, p):
            if not xs or len(xs) < p:
                return xs[-1] if xs else 0.0
            k = 2 / (p + 1)
            e = sum(xs[:p]) / p
            for x in xs[p:]:
                e = x * k + e * (1 - k)
            return e

        def _ema_full(xs, p):
            """EMA series same length as input, None-padded at start."""
            if len(xs) < p:
                return xs[:]
            k = 2 / (p + 1)
            out = [None] * (p - 1)
            e = sum(xs[:p]) / p
            out.append(e)
            for x in xs[p:]:
                e = x * k + e * (1 - k)
                out.append(e)
            return out

        def _rsi(xs, p=14):
            if len(xs) < p + 1:
                return 50.0
            g = [max(xs[i]-xs[i-1], 0) for i in range(1, len(xs))]
            l = [max(xs[i-1]-xs[i], 0) for i in range(1, len(xs))]
            ag = sum(g[-p:]) / p; al = sum(l[-p:]) / p
            return round(100 - 100 / (1 + ag / al), 1) if al else 100.0

        def _macd(xs):
            if len(xs) < 35:
                return 0.0, 0.0, 0.0
            e12 = _ema_full(xs, 12); e26 = _ema_full(xs, 26)
            ml  = [e12[i] - e26[i] for i in range(len(xs))
                   if e12[i] is not None and e26[i] is not None]
            if not ml:
                return 0.0, 0.0, 0.0
            sig = _ema(ml, 9)
            return round(ml[-1], 4), round(sig, 4), round(ml[-1] - sig, 4)

        def _bb(xs, p=20, m=2.0):
            w = xs[-p:] if len(xs) >= p else xs
            n = len(w); mu = sum(w) / n
            std = (sum((c - mu)**2 for c in w) / n) ** 0.5
            return round(mu + m*std, 2), round(mu, 2), round(mu - m*std, 2)

        def _atr(hs, ls, cs, p=14):
            if len(cs) < 2:
                return 0.0
            trs = [max(hs[i]-ls[i], abs(hs[i]-cs[i-1]), abs(ls[i]-cs[i-1]))
                   for i in range(1, len(cs))]
            n = min(p, len(trs))
            return round(sum(trs[-n:]) / n, 2) if n else 0.0

        # ── Scoring + options recommendation ─────────────────────────────
        def _analyze(sym, last, prev, closes, highs, lows, volumes):
            if len(closes) < 50:
                return None
            ema9  = _ema(closes, 9);  ema20 = _ema(closes, 20); ema50 = _ema(closes, 50)
            rsi   = _rsi(closes)
            m_val, sig_val, hist = _macd(closes)
            bb_up, bb_mid, bb_lo = _bb(closes)
            atr   = _atr(highs, lows, closes)
            n     = len(volumes)
            avg_v = sum(volumes[max(0,n-20):n]) / min(20,n) if volumes else 1.0
            vol_s = round((volumes[-1] if volumes else avg_v) / avg_v, 2) if avg_v else 1.0
            chg   = round((last - prev) / prev * 100, 2) if prev else 0.0

            bull = 0; bear = 0; sigs = []
            # EMA stack
            if last > ema9 > ema20: bull += 20; sigs.append("EMA bullish stack")
            elif last < ema9 < ema20: bear += 20; sigs.append("EMA bearish stack")
            if ema20 > ema50: bull += 15; sigs.append("Uptrend EMA20>50")
            elif ema20 < ema50: bear += 15; sigs.append("Downtrend EMA20<50")
            # RSI
            if 55 <= rsi <= 78: bull += 15; sigs.append(f"RSI {rsi:.0f} bullish")
            elif 22 <= rsi <= 45: bear += 15; sigs.append(f"RSI {rsi:.0f} bearish")
            elif rsi > 78: bull += 5
            elif rsi < 22: bear += 5
            # MACD
            if hist > 0 and m_val > 0: bull += 20; sigs.append("MACD above zero+signal")
            elif hist < 0 and m_val < 0: bear += 20; sigs.append("MACD below zero+signal")
            elif hist > 0: bull += 10; sigs.append("MACD bullish cross")
            elif hist < 0: bear += 10; sigs.append("MACD bearish cross")
            # Bollinger Bands
            bb_rng = bb_up - bb_lo
            if bb_rng > 0:
                bb_pos = (last - bb_lo) / bb_rng
                if bb_pos > 0.85: bull += 15; sigs.append("Near upper BB breakout")
                elif bb_pos < 0.15: bear += 15; sigs.append("Near lower BB breakdown")
                elif bb_pos > 0.6: bull += 5
                elif bb_pos < 0.4: bear += 5
            # Volume surge
            if vol_s >= 1.5:
                if chg > 0: bull += 15; sigs.append(f"Volume surge {vol_s:.1f}x ↑")
                else: bear += 15; sigs.append(f"Volume surge {vol_s:.1f}x ↓")
            elif vol_s >= 1.2:
                if chg > 0: bull += 7
                else: bear += 7
            # Price momentum
            if chg >= 1.5: bull += 10; sigs.append(f"Strong day +{chg:.1f}%")
            elif chg <= -1.5: bear += 10; sigs.append(f"Sell-off {chg:.1f}%")
            elif chg > 0.3: bull += 4
            elif chg < -0.3: bear += 4

            net = bull - bear
            if net >= 35:
                action = "BUY_CE"; conf = round(min(net / 90, 0.95), 2)
                key = [s for s in sigs if any(w in s.lower() for w in ("bull","upper","above","strong","uptrend"))]
            elif net <= -35:
                action = "BUY_PE"; conf = round(min(-net / 90, 0.95), 2)
                key = [s for s in sigs if any(w in s.lower() for w in ("bear","lower","below","sell","downtrend","surge"))]
            else:
                return None  # not trending strongly enough

            step  = (100 if last > 10000 else 50 if last > 3000 else
                     20  if last > 800  else 10  if last > 200  else 5)
            atm   = round(last / step) * step
            strike = atm + step if action == "BUY_CE" else atm - step

            return {
                "symbol": sym, "action": action, "direction": "BULLISH" if action == "BUY_CE" else "BEARISH",
                "confidence": conf, "score": conf,
                "ltp": round(last, 2), "change_pct": chg,
                "rsi": rsi, "macd_hist": round(hist, 3), "vol_surge": vol_s,
                "ema9": round(ema9,2), "ema20": round(ema20,2), "ema50": round(ema50,2),
                "bb_upper": bb_up, "bb_lower": bb_lo, "atr": atr,
                "bull_score": bull, "bear_score": bear,
                "strike": strike, "premium_est": round(atr * 0.35, 0),
                "target_pct": 50, "sl_pct": 25, "expiry": "weekly",
                "signals": (key or sigs)[:3],
            }

        # ── Fetch all NIFTY 50 stocks concurrently ───────────────────────
        sem = asyncio.Semaphore(12)

        async def _fetch(client, sym):
            async with sem:
                try:
                    r = await client.get(
                        f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}.NS",
                        params={"interval": "1d", "range": "4mo"},
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=9,
                    )
                    if r.status_code != 200:
                        return None
                    res = r.json().get("chart", {}).get("result", [])
                    if not res:
                        return None
                    meta = res[0].get("meta", {})
                    q    = res[0].get("indicators", {}).get("quote", [{}])[0]
                    def _clean(arr): return [x for x in (arr or []) if x is not None]
                    closes  = _clean(q.get("close"))
                    highs   = _clean(q.get("high"))
                    lows    = _clean(q.get("low"))
                    volumes = _clean(q.get("volume"))
                    if len(closes) < 50:
                        return None
                    last = meta.get("regularMarketPrice") or closes[-1]
                    prev = meta.get("chartPreviousClose") or closes[-2]
                    return _analyze(sym, last, prev, closes, highs, lows, volumes)
                except Exception:
                    return None

        async with _httpx.AsyncClient() as client:
            raw = await asyncio.gather(*[_fetch(client, s) for s in UNIVERSE])

        picks = sorted([p for p in raw if p], key=lambda p: p["score"], reverse=True)
        result = {
            "top_picks": picks[:12], "picks": picks[:12],
            "scanned_at": int(time.time()),
            "symbols_scanned": len(UNIVERSE),
            "trending_count": len(picks),
        }
        try:
            import redis.asyncio as aioredis, json as _j
            r = await aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
            await r.setex("scanner:latest", 900, _j.dumps(result))
            await r.aclose()
        except Exception:
            pass
        log.info("scanner.full.complete", total=len(UNIVERSE), picks=len(picks))
        return result

    @router.get("/api/ohlcv/{symbol}")
    async def get_ohlcv(symbol: str, days: int = 90):
        """OHLCV candlestick data for Lightweight Charts via Yahoo Finance v8 API."""
        import httpx as _httpx
        ticker = symbol.upper()
        _range = "1y" if days >= 200 else "6mo" if days >= 120 else "3mo"
        _hdrs = {"User-Agent": "Mozilla/5.0"}
        _base = "https://query1.finance.yahoo.com/v8/finance/chart"

        candles = []
        async with _httpx.AsyncClient(timeout=15) as client:
            for suffix in [".NS", ".BO", ""]:
                try:
                    resp = await client.get(
                        f"{_base}/{ticker}{suffix}",
                        params={"interval": "1d", "range": _range},
                        headers=_hdrs,
                    )
                    if resp.status_code != 200:
                        continue
                    data = resp.json().get("chart", {})
                    result = data.get("result", [])
                    if not result:
                        continue
                    timestamps = result[0].get("timestamp", [])
                    q = result[0].get("indicators", {}).get("quote", [{}])[0]
                    opens   = q.get("open",   [None] * len(timestamps))
                    highs   = q.get("high",   [None] * len(timestamps))
                    lows    = q.get("low",    [None] * len(timestamps))
                    closes  = q.get("close",  [None] * len(timestamps))
                    volumes = q.get("volume", [None] * len(timestamps))
                    for i, ts in enumerate(timestamps):
                        o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]
                        if c is None:
                            continue
                        candles.append({
                            "time": int(ts),
                            "open":   round(float(o or c), 2),
                            "high":   round(float(h or c), 2),
                            "low":    round(float(l or c), 2),
                            "close":  round(float(c), 2),
                            "volume": int(v or 0),
                        })
                    if candles:
                        break
                except Exception:
                    continue

        if not candles:
            return {"symbol": ticker, "candles": [], "error": "No data available"}
        return {"symbol": ticker, "candles": candles}

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

    async def _redis_set(key: str, value, ttl: int = 86400):
        """Safely set a Redis key with TTL."""
        try:
            import redis.asyncio as aioredis, json as _j
            r = await aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
            await r.setex(key, ttl, _j.dumps(value))
            await r.aclose()
            return True
        except Exception:
            return False

    def _kite_token_from_request(request: Request) -> str | None:
        """Read Kite access token from secure cookie (primary) or Redis fallback."""
        return request.cookies.get("kite_token") or None

    def _kite_headers(access_token: str) -> dict:
        return {
            "Authorization": f"token {settings.zerodha_api_key}:{access_token}",
            "X-Kite-Version": "3",
        }

    # ── Zerodha OAuth endpoints ─────────────────────────────────────────

    @router.get("/broker/zerodha/login")
    async def zerodha_login():
        """Redirect browser to Zerodha Kite login page."""
        from fastapi.responses import RedirectResponse
        api_key = settings.zerodha_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="ZERODHA_API_KEY not configured")
        url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
        return RedirectResponse(url)

    @router.get("/broker/zerodha/callback")
    async def zerodha_callback(request_token: str = "", status: str = "", error: str = ""):
        """Kite redirects here after login. Exchange request_token for access_token."""
        from fastapi.responses import HTMLResponse, Response
        import hashlib, httpx as _httpx

        if status != "success" or not request_token:
            return HTMLResponse(_broker_page("Login Failed",
                f"Kite returned: status={status} error={error}", ok=False))

        api_key    = settings.zerodha_api_key or ""
        api_secret = settings.zerodha_api_secret or ""
        checksum   = hashlib.sha256(f"{api_key}{request_token}{api_secret}".encode()).hexdigest()

        try:
            async with _httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.kite.trade/session/token",
                    data={"api_key": api_key, "request_token": request_token, "checksum": checksum},
                    headers={"X-Kite-Version": "3"},
                )
            if resp.status_code != 200:
                return HTMLResponse(_broker_page(
                    "Token Error", f"Kite error {resp.status_code}: {resp.text[:300]}", ok=False))

            data         = resp.json().get("data", {})
            access_token = data.get("access_token", "")
            user_name    = data.get("user_name", "")
            if not access_token:
                return HTMLResponse(_broker_page("Token Error", "No access_token in Kite response", ok=False))

            # Also cache in Redis if available (best-effort)
            await _redis_set("zerodha:access_token", access_token, ttl=86400)
            await _redis_set("zerodha:user_info",
                             {"user_name": user_name, "login_time": int(time.time())}, ttl=86400)

            log.info("zerodha.oauth.success", user=user_name)

            # Primary storage: secure HTTP-only cookie valid until tonight midnight
            html_resp = HTMLResponse(
                _broker_page("Connected!", f"Welcome {user_name}! Zerodha is now live.", ok=True))
            html_resp.set_cookie(
                key="kite_token",   value=access_token,
                max_age=86400,      httponly=True,
                secure=True,        samesite="lax",
                path="/",
            )
            html_resp.set_cookie(
                key="kite_user",    value=user_name,
                max_age=86400,      httponly=False,   # readable by JS for display
                secure=True,        samesite="lax",
                path="/",
            )
            return html_resp

        except Exception as e:
            log.error("zerodha.callback.error", error=str(e))
            return HTMLResponse(_broker_page("Error", str(e), ok=False))

    @router.get("/broker/zerodha/status")
    async def zerodha_status(request: Request):
        """Check whether a valid Kite access token is in the cookie."""
        import httpx as _httpx
        token = _kite_token_from_request(request)

        # Fallback: try Redis
        if not token:
            token = await _redis_get("zerodha:access_token")
            if token and not isinstance(token, str):
                token = None

        if not token:
            return {"connected": False, "message": "Not logged in"}

        # Quick verify against Kite profile endpoint
        try:
            async with _httpx.AsyncClient(timeout=8) as client:
                r = await client.get("https://api.kite.trade/user/profile",
                                     headers=_kite_headers(token))
            if r.status_code == 200:
                user_name = (r.json().get("data", {}).get("user_name", "")
                             or request.cookies.get("kite_user", ""))
                return {"connected": True, "user_name": user_name}
            # Token invalid — tell JS to clear it
            return {"connected": False, "expired": True,
                    "message": f"Token expired (HTTP {r.status_code}). Please re-login."}
        except Exception as e:
            return {"connected": False, "message": str(e)}

    @router.get("/broker/zerodha/logout")
    async def zerodha_logout():
        """Clear the Kite cookie and Redis entry."""
        from fastapi.responses import JSONResponse
        resp = JSONResponse({"message": "Logged out"})
        resp.delete_cookie("kite_token", path="/")
        resp.delete_cookie("kite_user",  path="/")
        # Best-effort Redis clear
        try:
            import redis.asyncio as aioredis
            r = await aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
            await r.delete("zerodha:access_token", "zerodha:user_info")
            await r.aclose()
        except Exception:
            pass
        return resp

    def _broker_page(title: str, msg: str, ok: bool) -> str:
        color = "#3fb950" if ok else "#f85149"
        icon  = "✅" if ok else "❌"
        redirect = '<meta http-equiv="refresh" content="3;url=/">' if ok else ""
        return f"""<!doctype html><html><head><meta charset="utf-8">{redirect}
        <style>body{{background:#0d1117;color:#e6edf3;font-family:sans-serif;display:flex;
        align-items:center;justify-content:center;height:100vh;margin:0}}
        .box{{text-align:center;max-width:400px;padding:40px;background:#161b22;
        border:1px solid #30363d;border-radius:12px}}
        h2{{color:{color};margin-bottom:12px}} p{{color:#8b949e;margin-bottom:20px}}
        a{{color:#58a6ff;text-decoration:none}}</style></head>
        <body><div class="box"><div style="font-size:48px">{icon}</div>
        <h2>{title}</h2><p>{msg}</p>
        {"<p style='color:#8b949e;font-size:0.85rem'>Redirecting to dashboard…</p>" if ok else
         '<a href="/broker/zerodha/login">↩ Try Again</a>'}</div></body></html>"""

    @router.get("/api/ai/signals")
    async def get_ai_signals():
        """Latest AI options signals from background analysis."""
        data = await _redis_get("ai:options:latest")
        if data:
            return data
        return {"signals": [], "message": "AI analysis runs every 30 min during market hours. Connect Redis to persist signals."}

    @router.get("/api/market-context")
    async def get_market_context(request: Request):
        """Live market context — Kite LTP if connected, else Yahoo Finance v8 API."""
        import asyncio, httpx as _httpx

        _hdrs  = {"User-Agent": "Mozilla/5.0"}
        _base  = "https://query2.finance.yahoo.com/v8/finance/chart"
        _ticks = {
            "nifty":     "^NSEI",
            "banknifty": "^NSEBANK",
            "niftyit":   "^CNXIT",
            "vix":       "^INDIAVIX",
            "sensex":    "^BSESN",
            "sp500":     "^GSPC",
            "nasdaq":    "^IXIC",
            "usdinr":    "INR=X",
            "crude":     "CL=F",
        }

        async def _fetch_one(client, name, sym):
            try:
                r = await client.get(f"{_base}/{sym}",
                                     params={"interval": "1d", "range": "5d"},
                                     headers=_hdrs, timeout=10)
                if r.status_code != 200:
                    return name, 0.0, 0.0
                res = r.json().get("chart", {}).get("result", [])
                if not res:
                    return name, 0.0, 0.0
                meta = res[0].get("meta", {})
                # Use regularMarketPrice from meta — always reflects the latest
                # trade/close (historical closes[] lag by one candle on Yahoo)
                last = meta.get("regularMarketPrice") or 0.0
                prev = (meta.get("chartPreviousClose")
                        or meta.get("previousClose") or last)
                if not last:
                    # Fallback to historical closes if meta is empty
                    q = res[0].get("indicators", {}).get("quote", [{}])[0]
                    closes = [x for x in (q.get("close") or []) if x is not None]
                    if not closes:
                        return name, 0.0, 0.0
                    last = closes[-1]
                    prev = closes[-2] if len(closes) > 1 else last
                pct = round((last - prev) / prev * 100, 2) if prev else 0.0
                return name, round(last, 2), pct
            except Exception:
                return name, 0.0, 0.0

        try:
            # Attempt Kite LTP for Indian indices (zero delay, real-time)
            kite_token = _kite_token_from_request(request) or await _redis_get("zerodha:access_token")
            kite_nifty = kite_vix = kite_sensex = None
            if kite_token:
                try:
                    async with _httpx.AsyncClient(timeout=6) as kc:
                        kr = await kc.get(
                            "https://api.kite.trade/quote/ltp",
                            params={"i": ["NSE:NIFTY 50", "NSE:INDIA VIX", "BSE:SENSEX"]},
                            headers=_kite_headers(kite_token),
                        )
                    if kr.status_code == 200:
                        kd = kr.json().get("data", {})
                        kite_nifty  = kd.get("NSE:NIFTY 50",   {}).get("last_price")
                        kite_vix    = kd.get("NSE:INDIA VIX",  {}).get("last_price")
                        kite_sensex = kd.get("BSE:SENSEX",     {}).get("last_price")
                except Exception:
                    pass

            async with _httpx.AsyncClient() as client:
                tasks = [_fetch_one(client, n, s) for n, s in _ticks.items()]
                raw = await asyncio.gather(*tasks)

            results = {name: {"last": last, "chg_pct": pct} for name, last, pct in raw}

            # Override with real-time Kite values where available
            if kite_nifty:
                prev = results.get("nifty", {}).get("last", kite_nifty)
                results["nifty"]["last"] = kite_nifty
                results["nifty"]["chg_pct"] = round((kite_nifty - prev) / prev * 100, 2) if prev else 0
                results["nifty"]["realtime"] = True
            if kite_vix:
                results["vix"]["last"] = kite_vix
                results["vix"]["realtime"] = True
            if kite_sensex:
                results["sensex"]["last"] = kite_sensex
                results["sensex"]["realtime"] = True

            n   = results.get("nifty",     {})
            bn  = results.get("banknifty", {})
            ni  = results.get("niftyit",   {})
            v   = results.get("vix",       {})
            sp  = results.get("sp500",     {})
            nq  = results.get("nasdaq",    {})
            fx  = results.get("usdinr",    {})
            cl  = results.get("crude",     {})

            vix_val = v.get("last", 15.0)
            n_chg   = n.get("chg_pct", 0.0)
            sp_chg  = sp.get("chg_pct", 0.0)

            bull_pts = sum([n_chg > 0.3, vix_val < 15, sp_chg > 0.2, nq.get("chg_pct", 0) > 0.2])
            bear_pts = sum([n_chg < -0.3, vix_val > 20, sp_chg < -0.2])
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
                "gift_nifty_change_pct": n_chg,
                "banknifty_last":     bn.get("last", 0),
                "banknifty_chg_pct":  bn.get("chg_pct", 0),
                "niftyit_last":       ni.get("last", 0),
                "niftyit_chg_pct":    ni.get("chg_pct", 0),
                "sensex_last":        results.get("sensex", {}).get("last", 0),
                "nifty_realtime": results.get("nifty", {}).get("realtime", False),
                "vix_realtime":   results.get("vix",   {}).get("realtime", False),
                "fii_net_crore": None,
                "dii_net_crore": None,
                "score": round((bull_pts - bear_pts) / 4.0, 2),
                "summary": (
                    f"NIFTY {n.get('last', 0):.0f} ({n_chg:+.2f}%) | "
                    f"VIX {vix_val:.1f} | "
                    f"S&P {sp_chg:+.2f}% | "
                    f"USD/INR {fx.get('last', 0):.2f} | "
                    f"Crude ${cl.get('last', 0):.1f}"
                ),
            }
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
