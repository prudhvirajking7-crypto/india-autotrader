"""
Global Market Context Tracker

Tracks macro signals that affect Indian markets:

  1. SGX NIFTY / Gift NIFTY futures    ← pre-market India direction
  2. US markets (S&P 500, Nasdaq)      ← overnight global cue
  3. India VIX                         ← fear gauge (>20 = risky)
  4. FII/DII activity                  ← institutional money flow (NSE data)
  5. USD/INR forex                     ← currency pressure on imports/exports
  6. Crude oil price                   ← inflation + energy sector impact
  7. Gold price                        ← risk-off indicator

Each factor contributes a +/- adjustment to the final signal score.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

log = structlog.get_logger(__name__)


@dataclass
class GlobalContext:
    # Pre-market cue
    gift_nifty_change_pct: float = 0.0       # >0 = gap-up, <0 = gap-down

    # US overnight
    sp500_change_pct: float = 0.0
    nasdaq_change_pct: float = 0.0
    dow_change_pct: float = 0.0

    # Volatility
    india_vix: float = 15.0                  # normal ~15, high risk >20, extreme >25
    vix_change_pct: float = 0.0

    # Forex & Commodities
    usdinr: float = 84.0                     # INR per USD
    crude_oil_usd: float = 75.0              # Brent crude $/barrel
    gold_usd: float = 2000.0

    # FII/DII (crore INR, positive = buying)
    fii_net_crore: float = 0.0
    dii_net_crore: float = 0.0

    # Computed
    market_bias: str = "neutral"             # bullish | bearish | neutral
    context_score: float = 0.0              # -1.0 to +1.0
    fetched_at: float = field(default_factory=time.time)
    data_quality: str = "unknown"           # live | cached | fallback


class MarketContextTracker:
    """
    Fetches global market context from free data sources.

    Sources used:
    - Yahoo Finance (via yfinance) for indices, forex, commodities
    - NSE India website for FII/DII provisional data
    - jugaad-data for India VIX
    """

    CACHE_TTL = 900  # 15 minutes

    def __init__(self) -> None:
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            from config.settings import settings
            self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def get_context(self) -> GlobalContext:
        """Return cached or fresh global market context."""
        r = await self._get_redis()
        cached = await r.get("market:context")
        if cached:
            import json
            data = json.loads(cached)
            ctx = GlobalContext(**data)
            ctx.data_quality = "cached"
            return ctx

        ctx = await self._fetch_context()
        import json
        await r.setex("market:context", self.CACHE_TTL, json.dumps({
            "gift_nifty_change_pct": ctx.gift_nifty_change_pct,
            "sp500_change_pct": ctx.sp500_change_pct,
            "nasdaq_change_pct": ctx.nasdaq_change_pct,
            "dow_change_pct": ctx.dow_change_pct,
            "india_vix": ctx.india_vix,
            "vix_change_pct": ctx.vix_change_pct,
            "usdinr": ctx.usdinr,
            "crude_oil_usd": ctx.crude_oil_usd,
            "gold_usd": ctx.gold_usd,
            "fii_net_crore": ctx.fii_net_crore,
            "dii_net_crore": ctx.dii_net_crore,
            "market_bias": ctx.market_bias,
            "context_score": ctx.context_score,
            "fetched_at": ctx.fetched_at,
            "data_quality": ctx.data_quality,
        }))
        return ctx

    async def _fetch_context(self) -> GlobalContext:
        """Fetch all global context data concurrently."""
        import asyncio

        ctx = GlobalContext()

        results = await asyncio.gather(
            self._fetch_yfinance_data(ctx),
            self._fetch_fii_dii(ctx),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                log.warning("market_context.fetch_partial_failure", error=str(r))

        ctx.context_score = self._compute_score(ctx)
        ctx.market_bias = (
            "bullish" if ctx.context_score > 0.15
            else "bearish" if ctx.context_score < -0.15
            else "neutral"
        )
        ctx.data_quality = "live"
        log.info(
            "market_context.fetched",
            bias=ctx.market_bias,
            score=f"{ctx.context_score:.2f}",
            vix=ctx.india_vix,
            fii=ctx.fii_net_crore,
        )
        return ctx

    async def _fetch_yfinance_data(self, ctx: GlobalContext) -> None:
        """Fetch US indices, VIX, forex, commodities via yfinance."""
        import asyncio
        loop = asyncio.get_event_loop()

        def _fetch():
            import yfinance as yf
            tickers = yf.download(
                tickers=["^GSPC", "^IXIC", "^DJI", "^INDIAVIX", "USDINR=X", "BZ=F", "GC=F", "NIFTY50.NS"],
                period="2d",
                interval="1d",
                progress=False,
                group_by="ticker",
            )
            return tickers

        try:
            data = await loop.run_in_executor(None, _fetch)

            def _pct_change(ticker: str) -> float:
                try:
                    closes = data[ticker]["Close"].dropna()
                    if len(closes) >= 2:
                        return float((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100)
                except Exception:
                    pass
                return 0.0

            def _latest(ticker: str) -> float:
                try:
                    return float(data[ticker]["Close"].dropna().iloc[-1])
                except Exception:
                    return 0.0

            ctx.sp500_change_pct = _pct_change("^GSPC")
            ctx.nasdaq_change_pct = _pct_change("^IXIC")
            ctx.dow_change_pct = _pct_change("^DJI")
            ctx.india_vix = _latest("^INDIAVIX")
            ctx.vix_change_pct = _pct_change("^INDIAVIX")
            ctx.usdinr = _latest("USDINR=X")
            ctx.crude_oil_usd = _latest("BZ=F")
            ctx.gold_usd = _latest("GC=F")
            ctx.gift_nifty_change_pct = _pct_change("NIFTY50.NS")

        except Exception as e:
            log.warning("market_context.yfinance_failed", error=str(e))

    async def _fetch_fii_dii(self, ctx: GlobalContext) -> None:
        """
        Fetch FII/DII provisional net buy/sell from NSE.
        NSE publishes this at ~17:00 IST each trading day.
        """
        try:
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            async with httpx.AsyncClient(
                timeout=8.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                    "Referer": "https://www.nseindia.com/",
                },
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

            for entry in data:
                category = entry.get("category", "").lower()
                net = float(entry.get("netPurchases", 0) or 0)
                if "fii" in category or "fpi" in category:
                    ctx.fii_net_crore = net
                elif "dii" in category:
                    ctx.dii_net_crore = net

        except Exception as e:
            log.warning("market_context.fii_dii_failed", error=str(e))

    def _compute_score(self, ctx: GlobalContext) -> float:
        """
        Convert raw market context into a -1.0 to +1.0 score.

        Weights:
          - SGX/Gift NIFTY:  30%  (most direct India pre-market signal)
          - US markets avg:  25%  (global risk-on/off)
          - India VIX:       20%  (inverse — high VIX = bearish penalty)
          - FII/DII flow:    15%  (institutional conviction)
          - Crude oil:        5%  (inverse for India — high oil = bearish)
          - USD/INR:          5%  (inverse — weak INR = bearish)
        """
        score = 0.0

        # Gift NIFTY pre-market cue (max ±0.30)
        gift_contribution = max(-0.30, min(0.30, ctx.gift_nifty_change_pct / 2.0))
        score += gift_contribution

        # US markets average (max ±0.25)
        us_avg = (ctx.sp500_change_pct + ctx.nasdaq_change_pct + ctx.dow_change_pct) / 3
        us_contribution = max(-0.25, min(0.25, us_avg / 2.0))
        score += us_contribution

        # VIX penalty (high VIX = bearish, normal = IndiaVIXLevel.NORMAL_LOW)
        from intelligence.knowledge_base import IndiaVIXLevel
        if ctx.india_vix > 0:
            vix_penalty = max(-0.20, -(ctx.india_vix - IndiaVIXLevel.NORMAL_LOW) / 50)
            score += vix_penalty

        # FII/DII net flow (max ±0.15)
        combined_flow = ctx.fii_net_crore + ctx.dii_net_crore
        flow_contribution = max(-0.15, min(0.15, combined_flow / 10000))
        score += flow_contribution

        # Crude oil (inverse signal for India, max ±0.05)
        crude_change = (ctx.crude_oil_usd - 75) / 75 * 100  # % from neutral
        score -= max(-0.05, min(0.05, crude_change / 40))

        # USD/INR (weak INR = bearish, max ±0.05)
        inr_change = (ctx.usdinr - 84) / 84 * 100
        score -= max(-0.05, min(0.05, inr_change / 10))

        return round(max(-1.0, min(1.0, score)), 3)

    def summarize(self, ctx: GlobalContext) -> str:
        """Human-readable context summary."""
        lines = [
            f"Market Bias: {ctx.market_bias.upper()} (score: {ctx.context_score:+.2f})",
            f"Gift NIFTY: {ctx.gift_nifty_change_pct:+.2f}%",
            f"S&P 500: {ctx.sp500_change_pct:+.2f}% | Nasdaq: {ctx.nasdaq_change_pct:+.2f}%",
            f"India VIX: {ctx.india_vix:.1f} ({ctx.vix_change_pct:+.1f}%)",
            f"FII: ₹{ctx.fii_net_crore:+,.0f}Cr | DII: ₹{ctx.dii_net_crore:+,.0f}Cr",
            f"USD/INR: {ctx.usdinr:.2f} | Crude: ${ctx.crude_oil_usd:.1f}",
        ]
        return "\n".join(lines)
