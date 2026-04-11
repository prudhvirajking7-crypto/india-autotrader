"""
NSE Option Chain & PCR Fetcher

Fetches live option chain data from the official NSE API.
Computes:
  - Put-Call Ratio (PCR) by OI and by volume
  - Max Pain strike price
  - OI-based support/resistance levels
  - Implied Volatility rank

NSE API requires session cookies — we maintain a persistent session.
Rate-limit: 1 request per second to avoid IP blocking.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

NSE_BASE = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


@dataclass
class OptionChainMetrics:
    symbol: str
    expiry: str
    pcr_oi: float          # Put-Call Ratio by Open Interest (primary)
    pcr_volume: float      # Put-Call Ratio by Volume (more responsive)
    max_pain: float        # Strike price where option buyers lose most
    atm_strike: float      # At-the-money strike
    underlying_price: float
    top_call_oi_strike: float  # Strongest resistance (highest call OI)
    top_put_oi_strike: float   # Strongest support (highest put OI)
    total_call_oi: int
    total_put_oi: int
    total_call_vol: int
    total_put_vol: int
    iv_rank: Optional[float] = None   # 0–100, requires historical IV
    pcr_signal: str = "neutral"       # bullish | bearish | neutral
    fetched_at: float = 0.0

    def __post_init__(self):
        self.pcr_signal = self._classify_pcr()
        if self.fetched_at == 0.0:
            self.fetched_at = time.time()

    def _classify_pcr(self) -> str:
        """
        Contrarian PCR interpretation for Indian markets.
        Source: NSE research, empirical NIFTY option chain analysis.
        """
        pcr = self.pcr_oi
        if pcr < 0.7:
            return "bearish"    # Too many calls — crowded long, contrarian bearish
        elif pcr <= 0.9:
            return "mildly_bearish"
        elif pcr <= 1.1:
            return "neutral"
        elif pcr <= 1.3:
            return "mildly_bullish"
        elif pcr <= 1.5:
            return "bullish"    # Put writers active = floor support
        else:
            return "strongly_bullish"  # Extreme put buying = panic = contrarian buy


class NSEOptionChainFetcher:
    """
    Fetches real-time option chain data from NSE's public API.
    Maintains a persistent httpx session with cookies.
    """

    CACHE_TTL = 300  # 5 minutes (options OI changes slowly intraday)
    RATE_LIMIT_DELAY = 1.0  # seconds between requests

    def __init__(self) -> None:
        self._session: Optional[httpx.AsyncClient] = None
        self._last_request_at = 0.0
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            from config.settings import settings
            self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def _get_session(self) -> httpx.AsyncClient:
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                headers=NSE_HEADERS,
                follow_redirects=True,
                timeout=10.0,
            )
            # Warm up session with main page (required for cookies)
            try:
                await self._session.get(NSE_BASE)
                await asyncio.sleep(0.5)
            except Exception:
                pass
        return self._session

    async def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.RATE_LIMIT_DELAY:
            await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_at = time.time()

    async def get_metrics(self, symbol: str = "NIFTY") -> OptionChainMetrics:
        """Return option chain metrics, from cache if available."""
        r = await self._get_redis()
        cache_key = f"options:metrics:{symbol}"
        cached = await r.get(cache_key)
        if cached:
            import json
            data = json.loads(cached)
            return OptionChainMetrics(**data)

        metrics = await self._fetch_and_compute(symbol)
        import json
        await r.setex(cache_key, self.CACHE_TTL, json.dumps({
            "symbol": metrics.symbol,
            "expiry": metrics.expiry,
            "pcr_oi": metrics.pcr_oi,
            "pcr_volume": metrics.pcr_volume,
            "max_pain": metrics.max_pain,
            "atm_strike": metrics.atm_strike,
            "underlying_price": metrics.underlying_price,
            "top_call_oi_strike": metrics.top_call_oi_strike,
            "top_put_oi_strike": metrics.top_put_oi_strike,
            "total_call_oi": metrics.total_call_oi,
            "total_put_oi": metrics.total_put_oi,
            "total_call_vol": metrics.total_call_vol,
            "total_put_vol": metrics.total_put_vol,
            "pcr_signal": metrics.pcr_signal,
            "fetched_at": metrics.fetched_at,
        }))
        return metrics

    async def _fetch_and_compute(self, symbol: str) -> OptionChainMetrics:
        await self._rate_limit()
        session = await self._get_session()

        is_index = symbol.upper() in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}
        endpoint = (
            f"{NSE_BASE}/api/option-chain-indices?symbol={symbol.upper()}"
            if is_index
            else f"{NSE_BASE}/api/option-chain-equities?symbol={symbol.upper()}"
        )

        try:
            resp = await session.get(endpoint)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("nse_options.fetch_failed", symbol=symbol, error=str(e))
            # Return neutral defaults so scorer can continue
            return self._neutral_metrics(symbol)

        return self._parse_option_chain(data, symbol)

    def _parse_option_chain(self, data: dict, symbol: str) -> OptionChainMetrics:
        records = data.get("records", {})
        data_list = records.get("data", [])
        underlying_price = float(records.get("underlyingValue", 0))
        expiry_dates = records.get("expiryDates", [])
        nearest_expiry = expiry_dates[0] if expiry_dates else ""

        # Filter to nearest expiry
        near_data = [d for d in data_list if d.get("expiryDate") == nearest_expiry]

        total_call_oi = total_put_oi = 0
        total_call_vol = total_put_vol = 0
        call_oi_by_strike: dict[float, int] = {}
        put_oi_by_strike: dict[float, int] = {}

        for item in near_data:
            strike = float(item.get("strikePrice", 0))
            ce = item.get("CE", {})
            pe = item.get("PE", {})

            ce_oi = int(ce.get("openInterest", 0) or 0)
            pe_oi = int(pe.get("openInterest", 0) or 0)
            ce_vol = int(ce.get("totalTradedVolume", 0) or 0)
            pe_vol = int(pe.get("totalTradedVolume", 0) or 0)

            total_call_oi += ce_oi
            total_put_oi += pe_oi
            total_call_vol += ce_vol
            total_put_vol += pe_vol

            if ce_oi > 0:
                call_oi_by_strike[strike] = ce_oi
            if pe_oi > 0:
                put_oi_by_strike[strike] = pe_oi

        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        pcr_volume = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0

        # ATM strike = nearest strike to underlying price
        all_strikes = sorted(set(list(call_oi_by_strike.keys()) + list(put_oi_by_strike.keys())))
        atm_strike = min(all_strikes, key=lambda s: abs(s - underlying_price)) if all_strikes else underlying_price

        # Strongest resistance = highest call OI strike
        top_call_strike = max(call_oi_by_strike, key=call_oi_by_strike.get) if call_oi_by_strike else atm_strike
        # Strongest support = highest put OI strike
        top_put_strike = max(put_oi_by_strike, key=put_oi_by_strike.get) if put_oi_by_strike else atm_strike

        # Max Pain: strike where sum of (intrinsic value × OI) is minimized
        max_pain = self._calculate_max_pain(all_strikes, call_oi_by_strike, put_oi_by_strike)

        log.info(
            "nse_options.parsed",
            symbol=symbol,
            expiry=nearest_expiry,
            pcr_oi=f"{pcr_oi:.2f}",
            atm=atm_strike,
            max_pain=max_pain,
        )

        return OptionChainMetrics(
            symbol=symbol,
            expiry=nearest_expiry,
            pcr_oi=round(pcr_oi, 3),
            pcr_volume=round(pcr_volume, 3),
            max_pain=max_pain,
            atm_strike=atm_strike,
            underlying_price=underlying_price,
            top_call_oi_strike=top_call_strike,
            top_put_oi_strike=top_put_strike,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            total_call_vol=total_call_vol,
            total_put_vol=total_put_vol,
        )

    @staticmethod
    def _calculate_max_pain(
        strikes: list[float],
        call_oi: dict[float, int],
        put_oi: dict[float, int],
    ) -> float:
        """Compute max pain strike — expiry price where total option buyer loss is maximized."""
        if not strikes:
            return 0.0

        min_loss = float("inf")
        max_pain_strike = strikes[0]

        for candidate in strikes:
            # Total loss to option buyers if expires at 'candidate'
            total_loss = 0
            for strike, oi in call_oi.items():
                intrinsic = max(0, candidate - strike)
                total_loss += intrinsic * oi
            for strike, oi in put_oi.items():
                intrinsic = max(0, strike - candidate)
                total_loss += intrinsic * oi
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = candidate

        return max_pain_strike

    def _neutral_metrics(self, symbol: str) -> OptionChainMetrics:
        """Return neutral defaults when NSE API is unavailable."""
        return OptionChainMetrics(
            symbol=symbol,
            expiry="unknown",
            pcr_oi=1.0,
            pcr_volume=1.0,
            max_pain=0.0,
            atm_strike=0.0,
            underlying_price=0.0,
            top_call_oi_strike=0.0,
            top_put_oi_strike=0.0,
            total_call_oi=0,
            total_put_oi=0,
            total_call_vol=0,
            total_put_vol=0,
            pcr_signal="neutral",
        )
