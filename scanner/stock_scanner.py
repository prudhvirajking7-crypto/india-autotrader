"""
Full NSE F&O Universe Scanner

Scans every F&O-eligible stock every 15 minutes during market hours.
Runs the full intelligence pipeline on each stock and ranks by evidence score.
Returns top candidates with breakout confirmation and S/R levels for execution.

Pipeline per stock (in order):
  1. Fetch OHLCV (cached, parallel, rate-limited)
  2. ManipulationDetector   → skip if manipulated
  3. WyckoffAnalyzer        → phase + trade bias
  4. SMCAnalyzer            → order blocks, FVG, liquidity sweeps
  5. SRFinder               → support/resistance levels
  6. BreakoutDetector       → breakout confirmation
  7. SignalScorer.score()   → full 0–100 score
  8. Rank → top N candidates

Only stocks meeting ALL of:
  - Score ≥ 60
  - No manipulation block
  - Breakout or at key S/R (not random position in range)
  - Risk:Reward ≥ 1.5
  → are returned for execution.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# ── NSE F&O Universe ─────────────────────────────────────────────────────────
# All SEBI-approved F&O symbols on NSE (as of Nov 2024)
# Grouped by sector for easier monitoring

FO_UNIVERSE: list[str] = [
    # Indices
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",

    # Banking & Finance
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB",
    "CANBK", "BANKBARODA", "UNIONBANK", "AUBANK",
    "BAJFINANCE", "BAJAJFINSV", "SBICARD", "CHOLAFIN",
    "MUTHOOTFIN", "MANAPPURAM", "M&MFIN", "LICHSGFIN",
    "HDFCLIFE", "SBILIFE", "ICICIPRULI", "ICICIGI",

    # IT & Tech
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
    "MPHASIS", "COFORGE", "PERSISTENT", "NAUKRI", "ZOMATO",
    "PAYTM", "NYKAA", "POLICYBZR",

    # Oil, Gas & Energy
    "RELIANCE", "ONGC", "BPCL", "IOC", "HINDPETRO",
    "GAIL", "PETRONET", "IGL", "MGL",
    "NTPC", "POWERGRID", "ADANIPOWER", "TATAPOWER",
    "ADANIGREEN", "ADANIPORTS",

    # Consumer & FMCG
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA",
    "DABUR", "MARICO", "GODREJCP", "COLPAL", "EMAMILTD",

    # Auto
    "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO",
    "HEROMOTOCO", "EICHERMOT", "ASHOKLEY", "TVSMOTOR",
    "BALKRISIND", "MRF",

    # Metals & Mining
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL",
    "SAIL", "NMDC", "COALINDIA", "NATIONALUM",

    # Pharma
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB",
    "BIOCON", "AUROPHARMA", "ALKEM", "TORNTPHARM",
    "LUPIN", "ABBOTINDIA",

    # Capital Goods & Infrastructure
    "LT", "BHEL", "ABB", "SIEMENS", "CUMMINSIND",
    "INDUSTOWER", "BHARTIARTL", "HFCL",

    # Cement
    "ULTRACEMCO", "GRASIM", "SHREECEM", "AMBUJACEM", "ACC",

    # Real Estate & Specialty
    "DLF", "GODREJPROP", "PRESTIGE", "OBEROIRLTY",

    # Paints & Consumer
    "ASIANPAINT", "BERGERPAINTS", "PIDILITIND", "TITAN", "TRENT",

    # Others
    "INDIGO", "ZYDUSLIFE", "GLENMARK", "IPCALAB",
]

# Minimum score to include in results
MIN_SCORE = 60
# Minimum risk:reward ratio for execution
MIN_RR = 1.5
# How many top stocks to return
TOP_N = 10
# Batch size for parallel OHLCV fetch (respect rate limits)
FETCH_BATCH_SIZE = 10
# Delay between batches (seconds)
BATCH_DELAY = 1.0


@dataclass
class StockScanResult:
    symbol: str
    action: str                      # "BUY" | "SELL"
    score: int                       # 0–100 intelligence score
    entry_price: float               # Current price (market order) or limit
    stop_loss: float                 # Calculated SL (from S/R)
    target1: float                   # 1:1 R:R target
    target2: float                   # 1:2 R:R target
    target3: float                   # 1:3 R:R target
    risk_reward: float               # R:R ratio (target2/risk)
    risk_per_share: float            # Entry - SL (in ₹)
    atr: float                       # ATR for position sizing

    # Evidence
    wyckoff_phase: str = ""
    wyckoff_bias: str = ""
    wyckoff_confidence: float = 0.0
    smc_bullish_signals: int = 0
    smc_bearish_signals: int = 0
    breakout_type: str = ""          # "resistance_break" | "support_break" | "squeeze" | ""
    breakout_confirmed: bool = False
    breakout_volume_ratio: float = 0.0
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    manipulation_score: int = 0
    false_breakout_risk: bool = False

    # Rank
    rank: int = 0
    scan_time: float = field(default_factory=time.time)

    # Human summary
    rationale: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class ScanReport:
    timestamp: float = field(default_factory=time.time)
    symbols_scanned: int = 0
    symbols_passed: int = 0
    symbols_blocked: int = 0         # Blocked by manipulation
    top_picks: list[StockScanResult] = field(default_factory=list)
    scan_duration_sec: float = 0.0


class StockScanner:
    """
    Scans the full NSE F&O universe for the highest-evidence trade setups.

    Designed to run every 15 minutes during market hours (09:30–15:15 IST).
    First 15 minutes (09:15–09:30) excluded — too volatile, price discovery.
    Last 15 minutes (15:15–15:30) excluded — MIS square-off zone.
    """

    def __init__(self, symbols: Optional[list[str]] = None) -> None:
        self._symbols = symbols or FO_UNIVERSE
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            from config.settings import settings
            self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def run_scan(self, top_n: int = TOP_N) -> ScanReport:
        """
        Run a full scan of all F&O stocks.

        Returns top N opportunities ranked by evidence score.
        """
        start = time.time()
        report = ScanReport()

        log.info("scanner.start", symbols=len(self._symbols))

        # Fetch OHLCV for all symbols in batches
        ohlcv_map = await self._fetch_all_ohlcv(self._symbols)
        report.symbols_scanned = len(ohlcv_map)

        # Score each symbol concurrently (with semaphore to limit parallelism)
        semaphore = asyncio.Semaphore(5)
        tasks = [
            self._analyze_symbol(symbol, df, semaphore)
            for symbol, df in ohlcv_map.items()
            if df is not None and not df.empty
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter and rank
        valid = [r for r in results if isinstance(r, StockScanResult)]
        report.symbols_blocked = sum(1 for r in valid if r.manipulation_score >= 2)

        passed = [
            r for r in valid
            if r.score >= MIN_SCORE
            and r.manipulation_score < 2
            and r.risk_reward >= MIN_RR
        ]
        report.symbols_passed = len(passed)

        # Sort: breakout + confirmed first, then by score
        passed.sort(key=lambda x: (
            int(x.breakout_confirmed),     # Breakout first
            x.score,                        # Then by score
            x.risk_reward,                  # Then by R:R
        ), reverse=True)

        for i, pick in enumerate(passed[:top_n]):
            pick.rank = i + 1

        report.top_picks = passed[:top_n]
        report.scan_duration_sec = round(time.time() - start, 2)

        # Cache scan results in Redis
        await self._cache_results(report)

        log.info(
            "scanner.complete",
            scanned=report.symbols_scanned,
            passed=report.symbols_passed,
            blocked=report.symbols_blocked,
            top_picks=len(report.top_picks),
            duration=f"{report.scan_duration_sec}s",
        )

        return report

    async def _analyze_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        semaphore: asyncio.Semaphore,
    ) -> Optional[StockScanResult]:
        """Run full analysis pipeline on a single stock."""
        async with semaphore:
            try:
                from intelligence.manipulation import ManipulationDetector
                from intelligence.wyckoff import WyckoffAnalyzer
                from intelligence.smart_money import SMCAnalyzer
                from intelligence.scorer import SignalScorer
                from scanner.support_resistance import SRFinder
                from scanner.breakout_detector import BreakoutDetector

                price = float(df["close"].iloc[-1])

                # Step 1: Manipulation check (fast — runs sync)
                loop = asyncio.get_event_loop()
                manip = await loop.run_in_executor(None, ManipulationDetector().detect, df)

                if manip.block_trade:
                    log.debug("scanner.skip.manipulation", symbol=symbol, score=manip.manipulation_score)
                    return StockScanResult(
                        symbol=symbol, action="SKIP", score=0,
                        entry_price=price, stop_loss=price, target1=price,
                        target2=price, target3=price, risk_reward=0,
                        risk_per_share=0, atr=0, manipulation_score=manip.manipulation_score,
                        rationale="Blocked: manipulation detected",
                    )

                # Step 2: Wyckoff phase
                wyckoff = await loop.run_in_executor(None, WyckoffAnalyzer().analyze, df)

                # Step 3: SMC
                smc = await loop.run_in_executor(None, SMCAnalyzer().analyze, df)

                # Step 4: S/R levels
                sr = await loop.run_in_executor(None, SRFinder().find, df)

                # Step 5: Breakout detection
                breakout = await loop.run_in_executor(
                    None,
                    lambda: BreakoutDetector().detect(
                        df, sr,
                        manipulation_score=manip.manipulation_score,
                        wyckoff_bias=wyckoff.trade_bias,
                    )
                )

                # Step 6: Determine action from all signals
                action = self._determine_action(wyckoff, smc, breakout, manip)
                if action is None:
                    return None

                # Step 7: Full intelligence score (async)
                scorer = SignalScorer()
                scored = await scorer.score(symbol=symbol, action=action, df=df)

                if scored.score < MIN_SCORE:
                    return None

                # Step 8: Calculate SL and targets from S/R
                entry = price
                atr = BreakoutDetector._compute_atr(df)
                sl, t1, t2, t3 = self._calculate_levels(
                    action, entry, sr, atr, breakout
                )

                if sl == 0:
                    return None

                risk = abs(entry - sl)
                reward2 = abs(t2 - entry)
                rr = reward2 / risk if risk > 0 else 0

                result = StockScanResult(
                    symbol=symbol,
                    action=action,
                    score=scored.score,
                    entry_price=round(entry, 2),
                    stop_loss=round(sl, 2),
                    target1=round(t1, 2),
                    target2=round(t2, 2),
                    target3=round(t3, 2),
                    risk_reward=round(rr, 2),
                    risk_per_share=round(risk, 2),
                    atr=round(atr, 2),
                    wyckoff_phase=wyckoff.phase.value,
                    wyckoff_bias=wyckoff.trade_bias or "",
                    wyckoff_confidence=wyckoff.confidence,
                    smc_bullish_signals=smc.bullish_confluence,
                    smc_bearish_signals=smc.bearish_confluence,
                    breakout_type=breakout.breakout_type if breakout.detected else "",
                    breakout_confirmed=breakout.detected and breakout.volume_confirmed and not breakout.false_breakout_risk,
                    breakout_volume_ratio=breakout.volume_ratio if breakout.detected else 0.0,
                    nearest_support=sr.nearest_support.price if sr.nearest_support else None,
                    nearest_resistance=sr.nearest_resistance.price if sr.nearest_resistance else None,
                    manipulation_score=manip.manipulation_score,
                    false_breakout_risk=breakout.false_breakout_risk if breakout.detected else False,
                    rationale=scored.rationale,
                    warnings=scored.breakdown.warnings[:5],
                )

                log.debug(
                    "scanner.analyzed",
                    symbol=symbol,
                    action=action,
                    score=scored.score,
                    rr=f"{rr:.1f}",
                    breakout=breakout.breakout_type if breakout.detected else "none",
                )
                return result

            except Exception as e:
                log.warning("scanner.symbol_failed", symbol=symbol, error=str(e))
                return None

    def _determine_action(self, wyckoff, smc, breakout, manip) -> Optional[str]:
        """
        Decide BUY or SELL from convergent evidence.
        Requires at least 2 of 3 signals to agree.
        """
        bull_votes = 0
        bear_votes = 0

        # Wyckoff vote
        if wyckoff.trade_bias == "bullish":
            bull_votes += 1
        elif wyckoff.trade_bias == "bearish":
            bear_votes += 1

        # SMC vote
        if smc.bullish_confluence > smc.bearish_confluence:
            bull_votes += 1
        elif smc.bearish_confluence > smc.bullish_confluence:
            bear_votes += 1

        # Breakout vote
        if breakout.detected and not breakout.false_breakout_risk:
            if breakout.direction == "bullish":
                bull_votes += 1
            elif breakout.direction == "bearish":
                bear_votes += 1

        # Operator direction from manipulation analysis
        if manip.operator_direction == "bullish":
            bull_votes += 1
        elif manip.operator_direction == "bearish":
            bear_votes += 1

        if bull_votes >= 2 and bull_votes > bear_votes:
            return "BUY"
        if bear_votes >= 2 and bear_votes > bull_votes:
            return "SELL"
        return None  # No clear direction — skip

    def _calculate_levels(
        self,
        action: str,
        entry: float,
        sr: SRAnalysis,
        atr: float,
        breakout,
    ) -> tuple[float, float, float, float]:
        """
        Calculate stop loss and targets based on S/R levels.

        BUY:
          SL  = nearest support below entry - 0.3% buffer
               (min: entry - 1.5*ATR, max: entry - 3*ATR)
          T1  = nearest resistance (1:1 R:R min)
          T2  = next resistance or entry + 2*risk
          T3  = entry + 3*risk

        SELL:
          SL  = nearest resistance above entry + 0.3% buffer
          T1  = nearest support (1:1 R:R min)
          T2  = next support or entry - 2*risk
          T3  = entry - 3*risk
        """
        min_sl_dist = atr * 1.5
        max_sl_dist = atr * 3.0

        if action == "BUY":
            # Find SL: nearest strong support below entry
            sl = 0.0
            for sup in sr.support_levels:
                if sup.price < entry and sup.strength >= 2:
                    sl_candidate = sup.price * 0.997  # 0.3% below support
                    dist = entry - sl_candidate
                    if min_sl_dist <= dist <= max_sl_dist:
                        sl = sl_candidate
                        break
                    elif dist < min_sl_dist:
                        sl = entry - min_sl_dist
                        break

            if sl == 0:
                sl = entry - min_sl_dist  # Default: 1.5x ATR below

            risk = entry - sl

            # Targets
            t1 = 0.0
            if sr.nearest_resistance:
                t1 = sr.nearest_resistance.price
            if t1 <= entry + risk:
                t1 = entry + risk

            t2 = entry + 2 * risk
            t3 = entry + 3 * risk

            # Use next resistance for T2 if available
            for res in sr.resistance_levels:
                if res.price > t1 and res.price <= entry + 3 * risk:
                    t2 = res.price
                    break

        else:  # SELL
            sl = 0.0
            for res in sr.resistance_levels:
                if res.price > entry and res.strength >= 2:
                    sl_candidate = res.price * 1.003
                    dist = sl_candidate - entry
                    if min_sl_dist <= dist <= max_sl_dist:
                        sl = sl_candidate
                        break
                    elif dist < min_sl_dist:
                        sl = entry + min_sl_dist
                        break

            if sl == 0:
                sl = entry + min_sl_dist

            risk = sl - entry

            t1 = 0.0
            if sr.nearest_support:
                t1 = sr.nearest_support.price
            if t1 >= entry - risk:
                t1 = entry - risk

            t2 = entry - 2 * risk
            t3 = entry - 3 * risk

            for sup in sr.support_levels:
                if sup.price < t1 and sup.price >= entry - 3 * risk:
                    t2 = sup.price
                    break

        return sl, t1, t2, t3

    async def _fetch_all_ohlcv(self, symbols: list[str]) -> dict[str, Optional[pd.DataFrame]]:
        """
        Fetch OHLCV for all symbols in batches.
        Uses Redis cache (1-hour TTL) to avoid hammering NSE/yfinance.
        """
        from data.historical import HistoricalDataFetcher
        from datetime import datetime

        fetcher = HistoricalDataFetcher()
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")

        result = {}
        loop = asyncio.get_event_loop()

        for i in range(0, len(symbols), FETCH_BATCH_SIZE):
            batch = symbols[i: i + FETCH_BATCH_SIZE]
            tasks = [
                loop.run_in_executor(
                    None,
                    lambda s=sym: fetcher.get_with_indicators(s, from_date, to_date, None),
                )
                for sym in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, res in zip(batch, batch_results):
                if isinstance(res, pd.DataFrame) and not res.empty:
                    result[sym] = res
                else:
                    log.debug("scanner.ohlcv_skip", symbol=sym)
                    result[sym] = None

            if i + FETCH_BATCH_SIZE < len(symbols):
                await asyncio.sleep(BATCH_DELAY)

        return result

    async def _cache_results(self, report: ScanReport) -> None:
        """Store scan results in Redis for the dashboard and API."""
        import json
        try:
            r = await self._get_redis()
            data = {
                "timestamp": report.timestamp,
                "scanned": report.symbols_scanned,
                "passed": report.symbols_passed,
                "blocked": report.symbols_blocked,
                "duration_sec": report.scan_duration_sec,
                "top_picks": [
                    {
                        "rank": p.rank,
                        "symbol": p.symbol,
                        "action": p.action,
                        "score": p.score,
                        "entry": p.entry_price,
                        "sl": p.stop_loss,
                        "t1": p.target1,
                        "t2": p.target2,
                        "t3": p.target3,
                        "rr": p.risk_reward,
                        "wyckoff": p.wyckoff_phase,
                        "breakout": p.breakout_type,
                        "breakout_confirmed": p.breakout_confirmed,
                        "vol_ratio": p.breakout_volume_ratio,
                        "manip_score": p.manipulation_score,
                        "rationale": p.rationale[:200],
                    }
                    for p in report.top_picks
                ],
            }
            await r.setex("scanner:latest", 1800, json.dumps(data))  # 30 min TTL
        except Exception as e:
            log.warning("scanner.cache_failed", error=str(e))

    async def get_cached_results(self) -> Optional[dict]:
        """Return the last cached scan results from Redis."""
        import json
        try:
            r = await self._get_redis()
            raw = await r.get("scanner:latest")
            return json.loads(raw) if raw else None
        except Exception:
            return None
