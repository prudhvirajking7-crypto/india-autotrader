"""
Smart Money Concepts (SMC) Analyzer

Detects WHERE institutional money has already acted — not where it might act.

Concepts implemented:
  1. Order Blocks (OB)       — last opposing candle before an impulse move
  2. Fair Value Gaps (FVG)   — 3-candle imbalances (price magnets for fill)
  3. Liquidity Sweeps        — stop hunts above/below swing levels
  4. Break of Structure (BOS)— confirmed trend direction
  5. Change of Character     — first counter-trend signal (CHoCH)

SMC is evidence-based: each signal requires confirmed price action,
not guesses or lagging indicators.

Usage:
    from intelligence.smart_money import SMCAnalyzer
    result = SMCAnalyzer().analyze(df)  # df must have OHLCV columns
    # result.bullish_ob_nearby → True if price is testing a bullish order block
    # result.liquidity_swept   → "sell_side" | "buy_side" | None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# Minimum candles required for meaningful SMC analysis
MIN_CANDLES = 50
# Impulse threshold: price move that qualifies a candle as "origin of impulse"
IMPULSE_PCT = 0.5       # 0.5% move to qualify as impulse
# Swing look-back window for liquidity detection
SWING_LOOKBACK = 20
# How close (%) to an order block counts as "test"
OB_PROXIMITY_PCT = 0.3


@dataclass
class OrderBlock:
    """A price zone where smart money placed significant orders."""
    direction: str          # "bullish" | "bearish"
    high: float
    low: float
    origin_index: int       # bar index where OB was formed
    strength: float         # 0–1 (impulse size after OB)
    mitigated: bool = False # True once price returned and passed through


@dataclass
class FairValueGap:
    """3-candle imbalance — untraded zone the market will likely revisit."""
    direction: str          # "bullish" (gap up) | "bearish" (gap down)
    upper: float
    lower: float
    origin_index: int
    filled: bool = False


@dataclass
class SMCResult:
    # Structure
    bos_direction: Optional[str] = None        # "bullish" | "bearish" | None (no BOS yet)
    choch_detected: bool = False               # First counter-trend swing = warning
    choch_direction: Optional[str] = None      # Direction of CHoCH

    # Order blocks
    nearest_bullish_ob: Optional[OrderBlock] = None
    nearest_bearish_ob: Optional[OrderBlock] = None
    price_at_bullish_ob: bool = False          # Current price testing bullish OB
    price_at_bearish_ob: bool = False          # Current price testing bearish OB

    # Fair Value Gaps
    open_bullish_fvg: Optional[FairValueGap] = None
    open_bearish_fvg: Optional[FairValueGap] = None
    price_in_bullish_fvg: bool = False         # Price inside unfilled bullish FVG
    price_in_bearish_fvg: bool = False

    # Liquidity
    buy_side_liquidity: Optional[float] = None    # Swing high = BSL (target for bears)
    sell_side_liquidity: Optional[float] = None   # Swing low  = SSL (target for bulls)
    liquidity_swept: Optional[str] = None         # "buy_side" | "sell_side" | None
    sweep_confirmed: bool = False                 # Sweep + reversal candle confirmed

    # Premium / Discount zones
    equilibrium: Optional[float] = None        # 50% of recent range (fair value)
    in_discount_zone: bool = False             # Price below equilibrium (bullish zone)
    in_premium_zone: bool = False              # Price above equilibrium (bearish zone)

    # Summary signals
    bullish_confluence: int = 0    # Count of bullish SMC signals confirmed
    bearish_confluence: int = 0    # Count of bearish SMC signals confirmed
    evidence_summary: list[str] = field(default_factory=list)


class SMCAnalyzer:
    """
    Analyzes OHLCV data for Smart Money Concept patterns.

    All signals are derived from confirmed price action.
    No indicators. No guesses.
    """

    def analyze(self, df: pd.DataFrame) -> SMCResult:
        """
        Full SMC analysis on OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index must be sorted ascending (oldest first).

        Returns:
            SMCResult with all detected structures.
        """
        result = SMCResult()

        if len(df) < MIN_CANDLES:
            log.warning("smc.insufficient_data", candles=len(df))
            return result

        df = df.copy().reset_index(drop=True)
        price = float(df["close"].iloc[-1])

        self._detect_structure(df, result)
        self._detect_order_blocks(df, result, price)
        self._detect_fvg(df, result, price)
        self._detect_liquidity(df, result, price)
        self._detect_premium_discount(df, result, price)
        self._compute_confluence(result)

        log.info(
            "smc.analyzed",
            bos=result.bos_direction,
            swept=result.liquidity_swept,
            at_bull_ob=result.price_at_bullish_ob,
            at_bear_ob=result.price_at_bearish_ob,
            bull_signals=result.bullish_confluence,
            bear_signals=result.bearish_confluence,
        )
        return result

    # ── 1. Break of Structure & Change of Character ───────────────────────

    def _detect_structure(self, df: pd.DataFrame, result: SMCResult) -> None:
        """
        BOS: Confirmed break of the most recent swing high/low.
        CHoCH: First time price breaks structure in the opposing direction.

        Uses zigzag-style swing detection (local maxima/minima).
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        # Find swing highs and lows (local extremes over 5-bar window)
        swing_highs = []  # (index, value)
        swing_lows = []

        for i in range(5, n - 5):
            if highs[i] == max(highs[i-5:i+6]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i-5:i+6]):
                swing_lows.append((i, lows[i]))

        if not swing_highs or not swing_lows:
            return

        current_close = closes[-1]
        last_swing_high = swing_highs[-1][1]
        last_swing_low = swing_lows[-1][1]
        prev_swing_high = swing_highs[-2][1] if len(swing_highs) >= 2 else last_swing_high
        prev_swing_low = swing_lows[-2][1] if len(swing_lows) >= 2 else last_swing_low

        # BOS bullish: current close above last significant swing high
        if current_close > last_swing_high:
            result.bos_direction = "bullish"
            # CHoCH: was previously bearish (lower highs sequence)?
            if last_swing_high < prev_swing_high:
                result.choch_detected = True
                result.choch_direction = "bullish"

        # BOS bearish: current close below last significant swing low
        elif current_close < last_swing_low:
            result.bos_direction = "bearish"
            if last_swing_low > prev_swing_low:
                result.choch_detected = True
                result.choch_direction = "bearish"

    # ── 2. Order Blocks ───────────────────────────────────────────────────

    def _detect_order_blocks(self, df: pd.DataFrame, result: SMCResult, price: float) -> None:
        """
        Bullish OB: Last bearish candle before a strong bullish impulse.
                    Smart money placed buy orders here.
        Bearish OB: Last bullish candle before a strong bearish impulse.
                    Smart money placed sell orders here.

        Only unmitigated OBs are tracked (price hasn't returned yet).
        """
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        bullish_obs = []
        bearish_obs = []

        for i in range(1, n - 5):
            # Bullish OB: bearish candle followed by strong up move
            is_bearish_candle = closes[i] < opens[i]
            if is_bearish_candle:
                # Check for impulse after this candle (next 3 candles)
                future_high = max(highs[i+1:min(i+6, n)])
                impulse = (future_high - closes[i]) / closes[i] * 100
                if impulse >= IMPULSE_PCT:
                    strength = min(1.0, impulse / 2.0)
                    ob = OrderBlock(
                        direction="bullish",
                        high=highs[i],
                        low=lows[i],
                        origin_index=i,
                        strength=strength,
                    )
                    # Check if mitigated (price returned to this zone after the impulse)
                    if i + 3 < n:
                        ob.mitigated = float(min(lows[i+3:])) <= ob.low
                    bullish_obs.append(ob)

            # Bearish OB: bullish candle followed by strong down move
            is_bullish_candle = closes[i] > opens[i]
            if is_bullish_candle:
                future_low = min(lows[i+1:min(i+6, n)])
                impulse = (closes[i] - future_low) / closes[i] * 100
                if impulse >= IMPULSE_PCT:
                    strength = min(1.0, impulse / 2.0)
                    ob = OrderBlock(
                        direction="bearish",
                        high=highs[i],
                        low=lows[i],
                        origin_index=i,
                        strength=strength,
                    )
                    if i + 3 < n:
                        ob.mitigated = float(max(highs[i+3:])) >= ob.high
                    bearish_obs.append(ob)

        # Get nearest unmitigated OBs
        active_bull_obs = [ob for ob in bullish_obs if not ob.mitigated and ob.high < price]
        active_bear_obs = [ob for ob in bearish_obs if not ob.mitigated and ob.low > price]

        if active_bull_obs:
            result.nearest_bullish_ob = max(active_bull_obs, key=lambda x: x.high)
            ob = result.nearest_bullish_ob
            proximity = (price - ob.high) / price * 100
            result.price_at_bullish_ob = proximity <= OB_PROXIMITY_PCT

        if active_bear_obs:
            result.nearest_bearish_ob = min(active_bear_obs, key=lambda x: x.low)
            ob = result.nearest_bearish_ob
            proximity = (ob.low - price) / price * 100
            result.price_at_bearish_ob = proximity <= OB_PROXIMITY_PCT

    # ── 3. Fair Value Gaps ────────────────────────────────────────────────

    def _detect_fvg(self, df: pd.DataFrame, result: SMCResult, price: float) -> None:
        """
        Bullish FVG: candle[i-1].high < candle[i+1].low — gap left behind on up move.
        Bearish FVG: candle[i-1].low > candle[i+1].high — gap on down move.

        FVGs act as magnets — price usually returns to fill them.
        Trading WITH the FVG direction (into discount/premium) = high probability.
        """
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        bull_fvgs = []
        bear_fvgs = []

        for i in range(1, n - 1):
            # Bullish FVG: gap between i-1 high and i+1 low
            if highs[i-1] < lows[i+1]:
                fvg = FairValueGap(
                    direction="bullish",
                    upper=lows[i+1],
                    lower=highs[i-1],
                    origin_index=i,
                )
                # Check if already filled
                if i + 2 < n:
                    fvg.filled = float(min(lows[i+2:])) <= fvg.lower
                bull_fvgs.append(fvg)

            # Bearish FVG: gap between i-1 low and i+1 high
            if lows[i-1] > highs[i+1]:
                fvg = FairValueGap(
                    direction="bearish",
                    upper=lows[i-1],
                    lower=highs[i+1],
                    origin_index=i,
                )
                if i + 2 < n:
                    fvg.filled = float(max(highs[i+2:])) >= fvg.upper
                bear_fvgs.append(fvg)

        # Most recent unfilled FVGs
        open_bull = [f for f in bull_fvgs if not f.filled and f.upper < price]
        open_bear = [f for f in bear_fvgs if not f.filled and f.lower > price]

        if open_bull:
            result.open_bullish_fvg = max(open_bull, key=lambda x: x.origin_index)
            fvg = result.open_bullish_fvg
            result.price_in_bullish_fvg = fvg.lower <= price <= fvg.upper

        if open_bear:
            result.open_bearish_fvg = min(open_bear, key=lambda x: x.lower)
            fvg = result.open_bearish_fvg
            result.price_in_bearish_fvg = fvg.lower <= price <= fvg.upper

    # ── 4. Liquidity Sweeps ───────────────────────────────────────────────

    def _detect_liquidity(self, df: pd.DataFrame, result: SMCResult, price: float) -> None:
        """
        Retail traders place stop losses just above swing highs (buy-side liquidity)
        and just below swing lows (sell-side liquidity).

        Smart money hunts these stops before reversing:
          - Sweep above swing high → price reverses down → bearish (BSL swept)
          - Sweep below swing low → price reverses up → bullish (SSL swept)

        A sweep is confirmed when:
          1. Price wicks above/below the swing level
          2. Candle closes back on the opposite side (rejection)
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        n = len(df)

        lookback = min(SWING_LOOKBACK, n - 5)

        # Recent swing high and low (excluding last 2 bars)
        recent_highs = highs[-(lookback + 2):-2]
        recent_lows = lows[-(lookback + 2):-2]
        swing_high = float(np.max(recent_highs))
        swing_low = float(np.min(recent_lows))

        result.buy_side_liquidity = swing_high
        result.sell_side_liquidity = swing_low

        # Check last 3 candles for a sweep
        for i in range(max(0, n - 3), n):
            h = highs[i]
            l = lows[i]
            c = closes[i]
            o = opens[i]

            # Buy-side liquidity sweep: wick above swing high, closes back below
            if h > swing_high and c < swing_high:
                result.liquidity_swept = "buy_side"
                # Confirmed if rejection candle (bearish close)
                result.sweep_confirmed = c < o

            # Sell-side liquidity sweep: wick below swing low, closes back above
            elif l < swing_low and c > swing_low:
                result.liquidity_swept = "sell_side"
                # Confirmed if bullish close
                result.sweep_confirmed = c > o

    # ── 5. Premium / Discount Zones ──────────────────────────────────────

    def _detect_premium_discount(self, df: pd.DataFrame, result: SMCResult, price: float) -> None:
        """
        Equilibrium (EQ) = 50% of the most recent swing range.
        Discount zone = below EQ (cheap — look for longs)
        Premium zone  = above EQ (expensive — look for shorts)

        Institutions BUY in discount and SELL in premium.
        """
        lookback = min(50, len(df))
        recent_high = float(df["high"].tail(lookback).max())
        recent_low = float(df["low"].tail(lookback).min())
        result.equilibrium = (recent_high + recent_low) / 2

        if price < result.equilibrium:
            result.in_discount_zone = True
        elif price > result.equilibrium:
            result.in_premium_zone = True

    # ── 6. Confluence Score ───────────────────────────────────────────────

    def _compute_confluence(self, result: SMCResult) -> None:
        """Count independent bullish/bearish evidence signals."""
        bull = []
        bear = []

        if result.bos_direction == "bullish":
            bull.append("BOS bullish (structure confirmed up)")
        elif result.bos_direction == "bearish":
            bear.append("BOS bearish (structure confirmed down)")

        if result.price_at_bullish_ob:
            bull.append(f"Price at bullish order block (strength {result.nearest_bullish_ob.strength:.2f})")
        if result.price_at_bearish_ob:
            bear.append(f"Price at bearish order block (strength {result.nearest_bearish_ob.strength:.2f})")

        if result.price_in_bullish_fvg:
            bull.append("Price inside unfilled bullish FVG (magnet above)")
        if result.price_in_bearish_fvg:
            bear.append("Price inside unfilled bearish FVG (magnet below)")

        if result.liquidity_swept == "sell_side" and result.sweep_confirmed:
            bull.append("Confirmed sell-side liquidity sweep (stop hunt below, reversal up)")
        if result.liquidity_swept == "buy_side" and result.sweep_confirmed:
            bear.append("Confirmed buy-side liquidity sweep (stop hunt above, reversal down)")

        if result.in_discount_zone:
            bull.append("Price in discount zone (below 50% range)")
        if result.in_premium_zone:
            bear.append("Price in premium zone (above 50% range)")

        if result.choch_detected:
            label = f"CHoCH {result.choch_direction} (first structure reversal — watch carefully)"
            if result.choch_direction == "bullish":
                bull.append(label)
            else:
                bear.append(label)

        result.bullish_confluence = len(bull)
        result.bearish_confluence = len(bear)
        result.evidence_summary = bull + ["— vs —"] + bear if (bull or bear) else []
