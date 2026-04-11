"""
Wyckoff Method — Accumulation / Distribution Phase Detector

Richard Wyckoff's law: large operators (composite man / smart money)
go through predictable phases when building or offloading positions.
Detecting the phase tells you WHETHER to trade, not just direction.

Accumulation (A→E = markup):
  Phase A: Selling exhaustion (PS → SC → AR → ST)
  Phase B: Building cause (secondary tests, range trading)
  Phase C: Spring (shakeout below support — smart money absorbs last sellers)
  Phase D: Demand dominates (SOS, LPS — rising lows, breaking resistance)
  Phase E: Markup (price leaves range)

Distribution (A→E = markdown):
  Phase A: Buying exhaustion (PSY → BC → AR → ST)
  Phase B: Building cause (UTAD tests, range trading)
  Phase C: UTAD / upthrust (shakeout above resistance — operators dump)
  Phase D: Supply dominates (SOW, LPSY — falling highs, breaking support)
  Phase E: Markdown (price leaves range)

Detection logic uses:
  - Volume patterns (SC = peak volume, Spring = low volume, SOS = high volume)
  - Price-volume divergence (distribution: price rising + volume falling)
  - Range structure (support/resistance of the trading range)
  - Candle character (wide range bars vs narrow range bars)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

MIN_CANDLES = 60  # Wyckoff phases need adequate history


class WyckoffPhase(str, Enum):
    # Accumulation phases
    ACC_A_SELLING_CLIMAX    = "ACC-A: Selling Climax"       # Panic selling, high vol
    ACC_B_TESTING           = "ACC-B: Testing Range"         # Sideways, decreasing vol
    ACC_C_SPRING            = "ACC-C: Spring (Shakeout)"     # False break below support
    ACC_D_SOS               = "ACC-D: Sign of Strength"      # Strong up moves, rising lows
    ACC_E_MARKUP            = "ACC-E: Markup"                # Price leaving range up

    # Distribution phases
    DIST_A_BUYING_CLIMAX    = "DIST-A: Buying Climax"        # Climax buying, high vol
    DIST_B_TESTING          = "DIST-B: Testing Range"         # Sideways, decreasing vol
    DIST_C_UTAD             = "DIST-C: UTAD (Upthrust)"      # False break above resistance
    DIST_D_SOW              = "DIST-D: Sign of Weakness"     # Strong down moves, lower highs
    DIST_E_MARKDOWN         = "DIST-E: Markdown"             # Price leaving range down

    # Neutral
    UNKNOWN                 = "Unknown"
    MARKUP                  = "Markup (trending up)"
    MARKDOWN                = "Markdown (trending down)"


@dataclass
class WyckoffResult:
    phase: WyckoffPhase = WyckoffPhase.UNKNOWN

    # Range structure
    trading_range_high: Optional[float] = None    # Resistance of the range
    trading_range_low: Optional[float] = None     # Support of the range
    in_trading_range: bool = False                # Price currently in a range

    # Key events detected
    selling_climax_detected: bool = False         # High-vol reversal bar after downtrend
    buying_climax_detected: bool = False          # High-vol reversal bar after uptrend
    spring_detected: bool = False                 # False break below support + reversal
    upthrust_detected: bool = False               # False break above resistance + reversal

    # Volume character
    vol_on_up_bars: float = 0.0       # Average volume on up bars (demand)
    vol_on_down_bars: float = 0.0     # Average volume on down bars (supply)
    demand_dominates: bool = False    # Vol heavier on up bars = accumulation
    supply_dominates: bool = False    # Vol heavier on down bars = distribution

    # Price-volume divergence
    pv_divergence: Optional[str] = None    # "bullish" | "bearish" | None
    # Bullish divergence: price falling but volume also declining (selling exhaustion)
    # Bearish divergence: price rising but volume declining (distribution, no demand)

    # Strength of cause built (potential effect)
    cause_count: int = 0              # Number of tests / oscillations in range
    # More tests = more cause = bigger move expected (Wyckoff's law of cause and effect)

    # Trading recommendation
    trade_bias: Optional[str] = None  # "bullish" | "bearish" | "wait" | "avoid"
    confidence: float = 0.0           # 0.0–1.0
    reasoning: list[str] = field(default_factory=list)


class WyckoffAnalyzer:
    """
    Identifies Wyckoff accumulation/distribution phases from OHLCV data.

    Returns phase + trading bias, not price targets.
    Do not trade during Phase A or B — let the phase confirm first.
    Only trade C (spring/UTAD) or D (SOS/SOW) phases with confirmation.
    """

    def analyze(self, df: pd.DataFrame) -> WyckoffResult:
        if len(df) < MIN_CANDLES:
            log.warning("wyckoff.insufficient_data", candles=len(df))
            return WyckoffResult(trade_bias="wait", reasoning=["Insufficient data for Wyckoff analysis"])

        df = df.copy().reset_index(drop=True)
        result = WyckoffResult()

        self._compute_volume_character(df, result)
        self._detect_trading_range(df, result)
        self._detect_climax_events(df, result)
        self._detect_pv_divergence(df, result)
        self._classify_phase(df, result)
        self._set_trade_bias(result)

        log.info(
            "wyckoff.result",
            phase=result.phase.value,
            bias=result.trade_bias,
            spring=result.spring_detected,
            utad=result.upthrust_detected,
            demand_dom=result.demand_dominates,
            confidence=f"{result.confidence:.2f}",
        )
        return result

    # ── Volume Character ─────────────────────────────────────────────────

    def _compute_volume_character(self, df: pd.DataFrame, result: WyckoffResult) -> None:
        """
        Separate volume on up-bars (demand) vs down-bars (supply).
        Wyckoff law: when demand > supply persistently = accumulation.
        """
        closes = df["close"].values
        opens = df["open"].values
        volumes = df["volume"].values

        # Last 30 bars for current character
        n = min(30, len(df))
        recent_closes = closes[-n:]
        recent_opens = opens[-n:]
        recent_volumes = volumes[-n:]

        up_bar_mask = recent_closes > recent_opens
        down_bar_mask = recent_closes < recent_opens

        result.vol_on_up_bars = float(np.mean(recent_volumes[up_bar_mask])) if up_bar_mask.any() else 0.0
        result.vol_on_down_bars = float(np.mean(recent_volumes[down_bar_mask])) if down_bar_mask.any() else 0.0

        if result.vol_on_up_bars > 0 and result.vol_on_down_bars > 0:
            ratio = result.vol_on_up_bars / result.vol_on_down_bars
            result.demand_dominates = ratio > 1.3
            result.supply_dominates = ratio < 0.77  # inverse: supply > demand

    # ── Trading Range Detection ──────────────────────────────────────────

    def _detect_trading_range(self, df: pd.DataFrame, result: WyckoffResult) -> None:
        """
        Identify if price is consolidating in a range (key Wyckoff condition).

        Range = last 20+ bars with price oscillating between consistent H/L.
        ATR-based check: low ATR relative to price range = consolidation.
        """
        lookback = min(40, len(df) - 5)
        recent = df.tail(lookback)

        rng_high = float(recent["high"].max())
        rng_low = float(recent["low"].min())
        range_size = rng_high - rng_low

        # ATR over lookback
        tr = pd.concat([
            recent["high"] - recent["low"],
            (recent["high"] - recent["close"].shift()).abs(),
            (recent["low"] - recent["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        avg_atr = float(tr.mean())

        # Range is "tight" if it spans < 15x ATR (sideways price action)
        if range_size < avg_atr * 15 and range_size > avg_atr * 3:
            result.trading_range_high = rng_high
            result.trading_range_low = rng_low
            result.in_trading_range = True

            # Count oscillations (tests of support/resistance = cause building)
            price = recent["close"].values
            tests = 0
            for i in range(1, len(price) - 1):
                if price[i] < rng_low * 1.005:  # Test of support
                    tests += 1
                elif price[i] > rng_high * 0.995:  # Test of resistance
                    tests += 1
            result.cause_count = tests

    # ── Climax Events ────────────────────────────────────────────────────

    def _detect_climax_events(self, df: pd.DataFrame, result: WyckoffResult) -> None:
        """
        Selling Climax (SC): High-volume reversal bar after a prolonged downtrend.
          - Wide spread down bar, high volume, closes off lows (wicks show demand)
          - Immediately followed by Automatic Rally (AR) = sharp bounce

        Buying Climax (BC): High-volume reversal bar after prolonged uptrend.
          - Wide spread up bar, high volume, closes near lows of the bar
          - Immediately followed by Automatic Reaction (AR) = sharp drop

        Spring: Price briefly dips below SC support on LOW volume → reversal up.
          (Low volume on breakdown = no real supply = shakeout, not genuine breakdown)

        UTAD: Price briefly exceeds BC resistance on LOW volume → reversal down.
          (Low volume on breakout = no real demand = upthrust, not genuine breakout)
        """
        closes = df["close"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values
        n = len(df)

        avg_vol = float(np.mean(volumes))
        avg_range = float(np.mean(highs - lows))

        for i in range(5, n - 3):
            bar_range = highs[i] - lows[i]
            bar_vol = volumes[i]
            bar_body = abs(closes[i] - opens[i])

            # High volume bar (2x average)
            is_high_vol = bar_vol > avg_vol * 2.0
            # Wide range bar (1.5x average)
            is_wide_range = bar_range > avg_range * 1.5

            if is_high_vol and is_wide_range:
                # Selling Climax: large down bar, closes above midpoint (buyers stepping in)
                if closes[i] < opens[i] and closes[i] > lows[i] + bar_range * 0.3:
                    # Check downtrend before (5+ consecutive lower closes)
                    prior_closes = closes[max(0, i-5):i]
                    if len(prior_closes) >= 3 and prior_closes[-1] < prior_closes[0]:
                        result.selling_climax_detected = True

                # Buying Climax: large up bar, closes below midpoint (sellers stepping in)
                if closes[i] > opens[i] and closes[i] < highs[i] - bar_range * 0.3:
                    prior_closes = closes[max(0, i-5):i]
                    if len(prior_closes) >= 3 and prior_closes[-1] > prior_closes[0]:
                        result.buying_climax_detected = True

        # Spring detection: false break below support on LOW volume (last 10 bars)
        if result.in_trading_range and result.trading_range_low is not None:
            for i in range(max(0, n - 10), n - 1):
                if (lows[i] < result.trading_range_low and
                        closes[i] > result.trading_range_low and
                        volumes[i] < avg_vol * 0.8):  # Low volume = no supply
                    result.spring_detected = True

        # UTAD detection: false break above resistance on LOW volume (last 10 bars)
        if result.in_trading_range and result.trading_range_high is not None:
            for i in range(max(0, n - 10), n - 1):
                if (highs[i] > result.trading_range_high and
                        closes[i] < result.trading_range_high and
                        volumes[i] < avg_vol * 0.8):
                    result.upthrust_detected = True

    # ── Price-Volume Divergence ───────────────────────────────────────────

    def _detect_pv_divergence(self, df: pd.DataFrame, result: WyckoffResult) -> None:
        """
        Bearish divergence: price trending up but volume trending down.
          → Distribution — operators selling into retail buying.

        Bullish divergence: price trending down but volume trending down.
          → Accumulation — selling exhaustion, operators absorbing supply.
        """
        lookback = min(20, len(df))
        prices = df["close"].values[-lookback:]
        volumes = df["volume"].values[-lookback:]

        # Linear regression slope for price and volume
        x = np.arange(lookback)
        price_slope = np.polyfit(x, prices, 1)[0]
        vol_slope = np.polyfit(x, volumes, 1)[0]

        # Normalize to percentage change
        price_trend = price_slope / prices[0] * 100 if prices[0] > 0 else 0
        vol_trend = vol_slope / volumes[0] * 100 if volumes[0] > 0 else 0

        # Bearish divergence: price rising + volume falling
        if price_trend > 0.05 and vol_trend < -0.05:
            result.pv_divergence = "bearish"

        # Bullish divergence: price falling + volume also falling (exhaustion)
        elif price_trend < -0.05 and vol_trend < -0.05:
            result.pv_divergence = "bullish"

    # ── Phase Classification ──────────────────────────────────────────────

    def _classify_phase(self, df: pd.DataFrame, result: WyckoffResult) -> None:
        """Map detected signals to Wyckoff phase."""
        closes = df["close"].values
        n = len(df)

        # Is price trending or in range?
        price_20d_change = (closes[-1] - closes[-min(20, n)]) / closes[-min(20, n)] * 100
        price_60d_change = (closes[-1] - closes[-min(60, n)]) / closes[-min(60, n)] * 100

        if not result.in_trading_range:
            if price_20d_change > 3:
                result.phase = WyckoffPhase.MARKUP
            elif price_20d_change < -3:
                result.phase = WyckoffPhase.MARKDOWN
            else:
                result.phase = WyckoffPhase.UNKNOWN
            return

        # In a trading range — determine accumulation or distribution
        # Key question: did we get here from below (acc) or above (dist)?
        downtrend_before = price_60d_change < -5    # Came from downtrend = likely acc
        uptrend_before = price_60d_change > 5       # Came from uptrend = likely dist

        current_price = closes[-1]
        range_pct = 0.0
        if result.trading_range_high and result.trading_range_low:
            rng = result.trading_range_high - result.trading_range_low
            if rng > 0:
                range_pct = (current_price - result.trading_range_low) / rng

        if downtrend_before or result.selling_climax_detected:
            # Accumulation structure
            if result.spring_detected:
                result.phase = WyckoffPhase.ACC_C_SPRING
            elif result.demand_dominates and range_pct > 0.5:
                result.phase = WyckoffPhase.ACC_D_SOS
            elif result.cause_count >= 4:
                result.phase = WyckoffPhase.ACC_B_TESTING
            else:
                result.phase = WyckoffPhase.ACC_A_SELLING_CLIMAX

        elif uptrend_before or result.buying_climax_detected:
            # Distribution structure
            if result.upthrust_detected:
                result.phase = WyckoffPhase.DIST_C_UTAD
            elif result.supply_dominates and range_pct < 0.5:
                result.phase = WyckoffPhase.DIST_D_SOW
            elif result.cause_count >= 4:
                result.phase = WyckoffPhase.DIST_B_TESTING
            else:
                result.phase = WyckoffPhase.DIST_A_BUYING_CLIMAX

    # ── Trade Bias ────────────────────────────────────────────────────────

    def _set_trade_bias(self, result: WyckoffResult) -> None:
        """
        Translate phase to actionable trade bias.

        Rule: Only trade Phase C and Phase D — the highest probability setups.
        Phase A and B = wait. Do NOT guess the direction mid-range.
        """
        phase = result.phase
        reasons = []
        confidence = 0.0

        if phase == WyckoffPhase.ACC_C_SPRING:
            result.trade_bias = "bullish"
            confidence = 0.85
            reasons.append("SPRING confirmed — operators absorbed last sellers, markup expected")
            reasons.append("Enter on re-test of spring low with rising volume")

        elif phase == WyckoffPhase.ACC_D_SOS:
            result.trade_bias = "bullish"
            confidence = 0.80
            reasons.append("Sign of Strength — demand dominates, price trending to top of range")
            reasons.append("Buy Last Point of Support (LPS) dips with decreasing volume")

        elif phase == WyckoffPhase.ACC_E_MARKUP:
            result.trade_bias = "bullish"
            confidence = 0.70
            reasons.append("Markup phase — price left accumulation range, trend is up")
            reasons.append("Buy pullbacks to old resistance-turned-support")

        elif phase == WyckoffPhase.DIST_C_UTAD:
            result.trade_bias = "bearish"
            confidence = 0.85
            reasons.append("UTAD confirmed — operators distributed into retail FOMO, markdown expected")
            reasons.append("Enter on return into range after upthrust rejection")

        elif phase == WyckoffPhase.DIST_D_SOW:
            result.trade_bias = "bearish"
            confidence = 0.80
            reasons.append("Sign of Weakness — supply dominates, price trending to bottom of range")
            reasons.append("Sell Last Point of Supply (LPSY) bounces with decreasing volume")

        elif phase == WyckoffPhase.DIST_E_MARKDOWN:
            result.trade_bias = "bearish"
            confidence = 0.70
            reasons.append("Markdown phase — price left distribution range, trend is down")
            reasons.append("Sell rallies to old support-turned-resistance")

        elif phase in (WyckoffPhase.ACC_B_TESTING, WyckoffPhase.DIST_B_TESTING):
            result.trade_bias = "wait"
            confidence = 0.0
            reasons.append("Phase B testing — cause still being built. Wait for Phase C trigger.")

        elif phase in (WyckoffPhase.ACC_A_SELLING_CLIMAX, WyckoffPhase.DIST_A_BUYING_CLIMAX):
            result.trade_bias = "wait"
            confidence = 0.0
            reasons.append("Phase A — initial stopping action, too early. Wait for range to develop.")

        elif phase == WyckoffPhase.MARKUP:
            result.trade_bias = "bullish"
            confidence = 0.50
            reasons.append("Uptrend (markup) — trade with trend, buy pullbacks only")

        elif phase == WyckoffPhase.MARKDOWN:
            result.trade_bias = "bearish"
            confidence = 0.50
            reasons.append("Downtrend (markdown) — trade with trend, sell rallies only")

        else:
            result.trade_bias = "avoid"
            confidence = 0.0
            reasons.append("Phase unidentifiable — no clear Wyckoff structure. Do not trade.")

        # Add price-volume divergence warnings
        if result.pv_divergence == "bearish":
            reasons.append("WARNING: Price rising on falling volume — distribution in progress")
            if result.trade_bias == "bullish":
                confidence *= 0.5

        if result.pv_divergence == "bullish":
            reasons.append("Selling exhaustion confirmed by falling volume on down moves")
            if result.trade_bias == "bullish":
                confidence = min(1.0, confidence * 1.2)

        result.confidence = round(confidence, 2)
        result.reasoning = reasons
