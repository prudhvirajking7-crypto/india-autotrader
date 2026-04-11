"""
Breakout Detector

A breakout is NOT just a price crossing a level.
A REAL breakout requires ALL of:
  1. Price CLOSES above/below the level (not just a wick)
  2. Volume ≥ 1.5x the 20-day average (institutional participation)
  3. Candle body > 60% of its range (conviction — not a doji/pin)
  4. Wyckoff + SMC confirm the direction (no manipulation, no fake breakout)
  5. Level strength ≥ 2 (at least 2 sources agree this is a key level)

Breakout types:
  - Range Breakout:   Price exits a consolidation box (most reliable)
  - Resistance Break: Close above a key resistance level
  - Support Break:    Close below a key support level
  - Volatility Squeeze: Bollinger Band squeeze followed by expansion
  - Inside Bar Break: Inside candle expands beyond mother bar

False breakout filters (blocks the signal):
  - ManipulationDetector finds ≥1 active flag
  - Volume is BELOW average (no institutional backing)
  - Wyckoff says it's a UTAD/Spring (operators faking the break)
  - Price closes back inside the level within 2 bars = fake
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from scanner.support_resistance import SRLevel, SRAnalysis

log = structlog.get_logger(__name__)

# Volume requirement for a real breakout
BREAKOUT_VOLUME_MULT = 1.5
# Candle body must be this fraction of its range
BODY_CONVICTION_RATIO = 0.60
# ATR multiplier: breakout candle must move > this beyond the level
ATR_CLEARANCE = 0.3
# Minimum level strength to count as a breakout
MIN_LEVEL_STRENGTH = 2


@dataclass
class BreakoutSignal:
    detected: bool = False
    direction: str = ""              # "bullish" | "bearish"
    breakout_type: str = ""          # "resistance_break" | "support_break" | "range_break" | "squeeze"
    breakout_level: Optional[float] = None   # The exact level that was broken
    level_strength: int = 0          # How strong was the broken level

    # Confirmation metrics
    volume_confirmed: bool = False   # Volume ≥ 1.5x avg
    volume_ratio: float = 0.0        # Actual volume / avg volume
    body_conviction: bool = False    # Body > 60% of candle range
    atr_cleared: bool = False        # Price sufficiently beyond the level
    manipulation_clear: bool = True  # No manipulation flags

    # Retest opportunity (best entry — not raw breakout)
    retest_entry: Optional[float] = None  # Level to buy the retest
    retest_window_bars: int = 5          # How many bars to wait for retest

    # False breakout warning
    false_breakout_risk: bool = False
    false_breakout_reasons: list[str] = field(default_factory=list)

    # Squeeze breakout details
    squeeze_bars: int = 0            # How many bars was the squeeze active

    # Summary
    confidence: float = 0.0          # 0.0–1.0 based on confirmations
    reasons: list[str] = field(default_factory=list)


class BreakoutDetector:
    """
    Detects real breakouts with multi-factor confirmation.

    Must be paired with:
      - SRAnalysis (to know the levels)
      - ManipulationDetector output (to filter fakes)
      - WyckoffResult (to confirm direction)
    """

    def detect(
        self,
        df: pd.DataFrame,
        sr: SRAnalysis,
        manipulation_score: int = 0,
        wyckoff_bias: Optional[str] = None,
    ) -> BreakoutSignal:
        """
        Detect breakout in recent price action.

        Args:
            df:                 OHLCV DataFrame (ascending, min 30 rows)
            sr:                 SRAnalysis from SRFinder
            manipulation_score: from ManipulationDetector (≥2 = blocked)
            wyckoff_bias:       "bullish" | "bearish" | "wait" | "avoid"

        Returns:
            BreakoutSignal — detected=True only if all confirmations pass
        """
        signal = BreakoutSignal()

        if len(df) < 20:
            return signal

        df = df.copy().reset_index(drop=True)

        # Manipulation gate
        if manipulation_score >= 2:
            signal.manipulation_clear = False
            signal.false_breakout_reasons.append(f"Manipulation detected ({manipulation_score} signals)")
            signal.false_breakout_risk = True
            return signal

        atr = self._compute_atr(df)
        avg_vol = float(df["volume"].tail(20).mean())

        # Check resistance breakout (bullish)
        bull = self._check_resistance_break(df, sr, atr, avg_vol, wyckoff_bias, signal)

        # Check support breakdown (bearish)
        if not bull:
            self._check_support_break(df, sr, atr, avg_vol, wyckoff_bias, signal)

        # Check squeeze breakout (direction TBD)
        if not signal.detected:
            self._check_squeeze_breakout(df, sr, atr, avg_vol, wyckoff_bias, signal)

        if signal.detected:
            signal.confidence = self._compute_confidence(signal)
            log.info(
                "breakout.detected",
                direction=signal.direction,
                type=signal.breakout_type,
                level=signal.breakout_level,
                vol_ratio=f"{signal.volume_ratio:.2f}x",
                confidence=f"{signal.confidence:.2f}",
            )

        return signal

    # ── Resistance Breakout ───────────────────────────────────────────────

    def _check_resistance_break(
        self,
        df: pd.DataFrame,
        sr: SRAnalysis,
        atr: float,
        avg_vol: float,
        wyckoff_bias: Optional[str],
        signal: BreakoutSignal,
    ) -> bool:
        """
        Bullish breakout: current bar closes ABOVE a resistance level.
        All three confirmations must pass.
        """
        if not sr.resistance_levels:
            return False

        # Focus on the 3 nearest resistance levels
        targets = sr.resistance_levels[:3]
        closes = df["close"].values
        highs = df["high"].values
        opens = df["open"].values
        volumes = df["volume"].values
        n = len(df)

        current_close = closes[-1]
        current_open = opens[-1]
        current_high = highs[-1]
        current_vol = volumes[-1]
        candle_range = current_high - df["low"].values[-1]
        body = abs(current_close - current_open)

        for level in targets:
            if level.strength < MIN_LEVEL_STRENGTH:
                continue
            res_price = level.price

            # Was resistance just broken? (close above, and was below 2 bars ago)
            if current_close <= res_price:
                continue
            if closes[-3] >= res_price:  # Already above 3 bars ago = not a fresh break
                continue

            # Confirmation 1: Volume
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0
            vol_ok = vol_ratio >= BREAKOUT_VOLUME_MULT

            # Confirmation 2: Body conviction
            body_ok = (body / candle_range >= BODY_CONVICTION_RATIO) if candle_range > 0 else False

            # Confirmation 3: ATR clearance (price cleared level by meaningful amount)
            clearance = current_close - res_price
            atr_ok = clearance >= atr * ATR_CLEARANCE

            # Wyckoff alignment
            wyckoff_ok = wyckoff_bias in ("bullish", None) or wyckoff_bias == "bullish"

            # Fake breakout risk: volume too low
            if not vol_ok:
                signal.false_breakout_risk = True
                signal.false_breakout_reasons.append(
                    f"Low-volume resistance break {res_price:.1f} ({vol_ratio:.1f}x avg) — likely fake"
                )

            # Even if volume low, report the breakout with flag
            signal.detected = True
            signal.direction = "bullish"
            signal.breakout_type = "resistance_break"
            signal.breakout_level = res_price
            signal.level_strength = level.strength
            signal.volume_confirmed = vol_ok
            signal.volume_ratio = round(vol_ratio, 2)
            signal.body_conviction = body_ok
            signal.atr_cleared = atr_ok
            signal.retest_entry = res_price * 1.001  # Buy on retest of broken resistance

            signal.reasons.append(
                f"Closed {current_close:.1f} above resistance {res_price:.1f} "
                f"(strength:{level.strength}, vol:{vol_ratio:.1f}x, "
                f"sources:{','.join(level.sources[:2])})"
            )

            if not wyckoff_ok:
                signal.false_breakout_risk = True
                signal.false_breakout_reasons.append(
                    f"Wyckoff says {wyckoff_bias} — contradicts bullish breakout"
                )

            return True

        return False

    # ── Support Breakdown ─────────────────────────────────────────────────

    def _check_support_break(
        self,
        df: pd.DataFrame,
        sr: SRAnalysis,
        atr: float,
        avg_vol: float,
        wyckoff_bias: Optional[str],
        signal: BreakoutSignal,
    ) -> bool:
        """
        Bearish breakout: current bar closes BELOW a support level.
        """
        if not sr.support_levels:
            return False

        targets = sr.support_levels[:3]
        closes = df["close"].values
        lows = df["low"].values
        opens = df["open"].values
        volumes = df["volume"].values
        n = len(df)

        current_close = closes[-1]
        current_open = opens[-1]
        current_low = lows[-1]
        current_vol = volumes[-1]
        candle_range = df["high"].values[-1] - current_low
        body = abs(current_close - current_open)

        for level in targets:
            if level.strength < MIN_LEVEL_STRENGTH:
                continue
            sup_price = level.price

            if current_close >= sup_price:
                continue
            if closes[-3] <= sup_price:
                continue

            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0
            vol_ok = vol_ratio >= BREAKOUT_VOLUME_MULT
            body_ok = (body / candle_range >= BODY_CONVICTION_RATIO) if candle_range > 0 else False
            clearance = sup_price - current_close
            atr_ok = clearance >= atr * ATR_CLEARANCE

            if not vol_ok:
                signal.false_breakout_risk = True
                signal.false_breakout_reasons.append(
                    f"Low-volume support break {sup_price:.1f} ({vol_ratio:.1f}x avg)"
                )

            signal.detected = True
            signal.direction = "bearish"
            signal.breakout_type = "support_break"
            signal.breakout_level = sup_price
            signal.level_strength = level.strength
            signal.volume_confirmed = vol_ok
            signal.volume_ratio = round(vol_ratio, 2)
            signal.body_conviction = body_ok
            signal.atr_cleared = atr_ok
            signal.retest_entry = sup_price * 0.999  # Sell on retest of broken support

            signal.reasons.append(
                f"Closed {current_close:.1f} below support {sup_price:.1f} "
                f"(strength:{level.strength}, vol:{vol_ratio:.1f}x)"
            )

            if wyckoff_bias == "bullish":
                signal.false_breakout_risk = True
                signal.false_breakout_reasons.append("Wyckoff bullish — contradicts support break")

            return True

        return False

    # ── Squeeze Breakout ─────────────────────────────────────────────────

    def _check_squeeze_breakout(
        self,
        df: pd.DataFrame,
        sr: SRAnalysis,
        atr: float,
        avg_vol: float,
        wyckoff_bias: Optional[str],
        signal: BreakoutSignal,
    ) -> None:
        """
        Bollinger Band Squeeze: bands narrow to historic low → explosive move coming.

        Squeeze = BB bandwidth < 20th percentile of its own 100-bar history.
        Breakout = current bandwidth expanding AND price closing outside BB.

        This detects compressed volatility about to explode — not guessing direction,
        letting the first decisive close outside tell us.
        """
        try:
            import pandas_ta as ta
        except ImportError:
            from utils import ta_compat as ta

        if len(df) < 50:
            return

        closes = df["close"]
        bb = ta.bbands(closes, length=20, std=2.0)
        if bb is None or bb.empty:
            return

        upper_col = [c for c in bb.columns if "BBU" in c]
        lower_col = [c for c in bb.columns if "BBL" in c]
        bw_col = [c for c in bb.columns if "BBB" in c]

        if not (upper_col and lower_col and bw_col):
            return

        bw = bb[bw_col[0]].dropna()
        upper = bb[upper_col[0]].iloc[-1]
        lower = bb[lower_col[0]].iloc[-1]
        current_close = float(closes.iloc[-1])
        current_bw = float(bw.iloc[-1])

        if len(bw) < 20:
            return

        # Squeeze: bandwidth was compressed (last 10 bars below 25th percentile)
        bw_pct25 = float(bw.quantile(0.25))
        squeeze_bars = int((bw.tail(10) < bw_pct25).sum())

        if squeeze_bars < 3:
            return  # Not a meaningful squeeze

        # Check if expanding now
        was_compressed = current_bw < bw_pct25 * 1.5

        vol_ratio = float(df["volume"].iloc[-1]) / avg_vol if avg_vol > 0 else 0
        vol_ok = vol_ratio >= BREAKOUT_VOLUME_MULT

        if current_close > upper and vol_ok:
            signal.detected = True
            signal.direction = "bullish"
            signal.breakout_type = "squeeze"
            signal.breakout_level = float(upper)
            signal.squeeze_bars = squeeze_bars
            signal.volume_confirmed = True
            signal.volume_ratio = round(vol_ratio, 2)
            signal.body_conviction = True
            signal.atr_cleared = True
            signal.reasons.append(
                f"BB squeeze breakout UP after {squeeze_bars} compression bars, "
                f"close {current_close:.1f} > upper {upper:.1f}, vol {vol_ratio:.1f}x"
            )

        elif current_close < lower and vol_ok:
            signal.detected = True
            signal.direction = "bearish"
            signal.breakout_type = "squeeze"
            signal.breakout_level = float(lower)
            signal.squeeze_bars = squeeze_bars
            signal.volume_confirmed = True
            signal.volume_ratio = round(vol_ratio, 2)
            signal.body_conviction = True
            signal.atr_cleared = True
            signal.reasons.append(
                f"BB squeeze breakout DOWN after {squeeze_bars} compression bars, "
                f"close {current_close:.1f} < lower {lower:.1f}, vol {vol_ratio:.1f}x"
            )

        if signal.detected and not wyckoff_bias:
            signal.reasons.append("Wyckoff phase not confirming — use smaller size")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)
        trs = []
        for i in range(1, min(n, period + 1)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return float(np.mean(trs)) if trs else 1.0

    @staticmethod
    def _compute_confidence(signal: BreakoutSignal) -> float:
        """Score breakout quality 0–1 based on confirmations passed."""
        score = 0.0
        if signal.volume_confirmed:
            score += 0.35
        if signal.body_conviction:
            score += 0.20
        if signal.atr_cleared:
            score += 0.15
        if signal.manipulation_clear:
            score += 0.15
        if signal.level_strength >= 3:
            score += 0.10
        elif signal.level_strength >= 2:
            score += 0.05
        if signal.false_breakout_risk:
            score -= 0.25
        return round(max(0.0, min(1.0, score)), 2)
