"""
Market Manipulation Detector

Indian markets — especially F&O — are heavily manipulated by operators.
This module detects manipulation patterns and blocks trades when found.

Patterns detected:
  1. Stop Hunt       — sharp spike beyond swing level, immediate reversal
  2. Fake Breakout   — closes above resistance, then closes back below in 1–3 bars
  3. Bull Trap       — breakout with low volume, trapped longs, reverses sharply
  4. Bear Trap       — breakdown with low volume, trapped shorts, reverses sharply
  5. Pump & Dump     — rapid price rise on thin volume, no accumulation base
  6. Churn           — high volume + narrow range = distribution disguised as consolidation
  7. OI Trap         — rising OI + rising price, sudden OI drop = operators exiting
  8. Volume-Price    — price at new high but volume declining (hidden selling)
     Divergence

Philosophy:
  Manipulation is NOT confirmed by a single signal.
  Multiple INDEPENDENT signals = confirmed manipulation = SKIP trade.
  One signal = warning, two = strong warning, three+ = block.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

MIN_CANDLES = 30
# ATR multiplier: spike > N * ATR is a potential stop hunt
STOP_HUNT_ATR_MULT = 2.5
# Volume threshold: breakout on < this fraction of avg vol = suspect
LOW_VOLUME_BREAKOUT_THRESH = 0.8
# How many bars to look back for swing levels
SWING_WINDOW = 15


@dataclass
class ManipulationFlags:
    # Individual flags (each independently detected)
    stop_hunt_bullish: bool = False      # Stop hunt below (operators bought, good for longs)
    stop_hunt_bearish: bool = False      # Stop hunt above (operators sold, good for shorts)
    fake_breakout_up: bool = False       # False break above resistance (bear signal)
    fake_breakdown_down: bool = False    # False break below support (bull signal)
    bull_trap: bool = False              # Retail longs trapped above resistance
    bear_trap: bool = False              # Retail shorts trapped below support
    pump_detected: bool = False          # Rapid rise without accumulation base
    dump_detected: bool = False          # Rapid fall without distribution phase
    churn_detected: bool = False         # High vol + tiny range = hidden distribution
    vol_price_divergence_bearish: bool = False  # Price up, volume down = sell into rally
    vol_price_divergence_bullish: bool = False  # Price down, volume down = absorbing
    oi_trap_bullish: bool = False        # OI + price rose, then OI dropped = operators out

    # Summary
    manipulation_score: int = 0          # 0–10 (number of independent signals)
    block_trade: bool = False            # True if ≥ 2 manipulation signals confirmed
    warnings: list[str] = field(default_factory=list)
    operator_direction: Optional[str] = None  # "bullish" | "bearish" — what operators did

    def add_warning(self, flag_name: str, message: str) -> None:
        self.manipulation_score += 1
        self.warnings.append(f"[{flag_name}] {message}")


class ManipulationDetector:
    """
    Scans price-volume data for operator manipulation patterns.

    Call detect() before any trade. If block_trade=True, skip the signal.
    """

    def detect(self, df: pd.DataFrame, oi_series: Optional[pd.Series] = None) -> ManipulationFlags:
        """
        Run all manipulation checks.

        Args:
            df: OHLCV DataFrame (ascending date order)
            oi_series: Optional open interest series (same index as df)

        Returns:
            ManipulationFlags with block_trade=True if ≥2 signals confirmed
        """
        flags = ManipulationFlags()

        if len(df) < MIN_CANDLES:
            flags.warnings.append("Insufficient data for manipulation analysis")
            return flags

        df = df.copy().reset_index(drop=True)

        self._check_stop_hunts(df, flags)
        self._check_fake_breakouts(df, flags)
        self._check_traps(df, flags)
        self._check_pump_dump(df, flags)
        self._check_churn(df, flags)
        self._check_vol_price_divergence(df, flags)

        if oi_series is not None and len(oi_series) >= MIN_CANDLES:
            self._check_oi_trap(df, oi_series, flags)

        # Block trade if ≥ 2 independent manipulation signals
        flags.block_trade = flags.manipulation_score >= 2

        # Determine operator direction (what are operators doing?)
        bull_manip = sum([
            flags.stop_hunt_bullish,       # Operators triggered sells to buy cheap
            flags.fake_breakdown_down,     # Operators faked breakdown to accumulate
            flags.bear_trap,               # Operators trapped shorts
            flags.vol_price_divergence_bullish,  # Operators buying on down volume
        ])
        bear_manip = sum([
            flags.stop_hunt_bearish,       # Operators triggered buys to sell expensive
            flags.fake_breakout_up,        # Operators faked breakout to distribute
            flags.bull_trap,               # Operators trapped longs
            flags.vol_price_divergence_bearish,  # Operators selling on up volume
            flags.pump_detected,           # Operators pumped to sell
            flags.churn_detected,          # Operators distributing at highs
        ])

        if bull_manip > bear_manip:
            flags.operator_direction = "bullish"
        elif bear_manip > bull_manip:
            flags.operator_direction = "bearish"

        log.info(
            "manipulation.scan",
            score=flags.manipulation_score,
            block=flags.block_trade,
            operator_dir=flags.operator_direction,
            signals=[w.split("]")[0][1:] for w in flags.warnings],
        )
        return flags

    # ── 1. Stop Hunts ────────────────────────────────────────────────────

    def _check_stop_hunts(self, df: pd.DataFrame, flags: ManipulationFlags) -> None:
        """
        Stop hunt = price briefly spikes beyond a key swing level then reverses.
        The wick that exceeds the level tells you which stops were triggered.

        Bullish stop hunt: price wicks BELOW swing low, close above it.
          Operators pushed price down to buy cheap from panicking retail.
          After: expect up move.

        Bearish stop hunt: price wicks ABOVE swing high, close below it.
          Operators pushed price up to sell expensive into retail FOMO.
          After: expect down move.
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        n = len(df)

        # ATR for spike threshold
        tr = np.array([
            max(highs[i] - lows[i],
                abs(highs[i] - closes[i-1]) if i > 0 else 0,
                abs(lows[i] - closes[i-1]) if i > 0 else 0)
            for i in range(n)
        ])
        atr = float(np.mean(tr[-20:]))

        # Recent swing levels (last SWING_WINDOW bars, excluding last 3)
        swing_high = float(np.max(highs[-(SWING_WINDOW + 3):-3]))
        swing_low = float(np.min(lows[-(SWING_WINDOW + 3):-3]))

        # Check last 5 bars for spike + reversal
        for i in range(max(0, n - 5), n):
            wick_down = closes[i-1] - lows[i] if i > 0 else 0
            wick_up = highs[i] - closes[i-1] if i > 0 else 0

            # Bullish stop hunt: wick below swing low + close above it
            if (lows[i] < swing_low and
                    closes[i] > swing_low and
                    wick_down > atr * STOP_HUNT_ATR_MULT):
                flags.stop_hunt_bullish = True
                flags.add_warning(
                    "STOP_HUNT_BULL",
                    f"Wick {wick_down/atr:.1f}x ATR below swing low {swing_low:.1f}, closed above — operators bought"
                )

            # Bearish stop hunt: wick above swing high + close below it
            if (highs[i] > swing_high and
                    closes[i] < swing_high and
                    wick_up > atr * STOP_HUNT_ATR_MULT):
                flags.stop_hunt_bearish = True
                flags.add_warning(
                    "STOP_HUNT_BEAR",
                    f"Wick {wick_up/atr:.1f}x ATR above swing high {swing_high:.1f}, closed below — operators sold"
                )

    # ── 2. Fake Breakouts ────────────────────────────────────────────────

    def _check_fake_breakouts(self, df: pd.DataFrame, flags: ManipulationFlags) -> None:
        """
        Fake breakout up: closes above resistance, then closes back below within 3 bars.
        Fake breakdown down: closes below support, then closes back above within 3 bars.

        This is how operators distribute (sell) into breakout FOMO
        or accumulate (buy) into breakdown panic.
        """
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values
        n = len(df)
        avg_vol = float(np.mean(volumes[-20:]))

        # Resistance = max high of prior 20 bars (excluding last 5)
        resistance = float(np.max(highs[-(25):-5])) if n > 25 else None
        support = float(np.min(lows[-(25):-5])) if n > 25 else None

        if resistance is None or support is None:
            return

        # Look at bars 3–7 from end to check for breakout that was later reversed
        for i in range(max(0, n - 7), n - 3):
            broke_up = closes[i] > resistance
            broke_down = closes[i] < support
            breakout_vol = volumes[i]

            if broke_up:
                # Check if price came back below resistance in next 1–3 bars
                subsequent_closes = closes[i+1:min(i+4, n)]
                if any(c < resistance for c in subsequent_closes):
                    flags.fake_breakout_up = True
                    low_vol = breakout_vol < avg_vol * LOW_VOLUME_BREAKOUT_THRESH
                    flags.add_warning(
                        "FAKE_BREAKOUT_UP",
                        f"Close above resistance {resistance:.1f}, reversed in <3 bars"
                        + (" (low volume breakout — classic distribution)" if low_vol else "")
                    )

            if broke_down:
                subsequent_closes = closes[i+1:min(i+4, n)]
                if any(c > support for c in subsequent_closes):
                    flags.fake_breakdown_down = True
                    low_vol = breakout_vol < avg_vol * LOW_VOLUME_BREAKOUT_THRESH
                    flags.add_warning(
                        "FAKE_BREAKDOWN_DOWN",
                        f"Close below support {support:.1f}, reversed in <3 bars"
                        + (" (low volume — operators absorbing)" if low_vol else "")
                    )

    # ── 3. Bull / Bear Traps ─────────────────────────────────────────────

    def _check_traps(self, df: pd.DataFrame, flags: ManipulationFlags) -> None:
        """
        Bull Trap: Price closes at multi-day high (exciting retail longs)
                   but volume is LOW (no institutional confirmation).
                   Price reverses in next 1-2 bars trapping new longs.

        Bear Trap: Price closes at multi-day low (panicking retail shorts)
                   but volume is LOW. Price reverses trapping new shorts.
        """
        closes = df["close"].values
        volumes = df["volume"].values
        n = len(df)

        lookback = min(20, n - 3)
        avg_vol = float(np.mean(volumes[-lookback:]))

        for i in range(max(0, n - 5), n - 2):
            c = closes[i]
            vol = volumes[i]

            # Is this a multi-day high?
            prior_closes = closes[max(0, i-lookback):i]
            is_high = c == max(prior_closes) if len(prior_closes) > 0 else False
            is_low = c == min(prior_closes) if len(prior_closes) > 0 else False

            # Subsequent reversal (next 2 bars)
            next_closes = closes[i+1:min(i+3, n)]

            if is_high and vol < avg_vol * 0.85 and len(next_closes) > 0:
                if next_closes[-1] < c * 0.99:  # Closed >1% below the high
                    flags.bull_trap = True
                    flags.add_warning(
                        "BULL_TRAP",
                        f"New {lookback}d high at {c:.1f} on LOW volume ({vol/avg_vol:.1f}x avg), reversed — longs trapped"
                    )

            if is_low and vol < avg_vol * 0.85 and len(next_closes) > 0:
                if next_closes[-1] > c * 1.01:
                    flags.bear_trap = True
                    flags.add_warning(
                        "BEAR_TRAP",
                        f"New {lookback}d low at {c:.1f} on LOW volume ({vol/avg_vol:.1f}x avg), reversed — shorts trapped"
                    )

    # ── 4. Pump & Dump ───────────────────────────────────────────────────

    def _check_pump_dump(self, df: pd.DataFrame, flags: ManipulationFlags) -> None:
        """
        Pump: ≥5% price rise in 3 bars WITHOUT prior accumulation base.
          (Legitimate moves have a base; pumps start from nowhere.)

        Dump: ≥5% price fall in 3 bars WITHOUT prior distribution phase.
        """
        closes = df["close"].values
        volumes = df["volume"].values
        n = len(df)

        if n < 20:
            return

        avg_vol = float(np.mean(volumes[-20:]))

        for i in range(max(0, n - 5), n - 2):
            move_3bar = (closes[i] - closes[max(0, i-3)]) / closes[max(0, i-3)] * 100
            vol_during = float(np.mean(volumes[max(0, i-3):i+1]))

            # Pump: sharp up move + volume spike + no consolidation before
            if move_3bar > 5.0 and vol_during > avg_vol * 2.0:
                # Check for consolidation base (last 10 bars before)
                prior_range = (max(closes[max(0, i-13):i-3]) - min(closes[max(0, i-13):i-3])) / closes[max(0, i-10)] * 100 if i > 13 else 999
                if prior_range > 3.0:  # No tight base = pump, not organic breakout
                    flags.pump_detected = True
                    flags.add_warning(
                        "PUMP",
                        f"+{move_3bar:.1f}% in 3 bars on {vol_during/avg_vol:.1f}x volume without accumulation base"
                    )

            # Dump: sharp down move + volume spike
            if move_3bar < -5.0 and vol_during > avg_vol * 2.0:
                prior_range = (max(closes[max(0, i-13):i-3]) - min(closes[max(0, i-13):i-3])) / closes[max(0, i-10)] * 100 if i > 13 else 999
                if prior_range > 3.0:
                    flags.dump_detected = True
                    flags.add_warning(
                        "DUMP",
                        f"{move_3bar:.1f}% in 3 bars on {vol_during/avg_vol:.1f}x volume without distribution base"
                    )

    # ── 5. Churn (Hidden Distribution) ───────────────────────────────────

    def _check_churn(self, df: pd.DataFrame, flags: ManipulationFlags) -> None:
        """
        Churn = high volume + narrow price range.

        When operators are distributing, they keep price flat (so it looks
        stable to retail) while dumping volume into every buy order.

        Signal: last 5 bars have above-average volume but range < 0.5x ATR.
        """
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values
        closes = df["close"].values
        n = len(df)

        lookback = min(20, n - 5)
        avg_vol = float(np.mean(volumes[-lookback:]))
        tr = np.array([highs[i] - lows[i] for i in range(n)])
        avg_tr = float(np.mean(tr[-lookback:]))

        recent = 5
        recent_vol = float(np.mean(volumes[-recent:]))
        recent_range = float(np.mean(tr[-recent:]))

        # High volume with narrow range at recent highs = churn
        price = closes[-1]
        high_20 = float(np.max(highs[-lookback:]))

        if (recent_vol > avg_vol * 1.5 and
                recent_range < avg_tr * 0.5 and
                price > high_20 * 0.97):  # Price near recent highs
            flags.churn_detected = True
            flags.add_warning(
                "CHURN",
                f"High volume ({recent_vol/avg_vol:.1f}x avg) + narrow range at highs — hidden distribution"
            )

    # ── 6. Volume-Price Divergence ────────────────────────────────────────

    def _check_vol_price_divergence(self, df: pd.DataFrame, flags: ManipulationFlags) -> None:
        """
        Bearish: price making new 10-day highs but volume declining trend.
          Operators are selling into retail euphoria. Every up bar = distribution.

        Bullish: price making new 10-day lows but volume declining trend.
          Selling pressure is exhausting. Operators are absorbing supply.
        """
        closes = df["close"].values
        volumes = df["volume"].values
        n = len(df)
        lookback = min(15, n)

        prices = closes[-lookback:]
        vols = volumes[-lookback:]

        x = np.arange(lookback)
        price_slope = np.polyfit(x, prices, 1)[0]
        vol_slope = np.polyfit(x, vols, 1)[0]

        price_trend_pct = price_slope / prices[0] * 100 if prices[0] > 0 else 0
        vol_trend_pct = vol_slope / vols[0] * 100 if vols[0] > 0 else 0

        # Bearish: price up + volume down
        if price_trend_pct > 0.1 and vol_trend_pct < -0.1:
            flags.vol_price_divergence_bearish = True
            flags.add_warning(
                "VOL_PRICE_DIV_BEAR",
                f"Price +{price_trend_pct:.2f}%/bar trend, volume {vol_trend_pct:.2f}%/bar trend — selling into rally"
            )

        # Bullish: price down + volume also down (no more sellers)
        elif price_trend_pct < -0.1 and vol_trend_pct < -0.1:
            flags.vol_price_divergence_bullish = True
            flags.add_warning(
                "VOL_PRICE_DIV_BULL",
                f"Price {price_trend_pct:.2f}%/bar, volume {vol_trend_pct:.2f}%/bar — selling exhaustion (bullish)"
            )

    # ── 7. OI Trap ────────────────────────────────────────────────────────

    def _check_oi_trap(self, df: pd.DataFrame, oi_series: pd.Series, flags: ManipulationFlags) -> None:
        """
        OI Trap (Long Unwinding):
          Phase 1: OI rises + price rises (fresh longs entering)
          Phase 2: OI suddenly drops + price drops (operators exit, retail trapped)

        Contrarian signal for bulls: OI dropping sharply on down move
        = operators finished selling, exhaustion near.

        Long trap signal for bears: OI peaked + now dropping as price falls.
        """
        oi = oi_series.values
        prices = df["close"].values
        n = min(len(oi), len(prices))

        if n < 10:
            return

        # Recent OI trend: last 5 vs prior 5
        recent_oi = float(np.mean(oi[-5:]))
        prior_oi = float(np.mean(oi[-10:-5]))
        recent_price = float(np.mean(prices[-5:]))
        prior_price = float(np.mean(prices[-10:-5]))

        oi_change_pct = (recent_oi - prior_oi) / prior_oi * 100 if prior_oi > 0 else 0
        price_change_pct = (recent_price - prior_price) / prior_price * 100 if prior_price > 0 else 0

        # OI dropping + price dropping = longs unwinding (trapped longs exiting)
        if oi_change_pct < -5 and price_change_pct < -1:
            flags.oi_trap_bullish = True  # After long unwinding, bottom is near
            flags.add_warning(
                "OI_TRAP",
                f"OI fell {oi_change_pct:.1f}% + price fell {price_change_pct:.1f}% — long unwinding (operators exited, bottom forming?)"
            )

        # OI rising + price falling = fresh shorts (not an OI trap per se but bearish)
        # OI rising + price rising = fresh longs (continuation signal, not manipulation)
