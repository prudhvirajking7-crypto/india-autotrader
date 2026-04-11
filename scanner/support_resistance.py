"""
Support & Resistance Level Finder

Identifies high-confidence S/R levels from multiple independent sources.
More sources confirming the same price zone = stronger level.

Sources (in priority order):
  1. Previous Day High / Low (PDH/PDL)   — most respected intraday levels
  2. Swing Highs / Lows                  — structural S/R (last 50 bars)
  3. Volume Point of Control (POC)       — price where most volume traded
  4. Order Block zones                   — from SMC (institutional memory)
  5. Round / Psychological numbers       — 100, 500, 1000, 2000 etc.
  6. Weekly / Monthly pivots             — floor trader pivots (P, R1, S1)

Every level gets a "strength" score 1–5:
  1 = one source only
  2 = two sources agree
  3 = three sources (very strong)
  4+ = cluster zone (extremely strong — breakout or bounce imminent)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# Zone tolerance: levels within this % of each other are merged
ZONE_TOLERANCE_PCT = 0.3

# How many bars to look back for swing detection
SWING_LOOKBACK = 50
SWING_WINDOW = 5  # Local max/min over ±5 bars

# Minimum volume at a level to count as POC
MIN_POC_VOLUME_PCT = 0.05  # Level must have ≥5% of total volume


@dataclass
class SRLevel:
    price: float
    level_type: str          # "support" | "resistance"
    strength: int            # 1–5 (count of confirming sources)
    sources: list[str] = field(default_factory=list)
    # How recently was this level tested? Lower = more recent
    bars_since_test: int = 0
    # Is this level currently being tested (price within 0.5%)?
    being_tested: bool = False


@dataclass
class SRAnalysis:
    # Ordered by proximity to current price
    support_levels: list[SRLevel] = field(default_factory=list)
    resistance_levels: list[SRLevel] = field(default_factory=list)

    # Key individual levels
    pdh: Optional[float] = None   # Previous Day High
    pdl: Optional[float] = None   # Previous Day Low
    poc: Optional[float] = None   # Volume Point of Control

    # Nearest S/R to current price
    nearest_support: Optional[SRLevel] = None
    nearest_resistance: Optional[SRLevel] = None

    # Zone descriptions
    current_zone: str = "between"   # "at_support" | "at_resistance" | "between"

    # Pivot levels (floor trader)
    pivot: Optional[float] = None
    r1: Optional[float] = None
    s1: Optional[float] = None
    r2: Optional[float] = None
    s2: Optional[float] = None


class SRFinder:
    """
    Identifies support and resistance levels from OHLCV data.

    Returns levels ordered by strength and proximity to current price.
    """

    def find(self, df: pd.DataFrame) -> SRAnalysis:
        """
        Compute all S/R levels for a symbol.

        Args:
            df: OHLCV DataFrame (ascending date, at minimum 30 rows)
                Expects columns: open, high, low, close, volume, date

        Returns:
            SRAnalysis with all levels found.
        """
        if len(df) < 30:
            return SRAnalysis()

        df = df.copy().reset_index(drop=True)
        price = float(df["close"].iloc[-1])
        result = SRAnalysis()

        # Gather raw price levels from all sources
        raw_levels: list[tuple[float, str, str]] = []  # (price, type, source_name)

        self._add_pdh_pdl(df, result, raw_levels)
        self._add_swing_levels(df, raw_levels)
        self._add_volume_poc(df, result, raw_levels)
        self._add_round_numbers(price, raw_levels)
        self._add_pivot_points(df, result, raw_levels)

        # Merge nearby levels into zones
        merged = self._merge_levels(raw_levels)

        # Classify as support or resistance relative to current price
        for price_level, type_, strength, sources in merged:
            if price_level < price * 0.998:  # Support = below price
                sr = SRLevel(
                    price=price_level,
                    level_type="support",
                    strength=strength,
                    sources=sources,
                    bars_since_test=self._bars_since_test(df, price_level, "support"),
                    being_tested=(abs(price - price_level) / price) < 0.005,
                )
                result.support_levels.append(sr)
            elif price_level > price * 1.002:  # Resistance = above price
                sr = SRLevel(
                    price=price_level,
                    level_type="resistance",
                    strength=strength,
                    sources=sources,
                    bars_since_test=self._bars_since_test(df, price_level, "resistance"),
                    being_tested=(abs(price - price_level) / price) < 0.005,
                )
                result.resistance_levels.append(sr)

        # Sort: supports descending (nearest first), resistances ascending
        result.support_levels.sort(key=lambda x: x.price, reverse=True)
        result.resistance_levels.sort(key=lambda x: x.price)

        if result.support_levels:
            result.nearest_support = result.support_levels[0]
        if result.resistance_levels:
            result.nearest_resistance = result.resistance_levels[0]

        # Determine current zone
        if result.nearest_support and abs(price - result.nearest_support.price) / price < 0.005:
            result.current_zone = "at_support"
        elif result.nearest_resistance and abs(price - result.nearest_resistance.price) / price < 0.005:
            result.current_zone = "at_resistance"
        else:
            result.current_zone = "between"

        log.info(
            "sr.found",
            supports=len(result.support_levels),
            resistances=len(result.resistance_levels),
            nearest_sup=result.nearest_support.price if result.nearest_support else None,
            nearest_res=result.nearest_resistance.price if result.nearest_resistance else None,
            zone=result.current_zone,
        )
        return result

    # ── Source: Previous Day High/Low ─────────────────────────────────────

    def _add_pdh_pdl(self, df: pd.DataFrame, result: SRAnalysis, levels: list) -> None:
        """
        Previous Day High/Low are the single most watched intraday levels.
        All professional traders and algos reference these.
        """
        if len(df) < 2:
            return

        # Group by date to get yesterday's candles
        if "date" in df.columns:
            df["_date"] = pd.to_datetime(df["date"]).dt.date
            grouped = df.groupby("_date")
            dates = sorted(grouped.groups.keys())
            if len(dates) >= 2:
                yesterday = dates[-2]
                y_data = grouped.get_group(yesterday)
                result.pdh = float(y_data["high"].max())
                result.pdl = float(y_data["low"].min())
        else:
            # Daily data: second-to-last row is "yesterday"
            result.pdh = float(df["high"].iloc[-2])
            result.pdl = float(df["low"].iloc[-2])

        if result.pdh:
            levels.append((result.pdh, "resistance", "PDH"))
        if result.pdl:
            levels.append((result.pdl, "support", "PDL"))

    # ── Source: Swing Highs and Lows ─────────────────────────────────────

    def _add_swing_levels(self, df: pd.DataFrame, levels: list) -> None:
        """
        Swing highs = price reached a local max then pulled back.
        Swing lows = price reached a local min then bounced.

        These represent points where buyers/sellers previously dominated.
        The more times a level was tested and held, the stronger it is.
        """
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        lookback = min(SWING_LOOKBACK, n - SWING_WINDOW - 1)

        for i in range(SWING_WINDOW, lookback):
            window_h = highs[i - SWING_WINDOW: i + SWING_WINDOW + 1]
            window_l = lows[i - SWING_WINDOW: i + SWING_WINDOW + 1]

            if highs[i] == max(window_h):
                levels.append((float(highs[i]), "resistance", f"SwingHigh@bar{n-i}"))
            if lows[i] == min(window_l):
                levels.append((float(lows[i]), "support", f"SwingLow@bar{n-i}"))

    # ── Source: Volume Point of Control ──────────────────────────────────

    def _add_volume_poc(self, df: pd.DataFrame, result: SRAnalysis, levels: list) -> None:
        """
        POC = price level where the most volume was traded in the lookback period.
        This is the strongest magnet — price gravitates toward it.
        Also marks Value Area High/Low (70% of volume).
        """
        lookback = min(30, len(df))
        recent = df.tail(lookback)

        highs = recent["high"].values
        lows = recent["low"].values
        volumes = recent["volume"].values
        closes = recent["close"].values

        # Build volume histogram across price range
        price_min = float(lows.min())
        price_max = float(highs.max())

        if price_max <= price_min:
            return

        n_bins = 50
        bins = np.linspace(price_min, price_max, n_bins + 1)
        vol_profile = np.zeros(n_bins)

        for i in range(len(recent)):
            # Distribute candle volume across bins it spans
            for b in range(n_bins):
                bin_low = bins[b]
                bin_high = bins[b + 1]
                overlap = min(highs[i], bin_high) - max(lows[i], bin_low)
                if overlap > 0:
                    candle_range = highs[i] - lows[i] if highs[i] > lows[i] else 1
                    vol_profile[b] += volumes[i] * (overlap / candle_range)

        poc_bin = int(np.argmax(vol_profile))
        poc_price = float((bins[poc_bin] + bins[poc_bin + 1]) / 2)
        result.poc = poc_price
        levels.append((poc_price, "both", "VolumePOC"))

    # ── Source: Round / Psychological Numbers ─────────────────────────────

    def _add_round_numbers(self, price: float, levels: list) -> None:
        """
        Retail and institutional orders cluster at round numbers.
        These act as magnets and reaction zones.

        Interval depends on price:
          < 100:   multiples of 5
          100–500: multiples of 10
          500–2000: multiples of 50
          > 2000:  multiples of 100 or 500
        """
        if price < 100:
            interval = 5
        elif price < 500:
            interval = 10
        elif price < 2000:
            interval = 50
        elif price < 10000:
            interval = 100
        else:
            interval = 500

        # Find round numbers within ±10% of current price
        low = price * 0.90
        high = price * 1.10

        n_low = int(low / interval)
        n_high = int(high / interval) + 2

        for n in range(n_low, n_high):
            level = n * interval
            if low <= level <= high:
                level_type = "resistance" if level > price else "support"
                levels.append((float(level), level_type, f"Round{level}"))

    # ── Source: Floor Trader Pivots ───────────────────────────────────────

    def _add_pivot_points(self, df: pd.DataFrame, result: SRAnalysis, levels: list) -> None:
        """
        Standard floor trader pivot based on previous day's HLC.
        Widely used by institutional desks and HFTs.

        P  = (H + L + C) / 3
        R1 = 2P - L
        S1 = 2P - H
        R2 = P + (H - L)
        S2 = P - (H - L)
        """
        if len(df) < 2:
            return

        try:
            prev = df.iloc[-2]
            H = float(prev["high"])
            L = float(prev["low"])
            C = float(prev["close"])

            P = (H + L + C) / 3
            R1 = 2 * P - L
            S1 = 2 * P - H
            R2 = P + (H - L)
            S2 = P - (H - L)

            result.pivot = round(P, 2)
            result.r1 = round(R1, 2)
            result.s1 = round(S1, 2)
            result.r2 = round(R2, 2)
            result.s2 = round(S2, 2)

            levels.append((P, "both", "Pivot"))
            levels.append((R1, "resistance", "R1"))
            levels.append((S1, "support", "S1"))
            levels.append((R2, "resistance", "R2"))
            levels.append((S2, "support", "S2"))
        except Exception:
            pass

    # ── Level Merging ─────────────────────────────────────────────────────

    def _merge_levels(self, raw: list) -> list[tuple[float, str, int, list[str]]]:
        """
        Merge raw (price, type, source) triples into zones.
        Levels within ZONE_TOLERANCE_PCT of each other → single zone.
        Returns (avg_price, type, strength, sources).
        """
        if not raw:
            return []

        # Sort by price
        sorted_raw = sorted(raw, key=lambda x: x[0])

        merged = []
        cluster_prices = [sorted_raw[0][0]]
        cluster_types = [sorted_raw[0][1]]
        cluster_sources = [sorted_raw[0][2]]

        for price, typ, source in sorted_raw[1:]:
            last_price = cluster_prices[-1]
            if abs(price - last_price) / last_price * 100 <= ZONE_TOLERANCE_PCT:
                cluster_prices.append(price)
                cluster_types.append(typ)
                cluster_sources.append(source)
            else:
                # Finalize current cluster
                avg_price = float(np.mean(cluster_prices))
                # Determine type: if any PDH/PDL or swing source, use that type
                lvl_type = "both" if "both" in cluster_types else (
                    "support" if cluster_types.count("support") > cluster_types.count("resistance")
                    else "resistance"
                )
                merged.append((
                    round(avg_price, 2),
                    lvl_type,
                    len(cluster_prices),
                    list(set(cluster_sources)),
                ))
                cluster_prices = [price]
                cluster_types = [typ]
                cluster_sources = [source]

        # Last cluster
        if cluster_prices:
            avg_price = float(np.mean(cluster_prices))
            lvl_type = "both" if "both" in cluster_types else (
                "support" if cluster_types.count("support") > cluster_types.count("resistance")
                else "resistance"
            )
            merged.append((
                round(avg_price, 2),
                lvl_type,
                len(cluster_prices),
                list(set(cluster_sources)),
            ))

        return merged

    def _bars_since_test(self, df: pd.DataFrame, level: float, level_type: str) -> int:
        """How many bars ago was this level last tested (price came within 0.5%)."""
        prices = df["close"].values
        for i in range(len(prices) - 1, -1, -1):
            if abs(prices[i] - level) / level < 0.005:
                return len(prices) - 1 - i
        return len(prices)  # Never tested in lookback
