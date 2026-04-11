"""
Technical Analysis Compatibility Layer

pandas-ta is no longer on PyPI. This module wraps the `ta` library
(which IS available) with a pandas_ta-compatible API so the rest of
the codebase doesn't need to change.

Supported functions (drop-in replacements):
    ema(series, length)  → pd.Series
    sma(series, length)  → pd.Series
    rsi(series, length)  → pd.Series
    macd(series)         → pd.DataFrame with MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    bbands(series, length, std) → pd.DataFrame with BBU_N_S, BBL_N_S, BBB_N_S
    supertrend(high, low, close, length, multiplier) → pd.DataFrame with SUPERTd_N_M
    atr(high, low, close, length) → pd.Series
    vwap(high, low, close, volume) → pd.Series
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ── EMA ──────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, length: int = 20, **kwargs) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


# ── SMA ──────────────────────────────────────────────────────────────────────

def sma(series: pd.Series, length: int = 20, **kwargs) -> pd.Series:
    return series.rolling(window=length).mean()


# ── RSI ──────────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, length: int = 14, **kwargs) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=length - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=length - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ─────────────────────────────────────────────────────────────────────

def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    **kwargs,
) -> pd.DataFrame:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        f"MACD_{fast}_{slow}_{signal}": macd_line,
        f"MACDs_{fast}_{slow}_{signal}": signal_line,
        f"MACDh_{fast}_{slow}_{signal}": histogram,
    })


# ── Bollinger Bands ──────────────────────────────────────────────────────────

def bbands(
    series: pd.Series,
    length: int = 20,
    std: float = 2.0,
    **kwargs,
) -> pd.DataFrame:
    mid = series.rolling(length).mean()
    stddev = series.rolling(length).std()
    upper = mid + std * stddev
    lower = mid - std * stddev
    bw = ((upper - lower) / mid) * 100  # Bandwidth %
    pct = (series - lower) / (upper - lower)  # %B
    k = f"{length}_{std:.1f}".replace(".0", "")
    return pd.DataFrame({
        f"BBU_{k}": upper,
        f"BBM_{k}": mid,
        f"BBL_{k}": lower,
        f"BBB_{k}": bw,
        f"BBP_{k}": pct,
    })


# ── ATR ──────────────────────────────────────────────────────────────────────

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    **kwargs,
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, adjust=False).mean()


# ── Supertrend ────────────────────────────────────────────────────────────────

def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 7,
    multiplier: float = 3.0,
    **kwargs,
) -> pd.DataFrame:
    atr_vals = atr(high, low, close, length)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr_vals
    lower_band = hl2 - multiplier * atr_vals

    supertrend_vals = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    upper_band_adj = upper_band.copy()
    lower_band_adj = lower_band.copy()

    for i in range(1, len(close)):
        # Adjust upper band
        if upper_band.iloc[i] < upper_band_adj.iloc[i - 1] or close.iloc[i - 1] > upper_band_adj.iloc[i - 1]:
            upper_band_adj.iloc[i] = upper_band.iloc[i]
        else:
            upper_band_adj.iloc[i] = upper_band_adj.iloc[i - 1]

        # Adjust lower band
        if lower_band.iloc[i] > lower_band_adj.iloc[i - 1] or close.iloc[i - 1] < lower_band_adj.iloc[i - 1]:
            lower_band_adj.iloc[i] = lower_band.iloc[i]
        else:
            lower_band_adj.iloc[i] = lower_band_adj.iloc[i - 1]

        # Direction: 1 = bullish (price above lower band), -1 = bearish
        if supertrend_vals.iloc[i - 1] == upper_band_adj.iloc[i - 1]:
            direction.iloc[i] = -1 if close.iloc[i] <= upper_band_adj.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] >= lower_band_adj.iloc[i] else -1

        supertrend_vals.iloc[i] = (
            lower_band_adj.iloc[i] if direction.iloc[i] == 1 else upper_band_adj.iloc[i]
        )

    k = f"{length}_{multiplier:.1f}".replace(".0", "")
    return pd.DataFrame({
        f"SUPERT_{k}": supertrend_vals,
        f"SUPERTd_{k}": direction,
        f"SUPERTl_{k}": lower_band_adj,
        f"SUPERTu_{k}": upper_band_adj,
    })


# ── VWAP ─────────────────────────────────────────────────────────────────────

def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    **kwargs,
) -> pd.Series:
    tp = (high + low + close) / 3
    cumvol = volume.cumsum()
    cumtpvol = (tp * volume).cumsum()
    return cumtpvol / cumvol.replace(0, np.nan)
