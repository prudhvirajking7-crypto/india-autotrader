from __future__ import annotations

import pandas as pd
import structlog

try:
    import pandas_ta as ta
except ImportError:
    from utils import ta_compat as ta

log = structlog.get_logger(__name__)


class HistoricalDataFetcher:
    """
    Fetches and enriches historical OHLCV data with technical indicators.
    Uses NSEDataProvider as the primary source, falls back to yfinance.
    """

    def __init__(self) -> None:
        from data.nse_data import NSEDataProvider
        self._nse = NSEDataProvider()

    def get_with_indicators(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        indicators: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV and append requested technical indicators.

        Args:
            symbol: NSE symbol
            from_date / to_date: "YYYY-MM-DD"
            indicators: list of indicator names e.g. ["ema_9", "ema_21", "rsi_14", "supertrend"]

        Returns:
            DataFrame with OHLCV + indicator columns
        """
        df = self._nse.get_equity_ohlcv(symbol, from_date, to_date)
        if df.empty:
            log.warning("historical.empty", symbol=symbol)
            return df

        if not indicators:
            return df

        df = df.set_index("date")

        for ind in indicators:
            df = self._add_indicator(df, ind)

        return df.reset_index()

    def _add_indicator(self, df: pd.DataFrame, indicator: str) -> pd.DataFrame:
        """Append a single indicator column to the DataFrame."""
        parts = indicator.lower().split("_")
        name = parts[0]

        try:
            if name == "ema":
                period = int(parts[1]) if len(parts) > 1 else 9
                df[f"ema_{period}"] = ta.ema(df["close"], length=period)

            elif name == "sma":
                period = int(parts[1]) if len(parts) > 1 else 20
                df[f"sma_{period}"] = ta.sma(df["close"], length=period)

            elif name == "rsi":
                period = int(parts[1]) if len(parts) > 1 else 14
                df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)

            elif name == "macd":
                macd = ta.macd(df["close"])
                df = pd.concat([df, macd], axis=1)

            elif name == "bb" or name == "bbands":
                bb = ta.bbands(df["close"])
                df = pd.concat([df, bb], axis=1)

            elif name == "supertrend":
                st = ta.supertrend(df["high"], df["low"], df["close"])
                df = pd.concat([df, st], axis=1)

            elif name == "atr":
                period = int(parts[1]) if len(parts) > 1 else 14
                df[f"atr_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)

            elif name == "vwap":
                df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

            else:
                log.warning("historical.unknown_indicator", indicator=indicator)

        except Exception as e:
            log.error("historical.indicator_failed", indicator=indicator, error=str(e))

        return df
