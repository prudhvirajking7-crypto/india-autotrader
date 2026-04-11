from __future__ import annotations

import time
import structlog
import pandas as pd
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

log = structlog.get_logger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


class NSEDataProvider:
    """
    NSE market data provider using jugaad-data (primary) with yfinance fallback.

    Provides:
    - NIFTY50 / NIFTY BANK constituent lists
    - NSE equity OHLCV (daily)
    - F&O instrument details (lot sizes, expiry dates)
    - Option chain snapshots
    """

    def get_nifty50_symbols(self) -> list[str]:
        """Return current NIFTY 50 constituents."""
        try:
            from jugaad_data.nse import NSELive
            nse = NSELive()
            data = nse.equities("NIFTY 50")
            return [row["symbol"] for row in data["data"]]
        except Exception as e:
            log.warning("nse.nifty50.fallback", error=str(e))
            # Hardcoded fallback for offline / rate-limited scenarios
            return [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
                "SUNPHARMA", "BAJFINANCE", "WIPRO", "ONGC", "NTPC",
            ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=settings.nse_rate_limit_delay, max=5))
    def get_equity_ohlcv(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetch daily OHLCV for an NSE equity symbol.

        Args:
            symbol: NSE symbol e.g. "RELIANCE"
            from_date: "YYYY-MM-DD"
            to_date: "YYYY-MM-DD"

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        try:
            from jugaad_data.nse import stock_df
            from datetime import datetime

            df = stock_df(
                symbol=symbol,
                from_date=datetime.strptime(from_date, "%Y-%m-%d").date(),
                to_date=datetime.strptime(to_date, "%Y-%m-%d").date(),
                series="EQ",
            )
            df = df.rename(columns={
                "DATE": "date", "OPEN": "open", "HIGH": "high",
                "LOW": "low", "CLOSE": "close", "VOLUME": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)

        except Exception as e:
            log.warning("nse.ohlcv.jugaad_failed", symbol=symbol, error=str(e))
            return self._yfinance_fallback(symbol, from_date, to_date)

    def _yfinance_fallback(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        import yfinance as yf
        ticker = f"{symbol}.NS"
        df = yf.download(ticker, start=from_date, end=to_date, progress=False)
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        df = df.reset_index()
        # Flatten MultiIndex columns returned by newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)

    def get_option_chain(self, symbol: str) -> dict:
        """Fetch current option chain for a symbol (NSE equity or index)."""
        time.sleep(settings.nse_rate_limit_delay)
        try:
            from jugaad_data.nse import NSELive
            nse = NSELive()
            return nse.equityDerivatives(symbol)
        except Exception as e:
            log.error("nse.option_chain.failed", symbol=symbol, error=str(e))
            return {}

    def get_fo_instrument_details(self, symbol: str) -> dict:
        """Return lot size and available expiry dates for F&O symbol."""
        from intelligence.knowledge_base import get_lot_size, get_expiry_day
        return {
            "lot_size": get_lot_size(symbol),
            "expiry_day": get_expiry_day(symbol),
            "symbol": symbol,
        }
