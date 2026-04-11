from __future__ import annotations

from datetime import datetime, time
import pytz
import holidays


IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


class MarketHoursFilter:
    """
    Checks whether the NSE/BSE market is currently open:
    - Mon–Fri only
    - 09:15–15:30 IST
    - Excluding NSE trading holidays (India national holidays via `holidays` lib)
    """

    def __init__(self) -> None:
        self._nse_holidays = holidays.India(state=None)  # national holidays

    def is_open(self, dt: datetime | None = None) -> bool:
        now = dt or datetime.now(IST)
        # Weekend check
        if now.weekday() >= 5:
            return False
        # Holiday check
        if now.date() in self._nse_holidays:
            return False
        # Time check
        current_time = now.time()
        return MARKET_OPEN <= current_time <= MARKET_CLOSE

    def time_to_open(self) -> int:
        """Seconds until next market open. Returns 0 if already open."""
        now = datetime.now(IST)
        if self.is_open(now):
            return 0
        # Find next trading day
        from datetime import timedelta
        candidate = now.replace(hour=9, minute=15, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        while candidate.weekday() >= 5 or candidate.date() in self._nse_holidays:
            candidate += timedelta(days=1)
        return int((candidate - now).total_seconds())


class SymbolNormalizer:
    """
    Normalizes TradingView symbol formats to clean NSE symbols.

    Examples:
    - "NSE:RELIANCE" → "RELIANCE"
    - "BSE:500325"   → "500325"   (BSE numeric code, pass through)
    - "NIFTY50"      → "NIFTY 50" (some brokers use space)
    - "BANKNIFTY"    → "BANKNIFTY"
    """

    # TradingView sometimes uses different names than broker instrument names
    _TV_TO_NSE: dict[str, str] = {
        "NIFTY50": "NIFTY 50",
        "NIFTYBANK": "NIFTY BANK",
        "NIFTYIT": "NIFTY IT",
        "CNXMIDCAP": "NIFTY MIDCAP 100",
        "NIFTYSMALLCAP": "NIFTY SMALLCAP 100",
    }

    def normalize(self, symbol: str, exchange: str = "NSE") -> str:
        clean = symbol.upper().strip()
        # Strip exchange prefix if present
        if ":" in clean:
            clean = clean.split(":")[1]
        # Apply known aliases
        return self._TV_TO_NSE.get(clean, clean)
