"""
Indian Market Knowledge Base

Static reference data for:
  - F&O lot sizes (SEBI revised Nov 2024)
  - Weekly expiry schedule
  - Circuit breaker rules
  - Market timing constants
  - VIX/PCR interpretation thresholds
  - Key NSE API endpoints

Used by scorer, risk manager, and backtest runner for accuracy.
"""
from __future__ import annotations

from datetime import date, time
from typing import Optional


# ── Market Timings (IST) ──────────────────────────────────────────────────────

class MarketTiming:
    PRE_OPEN_START    = time(9, 0)
    PRE_OPEN_END      = time(9, 15)
    MARKET_OPEN       = time(9, 15)
    MARKET_CLOSE      = time(15, 30)
    POST_CLOSE_START  = time(15, 30)
    POST_CLOSE_END    = time(16, 0)
    AMO_START         = time(17, 0)

    # F&O
    FO_OPEN           = time(9, 15)
    FO_CLOSE          = time(15, 30)

    # Currency Derivatives (CDS)
    CDS_OPEN          = time(9, 0)
    CDS_CLOSE         = time(17, 0)

    # MCX (Multi Commodity)
    MCX_MORNING_OPEN  = time(9, 0)
    MCX_MORNING_CLOSE = time(17, 0)
    MCX_EVENING_OPEN  = time(17, 0)
    MCX_EVENING_CLOSE = time(23, 55)

    # MIS auto square-off buffer
    MIS_SQUAREOFF     = time(15, 15)


# ── F&O Lot Sizes (SEBI Revised Nov 2024) ────────────────────────────────────
# Source: SEBI circular SEBI/HO/MRD/MRD-PoD-3/P/CIR/2024/115 (Oct 2024)
# Effective: contracts expiring after November 20, 2024
# Target contract value: minimum ₹15 lakh per lot

FO_LOT_SIZES: dict[str, int] = {
    # Indices
    "NIFTY": 75,
    "BANKNIFTY": 30,
    "FINNIFTY": 65,
    "MIDCPNIFTY": 75,
    "SENSEX": 20,       # BSE
    "BANKEX": 15,       # BSE

    # NIFTY 50 top stocks (approximate — verify on NSE website)
    "RELIANCE": 250,
    "TCS": 175,
    "HDFCBANK": 550,
    "INFY": 400,
    "ICICIBANK": 700,
    "HINDUNILVR": 300,
    "ITC": 3200,
    "SBIN": 1500,
    "BHARTIARTL": 475,
    "KOTAKBANK": 400,
    "LT": 175,
    "AXISBANK": 625,
    "ASIANPAINT": 200,
    "MARUTI": 30,
    "TITAN": 175,
    "SUNPHARMA": 350,
    "BAJFINANCE": 125,
    "WIPRO": 1500,
    "ONGC": 1925,
    "NTPC": 2250,
    "TECHM": 300,
    "HCLTECH": 350,
    "NESTLEIND": 40,
    "ULTRACEMCO": 100,
    "POWERGRID": 2900,
    "M&M": 175,
    "TATAMOTORS": 550,
    "ADANIPORTS": 600,
    "JSWSTEEL": 675,
    "TATASTEEL": 5500,

    # Mid-cap popular F&O stocks
    "NAUKRI": 150,
    "PIDILITIND": 250,
    "BANKBARODA": 2350,
    "MUTHOOTFIN": 300,
    "INDUSTOWER": 2800,
    "INDIGO": 300,
    "ZOMATO": 2812,
    "PAYTM": 2000,
    "NYKAA": 1100,
}

def get_lot_size(symbol: str) -> int:
    """Return F&O lot size for a symbol. Returns 1 if not in F&O."""
    return FO_LOT_SIZES.get(symbol.upper(), 1)


# ── Weekly Expiry Schedule ────────────────────────────────────────────────────
# Source: SEBI Oct 2024 circular — one weekly expiry per exchange
# NSE retained NIFTY + BANKNIFTY weekly; others monthly only

WEEKLY_EXPIRY_DAY: dict[str, str] = {
    "NIFTY": "Thursday",
    "BANKNIFTY": "Wednesday",   # Changed from Thursday
    "FINNIFTY": "Tuesday",
    "MIDCPNIFTY": "Monday",
    "SENSEX": "Friday",         # BSE
}

# Indices with MONTHLY expiry only (weekly removed per SEBI Oct 2024)
MONTHLY_EXPIRY_ONLY = {"NIFTYIT", "NIFTYMIDCAP", "NIFTYSMALLCAP"}

def get_expiry_day(symbol: str) -> Optional[str]:
    """Return weekly expiry day for an index. None if monthly only."""
    s = symbol.upper()
    if s in MONTHLY_EXPIRY_ONLY:
        return None
    return WEEKLY_EXPIRY_DAY.get(s)


# ── Circuit Breaker Rules ────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Market-wide circuit breaker rules (index-level halts).
    Source: SEBI market-wide circuit breaker framework.
    """

    # (trigger_pct, halt_before_1pm, halt_1pm_to_230, halt_after_230)
    RULES = [
        (10, 45, 15, 0),    # 10%: 45-min halt before 1pm, 15-min 1–2:30pm, no halt after
        (15, 105, 45, 0),   # 15%: 1h45m before 1pm, 45-min 1–2:30pm, rest of day after
        (20, 999, 999, 999),# 20%: halt for remainder of day always
    ]

    @classmethod
    def get_halt_minutes(cls, trigger_pct: int, current_time: time) -> int:
        """Return halt duration in minutes for a given circuit breaker trigger."""
        for pct, before_1pm, between, after_230 in cls.RULES:
            if trigger_pct >= pct:
                if current_time < time(13, 0):
                    return before_1pm
                elif current_time < time(14, 30):
                    return between
                else:
                    return after_230
        return 0

    # Individual stock price bands
    STOCK_BANDS: dict[str, int] = {
        "T2T_SURVEILLANCE": 2,    # Trade-to-Trade category: 2% limit
        "SURVEILLANCE": 5,        # Special surveillance: 5% limit
        "MIDCAP": 10,             # Many mid/small caps: 10%
        "LARGECAP": 20,           # Most NSE-listed: 20%
        "FO_STOCKS": 0,           # F&O stocks: no band
    }


# ── VIX Thresholds (Empirical NSE) ───────────────────────────────────────────

class IndiaVIXLevel:
    COMPLACENCY   = 12.0    # < 12: dangerous low vol, mean reversion risk
    LOW           = 15.0    # 12–15: low vol, trending, sell options
    NORMAL_LOW    = 18.0    # 15–18: normal, any strategy
    NORMAL_HIGH   = 20.0    # 18–20: slightly elevated, monitor
    ELEVATED      = 25.0    # 20–25: elevated, reduce intraday size
    HIGH          = 30.0    # 25–30: high fear, long options attractive
    CRISIS        = 40.0    # 30+: crisis; COVID peak was 86, 2008 was 65+

    @classmethod
    def describe(cls, vix: float) -> tuple[str, str]:
        """Returns (level_name, trading_implication)."""
        if vix < cls.COMPLACENCY:
            return "COMPLACENCY", "Sell options, watch for reversal breakout"
        elif vix < cls.LOW:
            return "LOW", "Trend-following, short straddles/iron condors"
        elif vix < cls.NORMAL_LOW:
            return "NORMAL", "Any strategy, good for swing trading"
        elif vix < cls.NORMAL_HIGH:
            return "SLIGHTLY_ELEVATED", "Reduce intraday leverage to 80%"
        elif vix < cls.ELEVATED:
            return "ELEVATED", "Reduce size to 50–60%, avoid naked shorts"
        elif vix < cls.HIGH:
            return "HIGH", "Long options strategies, protective puts, 30–40% size"
        else:
            return "CRISIS", "Minimal exposure, long puts/calls only, 0–20% size"


# ── PCR Thresholds (NSE empirical, contrarian signal) ────────────────────────

class PCRLevel:
    """
    Put-Call Ratio interpretation for NIFTY/BANKNIFTY.
    PCR is a CONTRARIAN indicator — extremes signal reversals.
    """

    STRONGLY_BEARISH = 0.7    # < 0.7: too many calls, crowded long
    MILDLY_BEARISH   = 0.9    # 0.7–0.9: mild call dominance
    NEUTRAL_LOW      = 1.1    # 0.9–1.1: balanced
    MILDLY_BULLISH   = 1.3    # 1.1–1.3: mild put dominance
    BULLISH          = 1.5    # 1.3–1.5: put writers active = support
    STRONGLY_BULLISH = 1.5    # > 1.5: extreme put buying = panic = buy

    @classmethod
    def interpret(cls, pcr: float) -> tuple[str, str]:
        """Returns (signal, description)."""
        if pcr < cls.STRONGLY_BEARISH:
            return "BEARISH", "Excessive call buying — crowded long, risky to buy"
        elif pcr < cls.MILDLY_BEARISH:
            return "MILDLY_BEARISH", "Mild call dominance"
        elif pcr < cls.NEUTRAL_LOW:
            return "NEUTRAL", "Balanced options market"
        elif pcr < cls.MILDLY_BULLISH:
            return "MILDLY_BULLISH", "Mild put buying — slight support"
        elif pcr < cls.STRONGLY_BULLISH:
            return "BULLISH", "Put writers defending — floor support"
        else:
            return "STRONGLY_BULLISH", "Extreme put buying — panic, contrarian buy"


# ── NSE API Endpoints ────────────────────────────────────────────────────────

class NSEEndpoints:
    BASE = "https://www.nseindia.com"
    API  = f"{BASE}/api"

    # Market data
    ALL_INDICES        = f"{API}/allIndices"
    MARKET_STATUS      = f"{API}/marketStatus"

    # Option chain
    OC_INDICES         = f"{API}/option-chain-indices?symbol={{symbol}}"
    OC_EQUITIES        = f"{API}/option-chain-equities?symbol={{symbol}}"

    # Institutional data
    FII_DII            = f"{API}/fiidiiTradeReact"

    # Historical
    EQUITY_HISTORICAL  = f"{API}/historical/cm/equity?symbol={{symbol}}&series=EQ&from={{from_date}}&to={{to_date}}"

    # OI analysis
    OI_SPURTS          = f"{API}/liveanalysis-oi/oi-spurts"
    OI_MOST_ACTIVE     = f"{API}/live-analysis-most-active-underlying-fo"

    # Bhavcopy (EOD bulk data)
    BHAVCOPY_EQUITY    = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"
    BHAVCOPY_FO        = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"

    # Required session headers
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.nseindia.com/",
        "Accept": "application/json, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }


# ── Sector Rotation Calendar ─────────────────────────────────────────────────

SECTOR_EVENTS: dict[str, list[str]] = {
    "Q1_Results": ["July", "August"],    # Apr–Jun quarter results
    "Q2_Results": ["October", "November"],
    "Q3_Results": ["January", "February"],
    "Q4_Results": ["April", "May"],
    "RBI_Policy": ["February", "April", "June", "August", "October", "December"],
    "Union_Budget": ["February"],        # February 1
    "Advance_Tax": ["June", "September", "December", "March"],
    "FO_Monthly_Expiry": ["Last Thursday of each month"],
}

# Sectors most sensitive to specific events
EVENT_SENSITIVE_SECTORS: dict[str, list[str]] = {
    "Union_Budget": ["Infrastructure", "Defense", "FMCG", "Auto", "Pharma"],
    "RBI_Policy": ["Banking", "NBFC", "Real_Estate", "Auto"],
    "US_Fed_Policy": ["IT", "Pharma", "Metal"],
    "Crude_Oil": ["OMCs", "Aviation", "Paints", "Tyre"],
    "USD_INR": ["IT", "Pharma", "Metal", "Textile"],
    "China_PMI": ["Metal", "Chemical"],
}
