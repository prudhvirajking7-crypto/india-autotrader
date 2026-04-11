"""
Indian Market Fee Model — FY2025 Accurate Charges
Based on Finance Act 2024, SEBI circulars, and NSE/BSE official fee schedules.

All rates verified against:
- STT: Section 107 of Finance Act 2004 (amended Budget 2024)
- Exchange charges: NSE/BSE official tariff sheets
- SEBI charges: SEBI circular
- Stamp duty: Indian Stamp Act (state-wise, using Maharashtra as baseline)
- GST: 18% on brokerage + exchange charges (not on STT/stamp)

Usage:
    from backtest.fees import IndianFeeModel
    fees = IndianFeeModel()
    cost = fees.total_round_trip_cost("equity_intraday", entry=500, exit=510, qty=100)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Segment(str, Enum):
    EQUITY_DELIVERY = "equity_delivery"
    EQUITY_INTRADAY = "equity_intraday"
    FO_FUTURES = "fo_futures"
    FO_OPTIONS_BUY = "fo_options_buy"
    FO_OPTIONS_SELL = "fo_options_sell"
    CURRENCY_FUTURES = "currency_futures"
    CURRENCY_OPTIONS = "currency_options"


@dataclass
class TradeCost:
    segment: str
    turnover: float
    brokerage: float
    stt: float
    exchange_charges: float
    sebi_charges: float
    stamp_duty: float
    gst: float
    total: float
    total_bps: float       # basis points of turnover
    note: str = ""


class IndianFeeModel:
    """
    Accurate FY2025 cost model for NSE/BSE segments.

    Brokerage assumed: ₹20 flat per order (Zerodha/Upstox/discount brokers).
    Adjust FLAT_BROKERAGE_PER_ORDER if using percentage-based brokers.
    """

    FLAT_BROKERAGE_PER_ORDER = 20.0   # INR per order (one side)
    GST_RATE = 0.18
    SEBI_CHARGE_PER_CRORE = 10.0      # ₹10 per crore of turnover

    # ── Segment-specific rates ────────────────────────────────────────────

    _RATES = {
        Segment.EQUITY_DELIVERY: {
            "stt_buy": 0.001,         # 0.1% both sides
            "stt_sell": 0.001,
            "exchange_nse": 0.0000335,  # 0.00335% both sides
            "exchange_bse": 0.0000375,
            "stamp_duty_buy": 0.00015,  # 0.015% on buy side
            "stamp_duty_sell": 0.0,
            "description": "CNC delivery — T+1 settlement",
        },
        Segment.EQUITY_INTRADAY: {
            "stt_buy": 0.0,           # STT only on sell side for intraday
            "stt_sell": 0.00025,      # 0.025% on sell side
            "exchange_nse": 0.0000335,
            "exchange_bse": 0.0000375,
            "stamp_duty_buy": 0.00003,  # 0.003% on buy side
            "stamp_duty_sell": 0.0,
            "description": "MIS intraday — squared off same day",
        },
        Segment.FO_FUTURES: {
            "stt_buy": 0.0,           # STT only on sell side
            "stt_sell": 0.0000125,    # 0.00125% on sell side (on notional)
            "exchange_nse": 0.0000188,  # 0.00188% on notional
            "exchange_bse": 0.0000188,
            "stamp_duty_buy": 0.00003,  # 0.003% on buy side (on premium/notional)
            "stamp_duty_sell": 0.0,
            "description": "F&O Futures — on notional value",
        },
        Segment.FO_OPTIONS_BUY: {
            "stt_buy": 0.0,           # No STT on options buy side
            "stt_sell": 0.0,          # STT on sell covered under FO_OPTIONS_SELL
            "exchange_nse": 0.0005,   # 0.05% on premium
            "exchange_bse": 0.0005,
            "stamp_duty_buy": 0.00003,  # 0.003% on premium
            "stamp_duty_sell": 0.0,
            "description": "F&O Options — Buy side (on premium value)",
        },
        Segment.FO_OPTIONS_SELL: {
            "stt_buy": 0.0,
            "stt_sell": 0.000625,     # 0.0625% on sell premium (Budget 2024 revised upward)
            "exchange_nse": 0.0005,   # 0.05% on premium
            "exchange_bse": 0.0005,
            "stamp_duty_buy": 0.0,
            "stamp_duty_sell": 0.0,
            "description": "F&O Options — Sell side (STT on premium, dominates cost)",
        },
        Segment.CURRENCY_FUTURES: {
            "stt_buy": 0.0,           # No STT on currency
            "stt_sell": 0.0,
            "exchange_nse": 0.0000009,  # ₹9 per crore
            "exchange_bse": 0.0000009,
            "stamp_duty_buy": 0.00003,
            "stamp_duty_sell": 0.0,
            "description": "Currency futures — USD/INR etc.",
        },
        Segment.CURRENCY_OPTIONS: {
            "stt_buy": 0.0,
            "stt_sell": 0.0,
            "exchange_nse": 0.000035,  # ₹35 per lakh on premium
            "exchange_bse": 0.000035,
            "stamp_duty_buy": 0.00003,
            "stamp_duty_sell": 0.0,
            "description": "Currency options",
        },
    }

    def calculate(
        self,
        segment: Segment | str,
        turnover: float,
        is_buy: bool = True,
        exchange: str = "NSE",
    ) -> TradeCost:
        """
        Calculate all charges for one side of a trade.

        Args:
            segment: Market segment
            turnover: Trade value in INR (price × qty for equity, premium × qty for options)
            is_buy: True for buy, False for sell
            exchange: NSE or BSE
        """
        seg = Segment(segment) if isinstance(segment, str) else segment
        rates = self._RATES[seg]

        exchange_key = "exchange_nse" if exchange.upper() == "NSE" else "exchange_bse"

        stt = turnover * (rates["stt_buy"] if is_buy else rates["stt_sell"])
        exchange_charge = turnover * rates[exchange_key]
        sebi_charge = turnover / 1e7 * self.SEBI_CHARGE_PER_CRORE  # ₹10 per crore
        stamp_duty = turnover * (rates["stamp_duty_buy"] if is_buy else rates["stamp_duty_sell"])
        brokerage = self.FLAT_BROKERAGE_PER_ORDER
        gst = (brokerage + exchange_charge) * self.GST_RATE

        total = stt + exchange_charge + sebi_charge + stamp_duty + brokerage + gst
        total_bps = (total / turnover) * 10000 if turnover > 0 else 0

        return TradeCost(
            segment=seg.value,
            turnover=turnover,
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange_charge,
            sebi_charges=sebi_charge,
            stamp_duty=stamp_duty,
            gst=gst,
            total=total,
            total_bps=total_bps,
            note=rates["description"],
        )

    def total_round_trip_cost(
        self,
        segment: Segment | str,
        entry_price: float,
        exit_price: float,
        qty: int,
        exchange: str = "NSE",
    ) -> float:
        """
        Total cost (INR) for a complete buy + sell round trip.

        For options: entry_price and exit_price are premium values (not notional).
        For futures/equity: they are the actual traded prices.
        """
        entry_turnover = entry_price * qty
        exit_turnover = exit_price * qty

        entry_cost = self.calculate(segment, entry_turnover, is_buy=True, exchange=exchange)
        exit_cost = self.calculate(segment, exit_turnover, is_buy=False, exchange=exchange)

        return entry_cost.total + exit_cost.total

    def vectorbt_fee_config(self, segment: Segment | str) -> dict:
        """
        Return fee configuration compatible with vectorbt Portfolio.from_signals().

        vectorbt uses:
          - fees: percentage of turnover (applied both sides)
          - fixed_fees: flat INR per trade (applied both sides)
          - slippage: percentage of turnover (applied both sides)
        """
        seg = Segment(segment) if isinstance(segment, str) else segment
        rates = self._RATES[seg]

        # Approximate symmetric fee rate for vectorbt
        # (vectorbt applies same % to buy and sell, so we use average)
        avg_stt = (rates["stt_buy"] + rates["stt_sell"]) / 2
        exchange_rate = rates["exchange_nse"]
        stamp_avg = (rates["stamp_duty_buy"] + rates["stamp_duty_sell"]) / 2
        sebi_rate = 1e-7  # ₹10/crore = negligible

        percentage_fee = avg_stt + exchange_rate + stamp_avg + sebi_rate
        # Add GST on exchange component
        percentage_fee += exchange_rate * self.GST_RATE

        # Slippage estimate by liquidity tier
        slippage_map = {
            Segment.EQUITY_DELIVERY: 0.0005,    # 5 bps — large cap delivery
            Segment.EQUITY_INTRADAY: 0.001,     # 10 bps — intraday spread
            Segment.FO_FUTURES: 0.0005,         # 5 bps — liquid futures
            Segment.FO_OPTIONS_BUY: 0.002,      # 20 bps — options spread
            Segment.FO_OPTIONS_SELL: 0.002,
            Segment.CURRENCY_FUTURES: 0.0002,
            Segment.CURRENCY_OPTIONS: 0.001,
        }

        return {
            "fees": percentage_fee,
            "fixed_fees": self.FLAT_BROKERAGE_PER_ORDER,
            "slippage": slippage_map.get(seg, 0.001),
        }


# ── Convenience functions ─────────────────────────────────────────────────────

def get_fee_config(segment: str = "equity_intraday") -> dict:
    """Shorthand for vectorbt integration."""
    model = IndianFeeModel()
    return model.vectorbt_fee_config(Segment(segment))


# ── Tax on profits ────────────────────────────────────────────────────────────

class TaxCalculator:
    """
    Capital gains tax computation for Indian traders (FY2025).
    Budget 2024: STCG 20%, LTCG 12.5% above ₹1.25L, F&O as business income.
    """

    LTCG_EXEMPTION = 125_000   # ₹1.25 lakh per FY
    LTCG_RATE = 0.125          # 12.5%
    STCG_RATE = 0.20           # 20% (revised from 15% in Budget July 2024)

    def equity_delivery_tax(self, profit: float, holding_days: int, income_slab_rate: float = 0.30) -> float:
        """Calculate tax on equity delivery profit."""
        if holding_days > 365:
            taxable = max(0, profit - self.LTCG_EXEMPTION)
            return taxable * self.LTCG_RATE
        else:
            return profit * self.STCG_RATE if profit > 0 else 0

    def fo_intraday_tax(self, profit: float, income_slab_rate: float = 0.30) -> float:
        """F&O and intraday profits taxed as business income at slab rate."""
        return profit * income_slab_rate if profit > 0 else 0

    def net_profit_after_tax(
        self,
        gross_profit: float,
        segment: str,
        holding_days: int = 0,
        income_slab_rate: float = 0.30,
    ) -> float:
        if segment in ("equity_delivery",):
            tax = self.equity_delivery_tax(gross_profit, holding_days, income_slab_rate)
        else:
            tax = self.fo_intraday_tax(gross_profit, income_slab_rate)
        return gross_profit - tax
