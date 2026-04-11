from __future__ import annotations

import math
from config.settings import settings


class PositionSizer:
    """
    Calculates recommended position size using risk-based sizing.

    Formula:
        qty = floor( (capital * risk_pct/100) / (entry_price * stop_loss_pct/100) )

    Respects MAX_POSITION_SIZE_PCT cap.
    """

    def calculate_qty(
        self,
        entry_price: float,
        stop_loss_pct: float,
        risk_pct: float | None = None,
        lot_size: int = 1,
    ) -> int:
        if entry_price <= 0 or stop_loss_pct <= 0:
            return 1

        effective_risk_pct = risk_pct if risk_pct is not None else settings.risk_per_trade_pct
        risk_amount = settings.max_capital * effective_risk_pct / 100
        risk_per_share = entry_price * stop_loss_pct / 100

        raw_qty = math.floor(risk_amount / risk_per_share)

        # Cap at max position size
        max_qty = math.floor(settings.max_position_size_inr / entry_price)
        qty = min(raw_qty, max_qty)

        # Round down to lot size
        if lot_size > 1:
            qty = (qty // lot_size) * lot_size

        return max(qty, lot_size)

    def calculate_stop_loss_price(
        self,
        entry_price: float,
        stop_loss_pct: float,
        is_long: bool = True,
    ) -> float:
        if is_long:
            return round(entry_price * (1 - stop_loss_pct / 100), 2)
        return round(entry_price * (1 + stop_loss_pct / 100), 2)

    def calculate_target_price(
        self,
        entry_price: float,
        target_pct: float,
        is_long: bool = True,
    ) -> float:
        if is_long:
            return round(entry_price * (1 + target_pct / 100), 2)
        return round(entry_price * (1 - target_pct / 100), 2)
