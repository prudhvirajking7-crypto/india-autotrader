from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TradingViewAlert(BaseModel):
    """
    Schema for TradingView webhook alert payload.

    Example Pine Script alert message (JSON format):
    {
        "symbol": "NSE:RELIANCE",
        "action": "BUY",
        "qty": 1,
        "price": 0,
        "strategy": "ema_cross",
        "timeframe": "5m",
        "comment": "EMA crossover signal"
    }
    """

    symbol: str = Field(..., description="Exchange:Symbol e.g. NSE:RELIANCE or just RELIANCE")
    action: str = Field(..., description="BUY or SELL")
    qty: int = Field(default=1, ge=1)
    price: float = Field(default=0.0, ge=0.0, description="0 means MARKET order")
    trigger_price: float = Field(default=0.0, ge=0.0)
    strategy: str = Field(default="default", description="Strategy name, maps to strategies.yaml")
    timeframe: str = Field(default="", description="Chart timeframe e.g. 5m, 15m, 1h")
    comment: Optional[str] = None
    exchange: Optional[str] = None    # Override exchange if provided separately
    order_type: Optional[str] = None  # Override order type

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in {"BUY", "SELL", "BUY_LONG", "SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT"}:
            raise ValueError(f"Invalid action: {v}. Must be BUY or SELL")
        # Normalize aliases
        if v in {"BUY_LONG"}:
            return "BUY"
        if v in {"SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT"}:
            return "SELL"
        return v

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @property
    def parsed_exchange(self) -> str:
        """Extract exchange from 'NSE:RELIANCE' format, default NSE."""
        if self.exchange:
            return self.exchange.upper()
        if ":" in self.symbol:
            return self.symbol.split(":")[0].upper()
        return "NSE"

    @property
    def parsed_symbol(self) -> str:
        """Extract clean symbol from 'NSE:RELIANCE' format."""
        if ":" in self.symbol:
            return self.symbol.split(":")[1].upper()
        return self.symbol.upper()
