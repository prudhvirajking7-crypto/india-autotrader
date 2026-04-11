from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_MARKET = "SL-M"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ProductType(str, Enum):
    MIS = "MIS"    # Intraday
    CNC = "CNC"    # Delivery equity
    NRML = "NRML"  # F&O overnight


@dataclass
class Order:
    symbol: str
    exchange: str
    side: OrderSide
    qty: int
    order_type: OrderType
    product: ProductType
    price: float = 0.0
    trigger_price: float = 0.0
    tag: str = ""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_price: float = 0.0
    broker_order_id: Optional[str] = None
    message: str = ""


@dataclass
class Position:
    symbol: str
    exchange: str
    product: ProductType
    qty: int            # positive = long, negative = short
    avg_price: float
    ltp: float = 0.0
    pnl: float = 0.0


class BrokerBase(ABC):
    """
    Abstract base for all Indian broker integrations.
    Subclasses implement the actual SDK calls; paper mode is handled here.
    """

    def __init__(self, paper_trading: bool = True) -> None:
        self.paper_trading = paper_trading
        self._logged_in = False

    # ── Auth ─────────────────────────────────────────────────────────────

    @abstractmethod
    def login(self) -> None:
        """Authenticate with the broker. Must set self._logged_in = True on success."""

    def ensure_logged_in(self) -> None:
        if not self._logged_in:
            self.login()

    # ── Orders ───────────────────────────────────────────────────────────

    def place_order(self, order: Order) -> Order:
        self.ensure_logged_in()
        if self.paper_trading:
            return self._paper_place_order(order)
        return self._live_place_order(order)

    def cancel_order(self, order_id: str) -> bool:
        self.ensure_logged_in()
        if self.paper_trading:
            return True
        return self._live_cancel_order(order_id)

    @abstractmethod
    def _live_place_order(self, order: Order) -> Order:
        """Place a real order via the broker SDK."""

    @abstractmethod
    def _live_cancel_order(self, order_id: str) -> bool:
        """Cancel a real order via the broker SDK."""

    def _paper_place_order(self, order: Order) -> Order:
        """Simulate order fill at current price."""
        order.status = OrderStatus.COMPLETE
        order.filled_qty = order.qty
        order.avg_price = order.price if order.price > 0 else self._get_ltp(order.symbol, order.exchange)
        order.broker_order_id = f"PAPER-{order.order_id[:8]}"
        return order

    # ── Portfolio ────────────────────────────────────────────────────────

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all open positions."""

    @abstractmethod
    def get_margins(self) -> dict:
        """Return available margin/cash details."""

    @abstractmethod
    def get_ltp(self, symbol: str, exchange: str) -> float:
        """Return last traded price for a symbol."""

    def _get_ltp(self, symbol: str, exchange: str) -> float:
        try:
            return self.get_ltp(symbol, exchange)
        except Exception:
            return 0.0

    # ── Instruments ──────────────────────────────────────────────────────

    @abstractmethod
    def get_instrument_token(self, symbol: str, exchange: str) -> str:
        """Resolve symbol to broker-specific instrument token."""

    # ── Health ───────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        try:
            self.ensure_logged_in()
            margins = self.get_margins()
            return {"status": "ok", "paper": self.paper_trading, "margins": margins}
        except Exception as e:
            return {"status": "error", "error": str(e)}
