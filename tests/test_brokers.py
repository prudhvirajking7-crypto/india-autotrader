"""Tests for broker abstraction layer (paper mode)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from brokers.base import (
    BrokerBase, Order, OrderSide, OrderStatus, OrderType, Position, ProductType
)


# ── Minimal concrete broker for testing the base class ───────────────────────

class MockBroker(BrokerBase):
    def login(self):
        self._logged_in = True

    def _live_place_order(self, order):
        order.broker_order_id = "LIVE-001"
        order.status = OrderStatus.OPEN
        return order

    def _live_cancel_order(self, order_id):
        return True

    def get_positions(self):
        return [Position("RELIANCE", "NSE", ProductType.MIS, 10, 2500.0, 2600.0, 1000.0)]

    def get_margins(self):
        return {"available_cash": 50000.0, "utilised": 10000.0, "net": 40000.0}

    def get_ltp(self, symbol, exchange):
        return 2600.0

    def get_instrument_token(self, symbol, exchange):
        return "738561"


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPaperTrading:
    def _make_order(self) -> Order:
        return Order(
            symbol="RELIANCE",
            exchange="NSE",
            side=OrderSide.BUY,
            qty=10,
            order_type=OrderType.MARKET,
            product=ProductType.MIS,
        )

    def test_paper_order_completes_immediately(self):
        broker = MockBroker(paper_trading=True)
        order = self._make_order()
        result = broker.place_order(order)
        assert result.status == OrderStatus.COMPLETE
        assert result.filled_qty == 10
        assert result.broker_order_id.startswith("PAPER-")

    def test_paper_order_uses_ltp_for_market(self):
        broker = MockBroker(paper_trading=True)
        order = self._make_order()
        order.price = 0  # market order
        result = broker.place_order(order)
        assert result.avg_price == 2600.0  # from get_ltp

    def test_paper_cancel_always_succeeds(self):
        broker = MockBroker(paper_trading=True)
        assert broker.cancel_order("any-id") is True

    def test_live_order_calls_implementation(self):
        broker = MockBroker(paper_trading=False)
        order = self._make_order()
        result = broker.place_order(order)
        assert result.broker_order_id == "LIVE-001"
        assert result.status == OrderStatus.OPEN

    def test_ensure_logged_in_auto_logins(self):
        broker = MockBroker(paper_trading=True)
        assert broker._logged_in is False
        broker.ensure_logged_in()
        assert broker._logged_in is True

    def test_health_check_ok(self):
        broker = MockBroker(paper_trading=True)
        result = broker.health_check()
        assert result["status"] == "ok"
        assert result["paper"] is True
