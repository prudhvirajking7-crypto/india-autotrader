"""Tests for risk management engine."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from brokers.base import Order, OrderSide, OrderType, ProductType
from risk.manager import RiskManager
from risk.position_sizer import PositionSizer


def _make_order(side=OrderSide.BUY, qty=10, price=1000.0) -> Order:
    return Order(
        symbol="RELIANCE",
        exchange="NSE",
        side=side,
        qty=qty,
        order_type=OrderType.LIMIT,
        product=ProductType.MIS,
        price=price,
    )


# ── PositionSizer tests ───────────────────────────────────────────────────────

class TestPositionSizer:
    def test_basic_calculation(self, monkeypatch):
        # capital=100000, risk=1% = ₹1000 risk amount
        # stop=0.5% of 1000 = ₹5/share → qty = 1000/5 = 200
        # max_position_size_pct must be high enough to not cap at 200
        monkeypatch.setattr("risk.position_sizer.settings.max_capital", 100_000.0)
        monkeypatch.setattr("risk.position_sizer.settings.risk_per_trade_pct", 1.0)
        monkeypatch.setattr("risk.position_sizer.settings.max_position_size_pct", 500.0)
        sizer = PositionSizer()
        qty = sizer.calculate_qty(entry_price=1000.0, stop_loss_pct=0.5)
        assert qty == 200

    def test_capped_by_max_position_size(self, monkeypatch):
        # max_position_size_pct = 20% of 100000 = 20000 → max_qty = 20000/100 = 200
        # risk-based would give 500, but cap applies
        monkeypatch.setattr("risk.position_sizer.settings.max_capital", 100_000.0)
        monkeypatch.setattr("risk.position_sizer.settings.max_position_size_pct", 20.0)
        monkeypatch.setattr("risk.position_sizer.settings.risk_per_trade_pct", 5.0)
        sizer = PositionSizer()
        qty = sizer.calculate_qty(entry_price=100.0, stop_loss_pct=0.5)
        assert qty <= 2000  # max_position = 20% of 100k = 20000 / 100

    def test_lot_size_rounding(self, monkeypatch):
        monkeypatch.setattr("risk.position_sizer.settings.max_capital", 100_000.0)
        monkeypatch.setattr("risk.position_sizer.settings.risk_per_trade_pct", 1.0)
        monkeypatch.setattr("risk.position_sizer.settings.max_position_size_pct", 500.0)
        sizer = PositionSizer()
        qty = sizer.calculate_qty(entry_price=1000.0, stop_loss_pct=0.5, lot_size=75)
        assert qty % 75 == 0

    def test_stop_loss_price_long(self):
        sizer = PositionSizer()
        sl = sizer.calculate_stop_loss_price(entry_price=1000.0, stop_loss_pct=1.0, is_long=True)
        assert sl == 990.0

    def test_target_price_short(self):
        sizer = PositionSizer()
        tp = sizer.calculate_target_price(entry_price=1000.0, target_pct=2.0, is_long=False)
        assert tp == 980.0


# ── RiskManager tests ─────────────────────────────────────────────────────────

class TestRiskManager:
    @pytest.mark.asyncio
    async def test_approve_normal_order(self, monkeypatch):
        rm = RiskManager()
        # daily_loss_limit_inr = max_capital * daily_loss_limit_pct / 100
        # 100000 * 3.0 / 100 = 3000.0
        monkeypatch.setattr("risk.manager.settings.max_capital", 100_000.0)
        monkeypatch.setattr("risk.manager.settings.daily_loss_limit_pct", 3.0)
        monkeypatch.setattr("risk.manager.settings.max_open_positions", 5)
        monkeypatch.setattr("risk.manager.settings.max_position_size_pct", 20.0)

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)  # no existing P&L, no positions
        mock_redis.set = AsyncMock(return_value=True)
        rm._redis = mock_redis

        order = _make_order(price=500.0, qty=10)  # notional = ₹5000
        ok, reason = await rm.approve(order)
        assert ok is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_block_on_daily_loss_limit(self, monkeypatch):
        rm = RiskManager()
        # daily_loss_limit_inr = 100000 * 3.0 / 100 = 3000.0
        monkeypatch.setattr("risk.manager.settings.max_capital", 100_000.0)
        monkeypatch.setattr("risk.manager.settings.daily_loss_limit_pct", 3.0)

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="-3500")  # already at loss
        rm._redis = mock_redis

        order = _make_order()
        ok, reason = await rm.approve(order)
        assert ok is False
        assert "Daily loss limit" in reason

    @pytest.mark.asyncio
    async def test_block_on_max_positions(self, monkeypatch):
        rm = RiskManager()
        monkeypatch.setattr("risk.manager.settings.max_open_positions", 3)
        # Set daily loss limit very high so it doesn't block
        monkeypatch.setattr("risk.manager.settings.max_capital", 100_000.0)
        monkeypatch.setattr("risk.manager.settings.daily_loss_limit_pct", 99.999)

        mock_redis = AsyncMock()
        # P&L check passes, position count check fails
        mock_redis.get = AsyncMock(side_effect=[None, "3"])  # pnl=None, positions=3
        rm._redis = mock_redis

        order = _make_order(side=OrderSide.BUY)
        ok, reason = await rm.approve(order)
        assert ok is False
        assert "Max open positions" in reason

    @pytest.mark.asyncio
    async def test_sell_bypasses_position_count(self, monkeypatch):
        """SELL orders should not be blocked by max open positions."""
        rm = RiskManager()
        monkeypatch.setattr("risk.manager.settings.max_open_positions", 0)  # would block BUY
        monkeypatch.setattr("risk.manager.settings.max_capital", 100_000.0)
        monkeypatch.setattr("risk.manager.settings.daily_loss_limit_pct", 99.999)
        monkeypatch.setattr("risk.manager.settings.max_position_size_pct", 999.999)

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        rm._redis = mock_redis

        order = _make_order(side=OrderSide.SELL)
        ok, _ = await rm.approve(order)
        assert ok is True
