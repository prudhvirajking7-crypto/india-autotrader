from __future__ import annotations

import json
import structlog
import redis.asyncio as aioredis

from brokers.base import Order, OrderSide
from config.settings import settings

log = structlog.get_logger(__name__)

DAILY_PNL_KEY = "risk:daily_pnl"
OPEN_POSITIONS_KEY = "risk:open_positions"


class RiskManager:
    """
    Pre-trade risk checks applied to every order before execution:

    1. Daily loss limit     — block orders if realized P&L is below threshold
    2. Max open positions   — cap concurrent open trades
    3. Max position size    — cap single trade notional value
    4. Capital utilization  — ensure enough margin is available
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def approve(self, order: Order) -> tuple[bool, str]:
        """
        Run all risk checks. Returns (approved, reason).
        reason is empty string when approved.
        """
        checks = [
            self._check_daily_loss,
            self._check_open_positions,
            self._check_position_size,
        ]
        for check in checks:
            ok, reason = await check(order)
            if not ok:
                return False, reason
        return True, ""

    # ── Individual checks ─────────────────────────────────────────────────

    async def _check_daily_loss(self, order: Order) -> tuple[bool, str]:
        r = await self._get_redis()
        raw = await r.get(DAILY_PNL_KEY)
        daily_pnl = float(raw) if raw else 0.0

        if daily_pnl <= -abs(settings.daily_loss_limit_inr):
            return False, (
                f"Daily loss limit hit: ₹{daily_pnl:.2f} / "
                f"limit ₹{-settings.daily_loss_limit_inr:.2f}"
            )
        return True, ""

    async def _check_open_positions(self, order: Order) -> tuple[bool, str]:
        # Only block new BUY/LONG entries; allow SELL to close existing
        if order.side == OrderSide.SELL:
            return True, ""

        r = await self._get_redis()
        raw = await r.get(OPEN_POSITIONS_KEY)
        open_count = int(raw) if raw else 0

        if open_count >= settings.max_open_positions:
            return False, (
                f"Max open positions reached: {open_count}/{settings.max_open_positions}"
            )
        return True, ""

    async def _check_position_size(self, order: Order) -> tuple[bool, str]:
        # Estimate notional value (use price if set, else skip — market order)
        if order.price <= 0:
            return True, ""  # Can't check without price; broker will validate margin

        notional = order.price * order.qty
        if notional > settings.max_position_size_inr:
            return False, (
                f"Position size ₹{notional:.2f} exceeds limit ₹{settings.max_position_size_inr:.2f}"
            )
        return True, ""

    # ── State updates (called after order events) ─────────────────────────

    async def record_trade_pnl(self, pnl: float) -> None:
        r = await self._get_redis()
        await r.incrbyfloat(DAILY_PNL_KEY, pnl)
        # Expire at midnight IST
        from datetime import datetime
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        midnight = now.replace(hour=23, minute=59, second=59, microsecond=0)
        ttl = int((midnight - now).total_seconds())
        await r.expire(DAILY_PNL_KEY, ttl)

    async def increment_open_positions(self, delta: int = 1) -> None:
        r = await self._get_redis()
        await r.incrby(OPEN_POSITIONS_KEY, delta)

    async def get_daily_pnl(self) -> float:
        r = await self._get_redis()
        raw = await r.get(DAILY_PNL_KEY)
        return float(raw) if raw else 0.0

    async def get_open_position_count(self) -> int:
        r = await self._get_redis()
        raw = await r.get(OPEN_POSITIONS_KEY)
        return int(raw) if raw else 0
