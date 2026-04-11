from __future__ import annotations

import asyncio
import time

import redis.asyncio as aioredis
import structlog

from config.settings import settings
from signals.filters import MarketHoursFilter, SymbolNormalizer
from webhook.parser import TradingViewAlert

log = structlog.get_logger(__name__)

DEDUP_TTL_SECONDS = 60


class SignalProcessor:
    """
    Receives a parsed TradingViewAlert and runs it through:
    1. Deduplication (Redis TTL key)
    2. Market hours gate (09:15–15:30 IST, Mon–Fri, NSE holidays excluded)
    3. Symbol normalization (NSE:RELIANCE → RELIANCE)
    4. Risk management check
    5. Broker order placement
    6. Notification dispatch
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def process(self, alert: TradingViewAlert, received_at: float) -> None:
        latency_ms = int((time.time() - received_at) * 1000)
        log.info("signal.processing", symbol=alert.parsed_symbol, action=alert.action, latency_ms=latency_ms)

        # 1. Deduplication
        if await self._is_duplicate(alert):
            log.warning("signal.duplicate.dropped", symbol=alert.parsed_symbol, strategy=alert.strategy)
            return

        # 2. Market hours gate
        hours_filter = MarketHoursFilter()
        if not hours_filter.is_open():
            log.warning("signal.market_closed.dropped", symbol=alert.parsed_symbol)
            await self._notify_skipped(alert, reason="market_closed")
            return

        # 3. Normalize symbol
        normalizer = SymbolNormalizer()
        clean_symbol = normalizer.normalize(alert.parsed_symbol, alert.parsed_exchange)
        exchange = alert.parsed_exchange

        log.info("signal.validated", symbol=clean_symbol, exchange=exchange, action=alert.action)

        # 4. Intelligence scoring — score the signal before committing to an order
        from intelligence.scorer import SignalScorer
        scorer = SignalScorer()
        scored = await scorer.score(symbol=clean_symbol, action=alert.action, strategy=alert.strategy)

        log.info(
            "signal.scored",
            symbol=clean_symbol,
            score=scored.score,
            strength=scored.strength.value,
            skip=scored.skip,
        )

        if scored.skip:
            log.warning(
                "signal.low_confidence.dropped",
                symbol=clean_symbol,
                score=scored.score,
                rationale=scored.rationale,
            )
            await self._notify_skipped(
                alert,
                reason=f"Low confidence score {scored.score}/100 ({scored.strength.value}). {scored.rationale}",
            )
            return

        # 5. Risk check + order placement (async, non-blocking)
        await self._execute_order(alert, clean_symbol, exchange, scored)

    # ── Deduplication ─────────────────────────────────────────────────────

    async def _is_duplicate(self, alert: TradingViewAlert) -> bool:
        r = await self._get_redis()
        key = f"signal:dedup:{alert.strategy}:{alert.parsed_symbol}:{alert.action}"
        result = await r.set(key, "1", ex=DEDUP_TTL_SECONDS, nx=True)
        # nx=True means set only if key doesn't exist
        # result is None if key already existed (duplicate)
        return result is None

    # ── Order execution pipeline ──────────────────────────────────────────

    async def _execute_order(self, alert: TradingViewAlert, symbol: str, exchange: str, scored=None) -> None:
        from brokers.base import Order, OrderSide, OrderType, ProductType
        from brokers.factory import get_broker
        from risk.manager import RiskManager
        from utils.notifications import Notifier

        # Load strategy config
        strategy_cfg = self._load_strategy_config(alert.strategy)

        # Build preliminary order
        side = OrderSide(alert.action)
        product = ProductType(strategy_cfg.get("product", "MIS"))
        order_type_str = alert.order_type or strategy_cfg.get("order_type", "MARKET")
        if alert.price > 0:
            order_type_str = "LIMIT"

        # Scale quantity by intelligence confidence (e.g. WEAK_BUY = 50% size)
        size_multiplier = scored.size_multiplier if scored else 1.0
        final_qty = max(1, int(alert.qty * size_multiplier))

        order = Order(
            symbol=symbol,
            exchange=exchange,
            side=side,
            qty=final_qty,
            order_type=OrderType(order_type_str),
            product=product,
            price=alert.price,
            trigger_price=alert.trigger_price,
            tag=f"{alert.strategy[:10]}_{symbol[:8]}",
        )

        # Risk check
        risk_manager = RiskManager()
        approved, reason = await risk_manager.approve(order)
        if not approved:
            log.warning("signal.risk_rejected", symbol=symbol, reason=reason)
            await Notifier().send_risk_rejection(order, reason)
            return

        # Place order
        broker = get_broker()
        try:
            placed = await asyncio.get_event_loop().run_in_executor(None, broker.place_order, order)
            log.info(
                "signal.order_placed",
                symbol=symbol,
                side=side,
                qty=order.qty,
                broker_id=placed.broker_order_id,
                paper=settings.paper_trading,
            )
            await Notifier().send_order_placed(placed)
            await self._record_order(placed)
        except Exception as e:
            log.error("signal.order_failed", symbol=symbol, error=str(e))
            await Notifier().send_order_error(order, str(e))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_strategy_config(self, strategy_name: str) -> dict:
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "config" / "strategies.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        defaults = cfg.get("defaults", {})
        strategy = cfg.get("strategies", {}).get(strategy_name, {})
        return {**defaults, **strategy}

    async def _record_order(self, order) -> None:
        r = await self._get_redis()
        import json
        key = f"order:{order.order_id}"
        await r.setex(
            key,
            86400,  # keep for 24 hours
            json.dumps({
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": order.qty,
                "price": order.avg_price,
                "broker_id": order.broker_order_id,
                "status": order.status.value,
                "ts": int(time.time()),
            }),
        )

    async def _notify_skipped(self, alert: TradingViewAlert, reason: str) -> None:
        from utils.notifications import Notifier
        await Notifier().send_signal_skipped(alert.parsed_symbol, alert.action, reason)
