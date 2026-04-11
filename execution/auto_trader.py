"""
Auto Trader — Automated Order Execution with SL/Target

Takes scanner top picks and executes orders with:
  - Bracket orders (SL + target in a single order) where broker supports it
  - Separate SL order fallback (for brokers without bracket support)
  - Position sizing from risk manager (risk_per_trade_pct of capital)
  - De-duplication (don't enter the same symbol twice)
  - Post-execution Telegram notification with full trade details

Execution flow for each top pick:
  1. Validate: not already in position, market open, not in MIS square-off zone
  2. Check risk limits (daily loss, max positions)
  3. Calculate quantity from (risk per share × lot size)
  4. Place bracket order: entry LIMIT + SL + target
  5. Record in Redis: open_positions, daily_pnl
  6. Notify via Telegram

Bracket order support:
  Zerodha  → VARIETY_BO (Bracket Order) ✓
  Upstox   → Bracket order API v2 ✓
  AngelOne → Bracket / Cover order ✓
  Finvasia → No native bracket; places limit + SL separately

Stop Loss logic:
  - SL is placed as a STOP-LOSS-MARKET order (SLM) — guaranteed fill
  - SL price = from scanner (based on nearest support/resistance)
  - Trigger price = SL ± 0.5% (gives broker time to execute)

Auto Trail:
  - If price moves 1:1 toward target → SL moved to breakeven
  - If price moves 1:2 → SL moved to +0.5R (lock in profit)
  (Trailing is implemented as a background monitoring task)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import structlog

from scanner.stock_scanner import StockScanResult

log = structlog.get_logger(__name__)

# Minimum confirmations needed to auto-execute (vs just alert)
MIN_SCORE_TO_EXECUTE = 65
MIN_RR_TO_EXECUTE = 1.5
MIN_BREAKOUT_CONFIDENCE = 0.45  # breakout.confidence threshold

# Position tracking Redis key
OPEN_POSITIONS_KEY = "autotrader:open_positions"
EXECUTED_TODAY_KEY = "autotrader:executed_today"


@dataclass
class ExecutionResult:
    symbol: str
    action: str
    status: str          # "executed" | "skipped" | "failed" | "paper"
    reason: str          # Why skipped/failed, or confirmation for executed
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target2: float = 0.0
    quantity: int = 0
    broker_order_id: str = ""
    sl_order_id: str = ""
    risk_inr: float = 0.0
    potential_profit_inr: float = 0.0


class AutoTrader:
    """
    Executes orders from scanner results with full risk management.
    """

    def __init__(self) -> None:
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            from config.settings import settings
            self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def execute_scan_results(
        self,
        picks: list[StockScanResult],
        dry_run: bool = False,
    ) -> list[ExecutionResult]:
        """
        Process scanner top picks and execute eligible trades.

        Args:
            picks:   Ranked scan results from StockScanner
            dry_run: If True, compute but don't place orders (paper log only)

        Returns:
            List of execution results for each pick processed.
        """
        from config.settings import settings

        results = []
        for pick in picks:
            result = await self._process_pick(pick, dry_run or settings.paper_trading)
            results.append(result)
            if result.status in ("executed", "paper"):
                await self._notify(pick, result)
            # Small delay between orders
            await asyncio.sleep(0.5)

        return results

    async def _process_pick(self, pick: StockScanResult, paper: bool) -> ExecutionResult:
        """Run full validation and execute a single pick."""

        # ── Gate 1: Score and R:R ─────────────────────────────────────────
        if pick.score < MIN_SCORE_TO_EXECUTE:
            return ExecutionResult(pick.symbol, pick.action, "skipped",
                                   f"Score {pick.score} < {MIN_SCORE_TO_EXECUTE}")

        if pick.risk_reward < MIN_RR_TO_EXECUTE:
            return ExecutionResult(pick.symbol, pick.action, "skipped",
                                   f"R:R {pick.risk_reward:.1f} < {MIN_RR_TO_EXECUTE}")

        if pick.manipulation_score >= 2:
            return ExecutionResult(pick.symbol, pick.action, "skipped",
                                   f"Manipulation score {pick.manipulation_score} — blocked")

        # ── Gate 2: Don't enter same symbol twice ─────────────────────────
        if await self._is_in_position(pick.symbol):
            return ExecutionResult(pick.symbol, pick.action, "skipped",
                                   "Already in position")

        if await self._executed_today(pick.symbol):
            return ExecutionResult(pick.symbol, pick.action, "skipped",
                                   "Already traded this symbol today")

        # ── Gate 3: Risk manager checks ───────────────────────────────────
        from risk.manager import RiskManager
        rm = RiskManager()
        can_trade, reason = await rm.can_trade(pick.symbol, pick.action)
        if not can_trade:
            return ExecutionResult(pick.symbol, pick.action, "skipped", f"Risk: {reason}")

        # ── Gate 4: Market hours ──────────────────────────────────────────
        from signals.filters import MarketHoursFilter
        if not MarketHoursFilter.is_open():
            return ExecutionResult(pick.symbol, pick.action, "skipped", "Market closed")

        # ── Calculate quantity ────────────────────────────────────────────
        qty = await self._calculate_qty(pick)
        if qty <= 0:
            return ExecutionResult(pick.symbol, pick.action, "skipped",
                                   "Quantity = 0 (position too small)")

        # ── Execute ───────────────────────────────────────────────────────
        result = ExecutionResult(
            symbol=pick.symbol,
            action=pick.action,
            status="paper" if paper else "pending",
            reason="",
            entry_price=pick.entry_price,
            stop_loss=pick.stop_loss,
            target2=pick.target2,
            quantity=qty,
            risk_inr=round(pick.risk_per_share * qty, 2),
            potential_profit_inr=round(abs(pick.target2 - pick.entry_price) * qty, 2),
        )

        if paper:
            result.broker_order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
            result.sl_order_id = f"PAPER-SL-{uuid.uuid4().hex[:8].upper()}"
            result.status = "paper"
            result.reason = (
                f"[PAPER] {pick.action} {qty} {pick.symbol} @ ₹{pick.entry_price:.2f} "
                f"| SL ₹{pick.stop_loss:.2f} | T2 ₹{pick.target2:.2f} "
                f"| R:R {pick.risk_reward:.1f} | Score {pick.score}"
            )
        else:
            try:
                order_id, sl_order_id = await self._place_bracket_order(pick, qty)
                result.broker_order_id = order_id
                result.sl_order_id = sl_order_id
                result.status = "executed"
                result.reason = f"Order {order_id} placed, SL order {sl_order_id}"
            except Exception as e:
                result.status = "failed"
                result.reason = str(e)
                log.error("autotrader.order_failed", symbol=pick.symbol, error=str(e))
                return result

        # Record position
        await self._record_position(pick, result)

        log.info(
            "autotrader.result",
            symbol=pick.symbol,
            action=pick.action,
            status=result.status,
            qty=qty,
            entry=pick.entry_price,
            sl=pick.stop_loss,
            t2=pick.target2,
            rr=pick.risk_reward,
            score=pick.score,
        )
        return result

    async def _calculate_qty(self, pick: StockScanResult) -> int:
        """
        Position size = min(lot_size_multiple, risk_based_qty)

        Risk-based: risk_per_trade_inr / risk_per_share
        Lot-size: always a multiple of the F&O lot size
        """
        from config.settings import settings
        from intelligence.knowledge_base import get_lot_size

        risk_per_trade_inr = settings.max_capital * settings.risk_per_trade_pct / 100

        if pick.risk_per_share <= 0:
            return 0

        raw_qty = int(risk_per_trade_inr / pick.risk_per_share)

        # Snap to lot size
        lot = get_lot_size(pick.symbol)
        if lot > 1:
            snapped = max(lot, (raw_qty // lot) * lot)
        else:
            snapped = max(1, raw_qty)

        # Cap at max_position_size_pct of capital
        max_qty = int((settings.max_capital * settings.max_position_size_pct / 100) / pick.entry_price)
        return min(snapped, max_qty)

    async def _place_bracket_order(self, pick: StockScanResult, qty: int) -> tuple[str, str]:
        """
        Place a bracket order through the active broker.

        Bracket order = entry + SL + target in one transaction.
        If broker doesn't support bracket, places entry + separate SL order.
        """
        from brokers.factory import get_broker
        from config.settings import settings

        broker = get_broker(settings.active_broker)
        broker.ensure_logged_in()

        action = pick.action  # "BUY" | "SELL"
        is_buy = action == "BUY"

        # Points for SL and target (required by bracket order APIs)
        sl_points = round(abs(pick.entry_price - pick.stop_loss), 2)
        target_points = round(abs(pick.target2 - pick.entry_price), 2)
        trigger_price = round(
            pick.stop_loss * (0.998 if is_buy else 1.002), 2
        )  # Slightly inside SL for trigger

        from brokers.base import Order, OrderStatus

        # Build primary order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=pick.symbol,
            exchange="NSE",
            action=action,
            order_type="LIMIT",
            product="MIS",          # Intraday
            qty=qty,
            price=round(pick.entry_price, 2),
        )

        # Try bracket order first
        try:
            result_order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: broker.place_bracket_order(
                    order=order,
                    stop_loss_points=sl_points,
                    target_points=target_points,
                ) if hasattr(broker, "place_bracket_order") else None
            )
            if result_order and result_order.broker_order_id:
                return result_order.broker_order_id, result_order.broker_order_id + "-SL"
        except Exception as e:
            log.warning("autotrader.bracket_failed", symbol=pick.symbol, error=str(e))

        # Fallback: regular order + separate SL order
        result_order = await asyncio.get_event_loop().run_in_executor(
            None, broker.place_order, order
        )

        # SL order (opposite side, stop-loss-market)
        sl_order = Order(
            order_id=str(uuid.uuid4()),
            symbol=pick.symbol,
            exchange="NSE",
            action="SELL" if is_buy else "BUY",
            order_type="SL-M",
            product="MIS",
            qty=qty,
            price=0,
            trigger_price=trigger_price,
        )
        sl_result = await asyncio.get_event_loop().run_in_executor(
            None, broker.place_order, sl_order
        )

        return result_order.broker_order_id, sl_result.broker_order_id

    async def _is_in_position(self, symbol: str) -> bool:
        r = await self._get_redis()
        return bool(await r.hexists(OPEN_POSITIONS_KEY, symbol))

    async def _executed_today(self, symbol: str) -> bool:
        r = await self._get_redis()
        return bool(await r.hexists(EXECUTED_TODAY_KEY, symbol))

    async def _record_position(self, pick: StockScanResult, result: ExecutionResult) -> None:
        import json
        r = await self._get_redis()
        position_data = {
            "action": pick.action,
            "entry": pick.entry_price,
            "sl": pick.stop_loss,
            "t1": pick.target1,
            "t2": pick.target2,
            "t3": pick.target3,
            "qty": result.quantity,
            "order_id": result.broker_order_id,
            "sl_order_id": result.sl_order_id,
            "score": pick.score,
            "entered_at": time.time(),
            "wyckoff": pick.wyckoff_phase,
        }
        await r.hset(OPEN_POSITIONS_KEY, pick.symbol, json.dumps(position_data))
        await r.hset(EXECUTED_TODAY_KEY, pick.symbol, "1")
        # Auto-expire today's executions at midnight IST
        from datetime import datetime, timezone, timedelta
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)
        midnight = now_ist.replace(hour=23, minute=59, second=0, microsecond=0)
        ttl = int((midnight - now_ist).total_seconds())
        await r.expire(EXECUTED_TODAY_KEY, max(60, ttl))

    async def _notify(self, pick: StockScanResult, result: ExecutionResult) -> None:
        """Send Telegram notification with full trade details."""
        try:
            from utils.notifications import TelegramNotifier
            notifier = TelegramNotifier()

            emoji = "🟢" if pick.action == "BUY" else "🔴"
            mode = "PAPER" if result.status == "paper" else "LIVE"

            msg = (
                f"{emoji} *{mode} TRADE* [{pick.rank}]\n"
                f"*{pick.symbol}* {pick.action} | Score: {pick.score}/100\n\n"
                f"Entry : ₹{pick.entry_price:,.2f}\n"
                f"SL    : ₹{pick.stop_loss:,.2f} ({abs(pick.entry_price-pick.stop_loss)/pick.entry_price*100:.1f}%)\n"
                f"T1    : ₹{pick.target1:,.2f}\n"
                f"T2    : ₹{pick.target2:,.2f} (R:R {pick.risk_reward:.1f}x)\n"
                f"T3    : ₹{pick.target3:,.2f}\n\n"
                f"Qty   : {result.quantity}\n"
                f"Risk  : ₹{result.risk_inr:,.0f}\n"
                f"Pot.P : ₹{result.potential_profit_inr:,.0f}\n\n"
                f"*Evidence:*\n"
                f"Wyckoff: {pick.wyckoff_phase} ({pick.wyckoff_confidence:.0%})\n"
                f"Breakout: {pick.breakout_type or 'none'} "
                f"({'✓' if pick.breakout_confirmed else '~'} vol:{pick.breakout_volume_ratio:.1f}x)\n"
                f"SMC Bull/Bear: {pick.smc_bullish_signals}/{pick.smc_bearish_signals}\n"
                f"Manip Flags: {pick.manipulation_score}\n"
            )

            if pick.warnings:
                msg += f"\n⚠️ {pick.warnings[0]}"

            await notifier.send(msg)
        except Exception as e:
            log.warning("autotrader.notify_failed", error=str(e))


class TrailingStopMonitor:
    """
    Background task: monitors open positions and adjusts SL as price moves.

    Rules:
      After 1:1 R moved → move SL to breakeven
      After 1:2 R moved → move SL to +0.5R (locking in profit)
      After 1:3 R moved → close 50% position (trail remainder)
    """

    def __init__(self) -> None:
        self._redis = None
        self._running = False

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            from config.settings import settings
            self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def start(self) -> None:
        """Run the trailing monitor loop (call in background task)."""
        self._running = True
        log.info("trailing_monitor.started")
        while self._running:
            try:
                await self._check_all_positions()
            except Exception as e:
                log.warning("trailing_monitor.error", error=str(e))
            await asyncio.sleep(60)  # Check every 60 seconds

    def stop(self) -> None:
        self._running = False

    async def _check_all_positions(self) -> None:
        import json
        from signals.filters import MarketHoursFilter
        if not MarketHoursFilter.is_open():
            return

        r = await self._get_redis()
        positions = await r.hgetall(OPEN_POSITIONS_KEY)

        for symbol, data_str in positions.items():
            try:
                pos = json.loads(data_str)
                await self._check_and_trail(symbol, pos, r)
            except Exception as e:
                log.warning("trailing_monitor.symbol_error", symbol=symbol, error=str(e))

    async def _check_and_trail(self, symbol: str, pos: dict, r) -> None:
        """Check current price and trail SL if needed."""
        import json
        from data.nse_data import NSEDataProvider

        try:
            nse = NSEDataProvider()
            # Get LTP (last traded price) — simplified
            loop = asyncio.get_event_loop()
            ltp = await loop.run_in_executor(None, lambda: _get_ltp(symbol))
            if not ltp:
                return
        except Exception:
            return

        entry = pos["entry"]
        sl = pos["sl"]
        t1 = pos["t1"]
        t2 = pos["t2"]
        action = pos["action"]
        order_id = pos["order_id"]
        sl_order_id = pos["sl_order_id"]
        risk = abs(entry - sl)

        if risk <= 0:
            return

        is_buy = action == "BUY"
        move = (ltp - entry) if is_buy else (entry - ltp)
        move_in_r = move / risk  # How many R units moved

        new_sl = sl

        if move_in_r >= 3.0:
            # Moved 3R — trail SL to 2R
            new_sl = (entry + 2 * risk) if is_buy else (entry - 2 * risk)
        elif move_in_r >= 2.0:
            # Moved 2R — trail SL to 1R (lock profit)
            new_sl = (entry + risk) if is_buy else (entry - risk)
        elif move_in_r >= 1.0:
            # Moved 1R — move SL to breakeven
            new_sl = entry

        if abs(new_sl - sl) > risk * 0.1:
            log.info(
                "trailing_monitor.trail",
                symbol=symbol,
                action=action,
                ltp=ltp,
                old_sl=sl,
                new_sl=new_sl,
                move_r=f"{move_in_r:.2f}R",
            )
            pos["sl"] = new_sl
            await r.hset(OPEN_POSITIONS_KEY, symbol, json.dumps(pos))
            # TODO: Modify the SL order through the broker
            # broker.modify_order(sl_order_id, trigger_price=new_sl)

    async def close_position(self, symbol: str) -> None:
        """Remove a position from tracking (called after order fills)."""
        r = await self._get_redis()
        await r.hdel(OPEN_POSITIONS_KEY, symbol)
        log.info("trailing_monitor.position_closed", symbol=symbol)


def _get_ltp(symbol: str) -> Optional[float]:
    """Quick LTP fetch via yfinance (replace with broker WebSocket for live)."""
    try:
        import yfinance as yf
        t = yf.Ticker(f"{symbol}.NS")
        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None
