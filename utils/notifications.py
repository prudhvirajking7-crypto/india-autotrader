from __future__ import annotations

import httpx
import structlog
from config.settings import settings

log = structlog.get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    """
    Sends Telegram notifications for order events.
    Gracefully silenced if TELEGRAM_BOT_TOKEN is not configured.
    """

    def __init__(self) -> None:
        self._enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)

    async def _send(self, text: str) -> None:
        if not self._enabled:
            return
        url = TELEGRAM_API.format(token=settings.telegram_bot_token)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(url, json={
                    "chat_id": settings.telegram_chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                })
        except Exception as e:
            log.warning("telegram.send_failed", error=str(e))

    async def send_order_placed(self, order) -> None:
        mode = "[PAPER]" if settings.paper_trading else "[LIVE]"
        emoji = "BUY" if order.side.value == "BUY" else "SELL"
        icon = "🟢" if emoji == "BUY" else "🔴"
        await self._send(
            f"{icon} <b>{mode} {emoji}</b>\n"
            f"Symbol: <code>{order.symbol}</code>\n"
            f"Qty: {order.qty} @ ₹{order.avg_price:.2f}\n"
            f"Broker ID: {order.broker_order_id}\n"
            f"Status: {order.status.value}"
        )

    async def send_order_error(self, order, error: str) -> None:
        await self._send(
            f"❌ <b>ORDER FAILED</b>\n"
            f"Symbol: <code>{order.symbol}</code>\n"
            f"Side: {order.side.value}\n"
            f"Error: {error}"
        )

    async def send_risk_rejection(self, order, reason: str) -> None:
        await self._send(
            f"⚠️ <b>RISK BLOCK</b>\n"
            f"Symbol: <code>{order.symbol}</code>\n"
            f"Side: {order.side.value}\n"
            f"Reason: {reason}"
        )

    async def send_signal_skipped(self, symbol: str, action: str, reason: str) -> None:
        await self._send(
            f"⏭️ <b>Signal Skipped</b>\n"
            f"Symbol: <code>{symbol}</code> {action}\n"
            f"Reason: {reason}"
        )

    async def send_daily_summary(self, pnl: float, trades: int, open_positions: int) -> None:
        icon = "📈" if pnl >= 0 else "📉"
        await self._send(
            f"{icon} <b>Daily Summary</b>\n"
            f"P&L: ₹{pnl:+.2f}\n"
            f"Trades: {trades}\n"
            f"Open Positions: {open_positions}"
        )

    async def send_options_signals(self, signals: list, market_ctx: dict) -> None:
        """Send AI-generated options signals via Telegram."""
        from intelligence.ai_analyst import OptionsSignal
        vix = market_ctx.get("india_vix", 0)
        bias = market_ctx.get("bias", "neutral").upper()

        header = (
            f"🤖 <b>AI Options Analysis</b>\n"
            f"Market: <b>{bias}</b> | VIX: <b>{vix:.1f}</b>\n"
            f"FII: ₹{market_ctx.get('fii_net_crore', 0):+.0f}Cr\n"
            f"{'─' * 30}\n"
        )

        body = ""
        for s in signals:
            action_icon = {
                "BUY_CE": "📈🟢", "BUY_PE": "📉🔴",
                "SELL_CE": "⬇️🟢", "SELL_PE": "⬆️🔴",
            }.get(s.action, "⚪")

            strike_str = f" ₹{s.strike:.0f}" if s.strike else ""
            prem_str = f" ~₹{s.premium_est:.0f}" if s.premium_est else ""

            body += (
                f"{action_icon} <b>{s.symbol} {s.action}</b>{strike_str} ({s.expiry}){prem_str}\n"
                f"  Target: +{s.target_pct:.0f}% | SL: -{s.sl_pct:.0f}% | Conf: {s.confidence:.0%}\n"
                f"  📝 {s.rationale[:150]}\n"
            )
            if s.risk_factors:
                body += f"  ⚠️ Risks: {', '.join(s.risk_factors[:2])}\n"
            body += "\n"

        await self._send(header + body)
