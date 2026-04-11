"""
India AutoTrader MCP Server

Exposes the trading system as MCP tools for Claude Code / Claude Desktop.
Allows natural-language queries like:
  - "What are my current open positions?"
  - "What is the daily P&L so far?"
  - "Run a backtest of EMA crossover on RELIANCE for the last 6 months"
  - "Place a paper trade: buy 10 RELIANCE at market"

Usage:
    python -m mcp.server
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import structlog

log = structlog.get_logger(__name__)


class AutoTraderMCPServer:
    """
    Lightweight MCP tool provider wrapping the autotrader core.
    Each tool is a Python async function with a clear description.
    """

    # ── Tool: get_positions ────────────────────────────────────────────────

    async def get_positions(self) -> str:
        """Return all currently open positions from the active broker."""
        from brokers.factory import get_broker
        broker = get_broker()
        positions = broker.get_positions()
        if not positions:
            return "No open positions."
        rows = [f"- {p.symbol} ({p.exchange}): qty={p.qty}, avg=₹{p.avg_price:.2f}, LTP=₹{p.ltp:.2f}, P&L=₹{p.pnl:+.2f}"
                for p in positions]
        return "\n".join(rows)

    # ── Tool: get_daily_pnl ────────────────────────────────────────────────

    async def get_daily_pnl(self) -> str:
        """Return today's realized P&L and open position count."""
        from risk.manager import RiskManager
        rm = RiskManager()
        pnl = await rm.get_daily_pnl()
        open_count = await rm.get_open_position_count()
        from config.settings import settings
        limit = settings.daily_loss_limit_inr
        remaining = limit + pnl  # pnl is negative for loss
        return (
            f"Daily P&L: ₹{pnl:+.2f}\n"
            f"Open positions: {open_count}/{settings.max_open_positions}\n"
            f"Loss limit remaining: ₹{remaining:.2f} of ₹{limit:.2f}"
        )

    # ── Tool: run_backtest ─────────────────────────────────────────────────

    async def run_backtest(
        self,
        symbol: str,
        strategy: str = "ema_cross",
        days: int = 180,
        segment: str = "equity_intraday",
    ) -> str:
        """
        Run a backtest and return key metrics.

        Args:
            symbol: NSE symbol e.g. RELIANCE
            strategy: ema_cross | rsi_mean_reversion | supertrend | macd
            days: lookback period in days
            segment: equity_delivery | equity_intraday | fo_futures | fo_options
        """
        from backtest.runner import BacktestRunner
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        runner = BacktestRunner()
        result = await asyncio.get_event_loop().run_in_executor(
            None, runner.run, symbol, strategy, from_date, to_date, segment
        )
        return (
            f"Backtest: {result.symbol} — {result.strategy}\n"
            f"Period: {from_date} to {to_date}\n"
            f"Total Return: {result.total_return_pct:.1f}%\n"
            f"CAGR: {result.cagr_pct:.1f}%\n"
            f"Sharpe: {result.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {result.max_drawdown_pct:.1f}%\n"
            f"Win Rate: {result.win_rate_pct:.1f}%\n"
            f"Total Trades: {result.total_trades}\n"
            f"Report saved: {result.report_path}"
        )

    # ── Tool: place_paper_trade ────────────────────────────────────────────

    async def place_paper_trade(
        self,
        symbol: str,
        action: str,
        qty: int = 1,
        price: float = 0.0,
    ) -> str:
        """
        Place a paper trade (simulation only, no real money).

        Args:
            symbol: NSE symbol
            action: BUY or SELL
            qty: number of shares
            price: limit price, 0 for market order
        """
        from brokers.base import Order, OrderSide, OrderType, ProductType
        from brokers.factory import get_broker

        order = Order(
            symbol=symbol.upper(),
            exchange="NSE",
            side=OrderSide(action.upper()),
            qty=qty,
            order_type=OrderType.LIMIT if price > 0 else OrderType.MARKET,
            product=ProductType.MIS,
            price=price,
            tag="MCP_PAPER",
        )

        # Force paper mode for MCP
        from config.settings import settings
        original = settings.paper_trading
        broker = get_broker()
        broker.paper_trading = True

        placed = broker.place_order(order)
        broker.paper_trading = original

        return (
            f"Paper trade placed!\n"
            f"Symbol: {placed.symbol}\n"
            f"Side: {placed.side.value}\n"
            f"Qty: {placed.filled_qty}\n"
            f"Avg Price: ₹{placed.avg_price:.2f}\n"
            f"Status: {placed.status.value}\n"
            f"Order ID: {placed.broker_order_id}"
        )

    # ── Tool: get_market_status ────────────────────────────────────────────

    async def get_market_status(self) -> str:
        """Return whether NSE market is currently open."""
        from signals.filters import MarketHoursFilter
        mhf = MarketHoursFilter()
        if mhf.is_open():
            return "NSE market is currently OPEN (09:15–15:30 IST)"
        secs = mhf.time_to_open()
        hrs, rem = divmod(secs, 3600)
        mins = rem // 60
        return f"NSE market is CLOSED. Opens in {hrs}h {mins}m"

    # ── Tool: score_signal ────────────────────────────────────────────────

    async def score_signal(self, symbol: str, action: str) -> str:
        """
        Get full multi-factor intelligence score for a BUY or SELL signal.

        Shows: technical alignment, news sentiment, global market context,
        momentum, and whether to trade or skip.
        """
        from intelligence.scorer import SignalScorer
        scorer = SignalScorer()
        result = await scorer.score(symbol=symbol.upper(), action=action.upper())
        bd = result.breakdown
        return (
            f"Signal Score: {result.score}/100 → {result.strength.value}\n"
            f"{'✅ TRADE' if not result.skip else '❌ SKIP (score too low)'}\n"
            f"Size multiplier: {result.size_multiplier:.0%}\n\n"
            f"Technical  ({bd.technical_pts:.0f}/40): "
            f"EMA {'✓' if bd.ema_aligned else '✗'} | "
            f"RSI {bd.rsi_value:.0f} {'✓' if bd.rsi_favorable else '✗'} | "
            f"MACD {'✓' if bd.macd_aligned else '✗'} | "
            f"Supertrend {'✓' if bd.supertrend_aligned else '✗'} | "
            f"Volume {'↑' if bd.volume_above_avg else '='}\n"
            f"News       ({bd.news_pts:.0f}/25): score {bd.news_score:+.2f}\n"
            f"Global     ({bd.context_pts:.0f}/20): context {bd.context_score:+.2f}, VIX {bd.vix_level:.1f}\n"
            f"Momentum   ({bd.momentum_pts:.0f}/15): 5d change {bd.momentum_5d:+.1f}%\n"
            + (f"\n⚠️ Warnings:\n" + "\n".join(f"  - {w}" for w in bd.warnings) if bd.warnings else "")
            + f"\n\nRationale: {result.rationale}"
        )

    # ── Tool: get_market_context ───────────────────────────────────────────

    async def get_market_context(self) -> str:
        """Get current global market context affecting Indian markets."""
        from intelligence.market_context import MarketContextTracker
        tracker = MarketContextTracker()
        ctx = await tracker.get_context()
        return tracker.summarize(ctx)

    # ── Tool: get_news_sentiment ───────────────────────────────────────────

    async def get_news_sentiment(self, symbol: str) -> str:
        """Get news sentiment for a given NSE symbol."""
        from intelligence.news import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer()
        s = await analyzer.get_symbol_sentiment(symbol.upper())
        label = (
            "STRONGLY BULLISH" if s.score > 0.5 else
            "BULLISH" if s.score > 0.15 else
            "STRONGLY BEARISH" if s.score < -0.5 else
            "BEARISH" if s.score < -0.15 else "NEUTRAL"
        )
        return (
            f"{symbol.upper()} News Sentiment: {label} ({s.score:+.2f})\n"
            f"Articles analyzed: {s.article_count} "
            f"({s.bullish_count} bullish, {s.bearish_count} bearish)\n\n"
            f"Top headlines:\n" + "\n".join(f"  • {h}" for h in s.top_headlines[:5])
        )

    # ── Tool: get_historical_data ──────────────────────────────────────────

    async def get_historical_data(self, symbol: str, days: int = 30) -> str:
        """Fetch recent OHLCV data for a symbol."""
        from data.nse_data import NSEDataProvider
        from datetime import datetime, timedelta

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        nse = NSEDataProvider()
        df = nse.get_equity_ohlcv(symbol, from_date, to_date)

        if df.empty:
            return f"No data found for {symbol}"

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change = latest["close"] - prev["close"]
        change_pct = (change / prev["close"]) * 100

        return (
            f"{symbol} — Last {days} days\n"
            f"Latest: ₹{latest['close']:.2f} ({change_pct:+.2f}%)\n"
            f"High: ₹{df['high'].max():.2f} | Low: ₹{df['low'].min():.2f}\n"
            f"Avg Volume: {df['volume'].mean():,.0f}\n"
            f"Rows: {len(df)}"
        )


# ── MCP entry point ────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict = {
    "get_positions": {
        "description": "Get all currently open positions from the active broker",
        "parameters": {},
    },
    "get_daily_pnl": {
        "description": "Get today's realized P&L and risk metrics",
        "parameters": {},
    },
    "get_market_status": {
        "description": "Check if NSE market is currently open",
        "parameters": {},
    },
    "run_backtest": {
        "description": "Run a backtest of a trading strategy on an NSE symbol",
        "parameters": {
            "symbol": {"type": "string", "description": "NSE symbol e.g. RELIANCE"},
            "strategy": {"type": "string", "enum": ["ema_cross", "rsi_mean_reversion", "supertrend", "macd"]},
            "days": {"type": "integer", "default": 180},
            "segment": {"type": "string", "default": "equity_intraday"},
        },
    },
    "place_paper_trade": {
        "description": "Simulate a paper trade (no real money)",
        "parameters": {
            "symbol": {"type": "string"},
            "action": {"type": "string", "enum": ["BUY", "SELL"]},
            "qty": {"type": "integer", "default": 1},
            "price": {"type": "number", "default": 0},
        },
    },
    "get_historical_data": {
        "description": "Fetch recent OHLCV price data for an NSE symbol",
        "parameters": {
            "symbol": {"type": "string"},
            "days": {"type": "integer", "default": 30},
        },
    },
    "score_signal": {
        "description": "Get multi-factor intelligence score (0–100) for a BUY or SELL signal. Shows technical, news, global context, and momentum breakdown.",
        "parameters": {
            "symbol": {"type": "string", "description": "NSE symbol e.g. RELIANCE"},
            "action": {"type": "string", "enum": ["BUY", "SELL"]},
        },
    },
    "get_market_context": {
        "description": "Get current global market context: Gift NIFTY, US markets, India VIX, FII/DII flow, crude oil, USD/INR",
        "parameters": {},
    },
    "get_news_sentiment": {
        "description": "Get news sentiment score for an NSE symbol from Google News and financial RSS feeds",
        "parameters": {
            "symbol": {"type": "string", "description": "NSE symbol e.g. RELIANCE"},
        },
    },
}


async def main() -> None:
    """Minimal stdio MCP server loop."""
    import sys
    server = AutoTraderMCPServer()
    log.info("mcp_server.started")

    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            tool = req.get("tool")
            params = req.get("params", {})

            if tool not in TOOL_REGISTRY:
                result = {"error": f"Unknown tool: {tool}"}
            else:
                fn = getattr(server, tool)
                result = {"result": await fn(**params)}

            print(json.dumps(result), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
