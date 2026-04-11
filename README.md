# India AutoTrader

Automated trading system for NSE/BSE — TradingView webhook alerts → risk checks → live broker order execution.

---

## Architecture

```
TradingView Pine Script Alert (JSON webhook)
            │
            ▼  POST /webhook/{secret}
    ┌──────────────────┐
    │  FastAPI Server  │  ← Responds in <200ms, processes async
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Signal Processor │  ← Dedup (60s Redis TTL) + Market hours gate
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Risk Manager    │  ← Daily P&L limit + Max positions + Position size cap
    └────────┬─────────┘
             │
    ┌────────▼─────────┐     ┌────────────────────────────┐
    │  Broker Router   │────▶│ Zerodha / Upstox / AngelOne│
    └──────────────────┘     │ Finvasia (zero-brokerage)  │
                             └────────────────────────────┘
             │
    ┌────────▼─────────┐
    │  Telegram Alert  │  ← Every order event
    └──────────────────┘
```

---

## Quick Start

### 1. Clone & configure
```bash
git clone <your-fork>
cd india-autotrader
cp .env.example .env
# Edit .env: set WEBHOOK_SECRET, broker credentials, TELEGRAM_BOT_TOKEN
```

### 2. Start with Docker (recommended)
```bash
docker compose up -d
# App: http://localhost:8000
# Docs: http://localhost:8000/docs
# Redis UI (dev): docker compose --profile dev up
```

### 3. Start locally
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
python main.py
```

---

## TradingView Webhook Setup

In TradingView **Pine Script**, trigger an alert with this JSON payload:

```json
{
  "symbol": "{{exchange}}:{{ticker}}",
  "action": "{{strategy.order.action}}",
  "qty": 1,
  "price": {{close}},
  "strategy": "ema_cross",
  "timeframe": "{{interval}}",
  "comment": "{{strategy.order.comment}}"
}
```

Set the **Webhook URL** to:
```
https://your-server.com/webhook/YOUR_WEBHOOK_SECRET
```

Test with the dry-run endpoint (no orders placed):
```
POST /webhook/YOUR_WEBHOOK_SECRET/test
```

---

## Supported Brokers

| Broker    | Free API? | Features                           |
|-----------|-----------|-----------------------------------|
| Zerodha   | ₹2000/mo  | Most mature SDK, official MCP     |
| Upstox    | Free      | v2 API, sandbox environment       |
| AngelOne  | Free      | TOTP auto-login, options Greeks   |
| Finvasia  | Free      | **Zero brokerage** — best for HFT |

Set `ACTIVE_BROKER` in `.env` to switch brokers.

---

## Paper Trading

All orders are simulated by default (`PAPER_TRADING=true`). Set to `false` only after:
1. Testing with paper mode for at least 1 week
2. Verifying all risk limits are correctly configured
3. Confirming broker API credentials are valid

---

## Backtesting

```bash
# Single symbol
python scripts/backtest.py --symbol RELIANCE --strategy ema_cross --days 180

# Scan NIFTY 50 and rank by Sharpe
python scripts/backtest.py scan --strategy supertrend --days 90
```

**Built-in strategies:**
- `ema_cross` — EMA 9/21 crossover
- `rsi_mean_reversion` — RSI 14 oversold/overbought
- `supertrend` — ATR-based trend following
- `macd` — MACD histogram crossover

**Indian fee model included:** intraday (₹20 flat + 0.03%), delivery (0.1%), F&O (₹20 flat)

Reports saved to `reports/SYMBOL_STRATEGY.html`

---

## Claude MCP Integration

This system integrates with Claude Code and Claude Desktop via MCP.

Add to your Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json`):
- See `mcp/claude_desktop_config.json` for the full config

**Available MCP tools:**
- `get_positions` — List open positions
- `get_daily_pnl` — Today's P&L and risk metrics
- `get_market_status` — NSE open/closed status
- `run_backtest` — Run strategy backtest
- `place_paper_trade` — Simulate a trade
- `get_historical_data` — Fetch OHLCV data

**Zerodha official MCP** (no install required):
```json
{ "command": "npx", "args": ["-y", "mcp-remote", "https://mcp.kite.trade/mcp"] }
```

**TradingView MCP** (30+ indicators, signal analysis):
```json
{ "command": "uvx", "args": ["tradingview-mcp"] }
```

---

## Risk Management

Configured in `.env`:

| Variable               | Default | Description                          |
|------------------------|---------|--------------------------------------|
| `MAX_CAPITAL`          | 100000  | Total capital in INR                 |
| `RISK_PER_TRADE_PCT`   | 1.0     | % of capital risked per trade        |
| `MAX_OPEN_POSITIONS`   | 5       | Maximum concurrent open positions    |
| `DAILY_LOSS_LIMIT_PCT` | 3.0     | Stop trading at this daily loss %    |
| `MAX_POSITION_SIZE_PCT`| 20.0    | Max single position size %           |

---

## Key GitHub Repositories Used

| Repo | Purpose |
|------|---------|
| [marketcalls/openalgo](https://github.com/marketcalls/openalgo) | Reference: 30+ broker webhook platform |
| [zerodha/pykiteconnect](https://github.com/zerodha/pykiteconnect) | Zerodha SDK |
| [upstox/upstox-python](https://github.com/upstox/upstox-python) | Upstox SDK |
| [angel-one/smartapi-python](https://github.com/angel-one/smartapi-python) | AngelOne SDK |
| [Shoonya-Dev/ShoonyaApi-py](https://github.com/Shoonya-Dev/ShoonyaApi-py) | Finvasia SDK |
| [zerodha/kite-mcp-server](https://github.com/zerodha/kite-mcp-server) | Zerodha MCP (hosted) |
| [atilaahmettaner/tradingview-mcp](https://github.com/atilaahmettaner/tradingview-mcp) | TradingView MCP |
| [jugaad-py/jugaad-data](https://github.com/jugaad-py/jugaad-data) | NSE market data |
| [marketcalls/vectorbt-backtesting-skills](https://github.com/marketcalls/vectorbt-backtesting-skills) | Backtesting reference |

---

## Project Structure

```
india-autotrader/
├── main.py                  # FastAPI entrypoint
├── config/
│   ├── settings.py          # Pydantic env-driven config
│   └── strategies.yaml      # Per-strategy risk parameters
├── brokers/
│   ├── base.py              # Abstract broker interface
│   ├── zerodha.py           # Kite Connect
│   ├── upstox.py            # Upstox v2
│   ├── angelone.py          # SmartAPI
│   └── finvasia.py          # Shoonya
├── webhook/
│   ├── server.py            # FastAPI app + routes
│   └── parser.py            # TradingViewAlert schema
├── signals/
│   ├── processor.py         # Full signal pipeline
│   └── filters.py           # Market hours + symbol normalizer
├── risk/
│   ├── manager.py           # Risk checks (P&L, positions, size)
│   └── position_sizer.py    # ATR-based position sizing
├── data/
│   ├── nse_data.py          # NSE/BSE data via jugaad-data
│   └── historical.py        # OHLCV + technical indicators
├── backtest/
│   └── runner.py            # vectorbt + Indian fee model
├── mcp/
│   ├── server.py            # MCP tool provider
│   └── claude_desktop_config.json
├── utils/
│   ├── logger.py            # Structured logging
│   └── notifications.py     # Telegram alerts
├── tests/                   # pytest test suite
├── scripts/
│   └── backtest.py          # CLI backtest runner
├── docker-compose.yml
└── Dockerfile
```

---

## Disclaimer

This software is for educational purposes. Algorithmic trading carries significant financial risk.
Always test in paper mode before deploying real capital. The authors are not responsible for trading losses.
