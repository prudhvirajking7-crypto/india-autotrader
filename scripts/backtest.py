"""
CLI backtest runner.

Usage:
    python scripts/backtest.py --symbol RELIANCE --strategy ema_cross --days 180
    python scripts/backtest.py --symbol NIFTY --strategy supertrend --segment fo_futures
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Run backtests on NSE symbols with Indian fee modeling")
console = Console()


@app.command()
def run(
    symbol: str = typer.Option(..., "--symbol", "-s", help="NSE symbol e.g. RELIANCE"),
    strategy: str = typer.Option("ema_cross", "--strategy", help="Strategy name"),
    days: int = typer.Option(180, "--days", "-d", help="Lookback period in days"),
    segment: str = typer.Option("equity_intraday", "--segment", help="Market segment"),
):
    from datetime import datetime, timedelta
    from backtest.runner import BacktestRunner

    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    console.print(f"\n[bold cyan]Running backtest:[/] {symbol} / {strategy} | {from_date} → {to_date}\n")

    runner = BacktestRunner()
    result = runner.run(symbol, strategy, from_date, to_date, segment)

    table = Table(title=f"{symbol} — {strategy}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    rows = [
        ("Total Return", f"{result.total_return_pct:.2f}%"),
        ("CAGR", f"{result.cagr_pct:.2f}%"),
        ("Sharpe Ratio", f"{result.sharpe_ratio:.2f}"),
        ("Max Drawdown", f"{result.max_drawdown_pct:.2f}%"),
        ("Win Rate", f"{result.win_rate_pct:.2f}%"),
        ("Total Trades", str(result.total_trades)),
        ("Report", result.report_path),
    ]
    for metric, value in rows:
        table.add_row(metric, value)

    console.print(table)


@app.command()
def scan(
    strategy: str = typer.Option("ema_cross", "--strategy"),
    days: int = typer.Option(90, "--days"),
):
    """Scan NIFTY 50 stocks and rank by backtest Sharpe ratio."""
    from data.nse_data import NSEDataProvider
    from backtest.runner import BacktestRunner
    from datetime import datetime, timedelta

    symbols = NSEDataProvider().get_nifty50_symbols()[:10]  # top 10 for demo
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    runner = BacktestRunner()
    results = []

    with console.status("[bold green]Scanning NIFTY 50..."):
        for sym in symbols:
            try:
                r = runner.run(sym, strategy, from_date, to_date)
                results.append(r)
            except Exception as e:
                console.print(f"[red]Skipped {sym}: {e}")

    results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

    table = Table(title=f"NIFTY 50 Scan — {strategy}", show_header=True)
    for col in ["Symbol", "Return %", "Sharpe", "Drawdown %", "Win Rate %", "Trades"]:
        table.add_column(col)

    for r in results:
        table.add_row(
            r.symbol,
            f"{r.total_return_pct:.1f}",
            f"{r.sharpe_ratio:.2f}",
            f"{r.max_drawdown_pct:.1f}",
            f"{r.win_rate_pct:.1f}",
            str(r.total_trades),
        )
    console.print(table)


if __name__ == "__main__":
    app()
