from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import structlog
import vectorbt as vbt

log = structlog.get_logger(__name__)

# ── Indian fee model ──────────────────────────────────────────────────────────
# SEBI/Exchange charges as of FY2025

def _indian_fee_model(
    segment: Literal["equity_delivery", "equity_intraday", "fo_futures", "fo_options", "fo_options_buy", "fo_options_sell"],
) -> dict:
    """
    Return accurate FY2025 per-trade cost model for vectorbt.
    Uses IndianFeeModel from backtest/fees.py.
    """
    from backtest.fees import IndianFeeModel, Segment
    model = IndianFeeModel()
    seg_map = {
        "equity_delivery": Segment.EQUITY_DELIVERY,
        "equity_intraday": Segment.EQUITY_INTRADAY,
        "fo_futures": Segment.FO_FUTURES,
        "fo_options": Segment.FO_OPTIONS_BUY,
        "fo_options_buy": Segment.FO_OPTIONS_BUY,
        "fo_options_sell": Segment.FO_OPTIONS_SELL,
    }
    seg = seg_map.get(segment, Segment.EQUITY_INTRADAY)
    return model.vectorbt_fee_config(seg)


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    report_path: str


class BacktestRunner:
    """
    Runs vectorbt backtests against NSE historical data with Indian fee modeling.
    Outputs QuantStats HTML tearsheets.
    """

    OUTPUT_DIR = Path(__file__).parent.parent / "reports"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(exist_ok=True)

    def run(
        self,
        symbol: str,
        strategy: str,
        from_date: str,
        to_date: str,
        segment: str = "equity_intraday",
        **strategy_params,
    ) -> BacktestResult:
        from data.historical import HistoricalDataFetcher

        log.info("backtest.start", symbol=symbol, strategy=strategy, from_date=from_date, to_date=to_date)

        fetcher = HistoricalDataFetcher()
        df = fetcher.get_with_indicators(symbol, from_date, to_date, indicators=self._required_indicators(strategy))

        if df.empty or len(df) < 50:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} rows")

        df = df.set_index("date")
        fees = _indian_fee_model(segment)

        entries, exits = self._generate_signals(df, strategy, **strategy_params)

        portfolio = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries,
            exits=exits,
            fees=fees["fees"],
            fixed_fees=fees["fixed_fees"],
            slippage=fees["slippage"],
            init_cash=100_000,
            freq="D",
        )

        result = self._extract_metrics(portfolio, symbol, strategy)
        self._save_tearsheet(portfolio, df, symbol, strategy, result.report_path)

        log.info(
            "backtest.complete",
            symbol=symbol,
            strategy=strategy,
            total_return=f"{result.total_return_pct:.1f}%",
            sharpe=f"{result.sharpe_ratio:.2f}",
            trades=result.total_trades,
        )
        return result

    # ── Signal generators ─────────────────────────────────────────────────

    def _required_indicators(self, strategy: str) -> list[str]:
        mapping = {
            "ema_cross": ["ema_9", "ema_21"],
            "rsi_mean_reversion": ["rsi_14"],
            "supertrend": ["supertrend"],
            "macd": ["macd"],
            "bb_squeeze": ["bb", "atr_14"],
        }
        return mapping.get(strategy, [])

    def _generate_signals(
        self, df: pd.DataFrame, strategy: str, **params
    ) -> tuple[pd.Series, pd.Series]:
        if strategy == "ema_cross":
            return self._ema_crossover(df, **params)
        if strategy == "rsi_mean_reversion":
            return self._rsi_mean_reversion(df, **params)
        if strategy == "supertrend":
            return self._supertrend_signals(df)
        if strategy == "macd":
            return self._macd_signals(df)
        raise ValueError(f"Unknown strategy: {strategy}")

    def _ema_crossover(
        self, df: pd.DataFrame, fast: int = 9, slow: int = 21
    ) -> tuple[pd.Series, pd.Series]:
        fast_col = f"ema_{fast}"
        slow_col = f"ema_{slow}"
        if fast_col not in df.columns:
            try:
                import pandas_ta as ta
            except ImportError:
                from utils import ta_compat as ta
            df[fast_col] = ta.ema(df["close"], length=fast)
            df[slow_col] = ta.ema(df["close"], length=slow)

        entries = (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1))
        exits = (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1))
        return entries, exits

    def _rsi_mean_reversion(
        self, df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70
    ) -> tuple[pd.Series, pd.Series]:
        rsi_col = f"rsi_{period}"
        if rsi_col not in df.columns:
            try:
                import pandas_ta as ta
            except ImportError:
                from utils import ta_compat as ta
            df[rsi_col] = ta.rsi(df["close"], length=period)

        entries = df[rsi_col] < oversold
        exits = df[rsi_col] > overbought
        return entries, exits

    def _supertrend_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        # Supertrend direction: 1 = bullish, -1 = bearish
        st_dir_col = [c for c in df.columns if c.startswith("SUPERTd")]
        if not st_dir_col:
            try:
                import pandas_ta as ta
            except ImportError:
                from utils import ta_compat as ta
            st = ta.supertrend(df["high"], df["low"], df["close"])
            df = pd.concat([df, st], axis=1)
            st_dir_col = [c for c in df.columns if c.startswith("SUPERTd")]

        direction = df[st_dir_col[0]]
        entries = (direction == 1) & (direction.shift(1) == -1)
        exits = (direction == -1) & (direction.shift(1) == 1)
        return entries, exits

    def _macd_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        macd_col = [c for c in df.columns if c.startswith("MACD_")]
        sig_col = [c for c in df.columns if c.startswith("MACDs_")]
        if not macd_col:
            try:
                import pandas_ta as ta
            except ImportError:
                from utils import ta_compat as ta
            macd = ta.macd(df["close"])
            df = pd.concat([df, macd], axis=1)
            macd_col = [c for c in df.columns if c.startswith("MACD_")]
            sig_col = [c for c in df.columns if c.startswith("MACDs_")]

        macd = df[macd_col[0]]
        signal = df[sig_col[0]]
        entries = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        exits = (macd < signal) & (macd.shift(1) >= signal.shift(1))
        return entries, exits

    # ── Metrics & output ──────────────────────────────────────────────────

    def _extract_metrics(self, portfolio, symbol: str, strategy: str) -> BacktestResult:
        stats = portfolio.stats()
        report_path = str(self.OUTPUT_DIR / f"{symbol}_{strategy}.html")
        return BacktestResult(
            symbol=symbol,
            strategy=strategy,
            total_return_pct=float(stats.get("Total Return [%]", 0)),
            cagr_pct=float(stats.get("Annualized Return [%]", 0)),
            sharpe_ratio=float(stats.get("Sharpe Ratio", 0)),
            max_drawdown_pct=float(stats.get("Max Drawdown [%]", 0)),
            win_rate_pct=float(stats.get("Win Rate [%]", 0)),
            total_trades=int(stats.get("Total Trades", 0)),
            report_path=report_path,
        )

    def _save_tearsheet(self, portfolio, df, symbol, strategy, output_path: str) -> None:
        try:
            import quantstats as qs
            returns = portfolio.returns()
            qs.reports.html(
                returns,
                output=output_path,
                title=f"{symbol} — {strategy} backtest",
                benchmark=None,
            )
            log.info("backtest.tearsheet_saved", path=output_path)
        except Exception as e:
            log.warning("backtest.tearsheet_failed", error=str(e))
