"""Technical indicators: SMA, daily returns, momentum and drawdown."""

from __future__ import annotations

import pandas as pd


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` periods."""
    return series.rolling(window=window, min_periods=window).mean()


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage returns for each column."""
    return prices.pct_change()


def calculate_momentum(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Rolling momentum = trailing return over ``lookback_days``."""
    return prices.pct_change(lookback_days)


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Drawdown series relative to the running maximum of the equity curve."""
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0
