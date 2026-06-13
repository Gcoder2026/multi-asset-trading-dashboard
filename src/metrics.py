"""Performance and risk metrics.

A consistent 252-trading-day annualisation is used as a simplifying assumption
for ETFs, stocks and crypto alike (noted in the dashboard Methodology tab).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def total_return(equity_curve: pd.Series) -> float:
    """Total return over the full equity curve."""
    if equity_curve is None or len(equity_curve) == 0:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Geometric annualised return from a series of periodic returns."""
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return 0.0
    cumulative = float((1.0 + returns).prod())
    if cumulative <= 0:
        return -1.0
    return cumulative ** (periods_per_year / n) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualised standard deviation of returns."""
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio (risk-free rate is an annual figure)."""
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (a negative number)."""
    if equity_curve is None or len(equity_curve) == 0:
        return 0.0
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    """Annualised return divided by the absolute maximum drawdown."""
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return np.nan
    return annualized_return(returns) / mdd


def win_rate(returns: pd.Series) -> float:
    """Share of active (non-zero) periods with a positive return."""
    active = returns.dropna()
    active = active[active != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def performance_summary(returns: pd.Series, equity_curve: pd.Series) -> dict:
    """Bundle all headline metrics into a dict (fractions, not percentages)."""
    return {
        "Total Return": total_return(equity_curve),
        "Annualized Return": annualized_return(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns),
        "Max Drawdown": max_drawdown(equity_curve),
        "Calmar Ratio": calmar_ratio(returns, equity_curve),
        "Win Rate": win_rate(returns),
    }
