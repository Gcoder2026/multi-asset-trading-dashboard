"""Backtesting engine for single-asset and portfolio strategies.

Look-ahead bias is avoided by using positions/weights that are already shifted
relative to the returns they earn. Transaction costs are charged on traded
notional (turnover) at a configurable basis-point rate.
"""

from __future__ import annotations

import pandas as pd


def backtest_single_asset_strategy(
    price_series: pd.Series,
    position: pd.Series,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Backtest a single-asset long/cash strategy.

    ``position`` is the already-shifted holding (1 = invested, 0 = cash) so
    that returns on day *t* are earned by the position decided at *t-1*.

    Returns a DataFrame with columns:
        position, asset_return, strategy_return, cost,
        net_return, equity_curve
    """
    asset_return = price_series.pct_change().fillna(0.0)
    position = position.reindex(price_series.index).fillna(0.0)

    strategy_return = position * asset_return
    # A trade happens whenever the position changes; cost on the traded amount.
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    net_return = strategy_return - cost

    equity_curve = (1.0 + net_return).cumprod()

    return pd.DataFrame(
        {
            "position": position,
            "asset_return": asset_return,
            "strategy_return": strategy_return,
            "cost": cost,
            "net_return": net_return,
            "equity_curve": equity_curve,
        }
    )


def backtest_portfolio_strategy(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Backtest a multi-asset portfolio given target weights over time.

    Weights are shifted by one day before earning returns (no look-ahead).
    Transaction costs are charged on portfolio turnover at each change.

    Returns a DataFrame with columns:
        portfolio_return, cost, net_return, equity_curve
    """
    returns = prices.pct_change().fillna(0.0)
    weights = weights.reindex(prices.index).fillna(0.0)

    # Act on yesterday's target weights to avoid look-ahead bias.
    held_weights = weights.shift(1).fillna(0.0)

    portfolio_return = (held_weights * returns).sum(axis=1)
    turnover = (held_weights - held_weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = turnover * (transaction_cost_bps / 10_000.0)
    net_return = portfolio_return - cost

    equity_curve = (1.0 + net_return).cumprod()

    return pd.DataFrame(
        {
            "portfolio_return": portfolio_return,
            "cost": cost,
            "net_return": net_return,
            "equity_curve": equity_curve,
        }
    )


def calculate_buy_and_hold(price_series: pd.Series) -> pd.Series:
    """Buy-and-hold equity curve (starts at 1.0)."""
    returns = price_series.pct_change().fillna(0.0)
    return (1.0 + returns).cumprod()


def calculate_equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight, daily-rebalanced benchmark equity curve (starts at 1.0)."""
    returns = prices.pct_change().fillna(0.0)
    equal_weight_return = returns.mean(axis=1)
    return (1.0 + equal_weight_return).cumprod()
