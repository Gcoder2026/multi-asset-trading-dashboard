"""Rule-based trading strategies.

Strategy 1: Moving Average Crossover (single asset, long-only).
Strategy 2: Momentum Ranking Portfolio (multi-asset, monthly rebalance).

These rule-based strategies are the single source of truth for all signals.
The optional LLM module only *explains* their output; it never decides trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import calculate_sma


# --------------------------------------------------------------------------- #
# Strategy 1: Moving Average Crossover
# --------------------------------------------------------------------------- #
def generate_ma_signals(
    price_series: pd.Series,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """Generate long-only moving-average crossover signals for one asset.

    Output columns:
        price, short_sma, long_sma, raw_signal, position,
        trade_signal, current_signal

    ``raw_signal`` is 1 when short SMA > long SMA, else 0.
    ``position`` is the shifted raw signal (no look-ahead: a signal observed at
    today's close is only acted on from the next day).
    ``trade_signal`` labels each day Buy / Sell / Hold / Cash.
    """
    df = pd.DataFrame(index=price_series.index)
    df["price"] = price_series
    df["short_sma"] = calculate_sma(price_series, short_window)
    df["long_sma"] = calculate_sma(price_series, long_window)

    df["raw_signal"] = (df["short_sma"] > df["long_sma"]).astype(int)
    # Shift to avoid look-ahead bias: act on the prior close's signal.
    df["position"] = df["raw_signal"].shift(1).fillna(0).astype(int)

    change = df["position"].diff().fillna(0)

    def _label(row_change: float, pos: int) -> str:
        if row_change == 1:
            return "Buy"
        if row_change == -1:
            return "Sell"
        return "Hold" if pos == 1 else "Cash"

    df["trade_signal"] = [
        _label(c, p) for c, p in zip(change, df["position"])
    ]
    df["current_signal"] = df["trade_signal"].iloc[-1] if len(df) else "Cash"
    return df


def get_current_ma_signal(signal_df: pd.DataFrame) -> dict:
    """Return the latest signal label plus a plain-English reason."""
    if signal_df.empty:
        return {"signal": "Cash", "reason": "No data available.", "date": None}

    last = signal_df.iloc[-1]
    signal = last["trade_signal"]
    short_above = bool(last["short_sma"] > last["long_sma"])

    reasons = {
        "Buy": "The short SMA has just crossed above the long SMA, so the model moves to invested.",
        "Hold": "The short SMA remains above the long SMA, so the model stays invested.",
        "Sell": "The short SMA has just crossed below the long SMA, so the model moves to cash.",
        "Cash": "The short SMA remains below the long SMA, so the model stays in cash.",
    }
    # Fall back to a position-based reason if SMAs are still warming up.
    reason = reasons.get(signal)
    if reason is None or pd.isna(last["short_sma"]):
        reason = (
            "The short SMA is above the long SMA."
            if short_above
            else "The short SMA is below the long SMA."
        )

    return {
        "signal": signal,
        "reason": reason,
        "date": signal_df.index[-1],
        "price": float(last["price"]) if not pd.isna(last["price"]) else None,
        "short_sma": float(last["short_sma"]) if not pd.isna(last["short_sma"]) else None,
        "long_sma": float(last["long_sma"]) if not pd.isna(last["long_sma"]) else None,
    }


# --------------------------------------------------------------------------- #
# Strategy 2: Momentum Ranking Portfolio
# --------------------------------------------------------------------------- #
def _rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading day of each month in the index."""
    grouped = pd.Series(index, index=index).groupby([index.year, index.month]).last()
    return pd.DatetimeIndex(sorted(grouped.values))


def generate_momentum_weights(
    prices: pd.DataFrame,
    lookback_days: int,
    top_n: int,
    positive_momentum_filter: bool = True,
    max_weight_per_asset: float = 0.40,
) -> pd.DataFrame:
    """Monthly momentum-ranked, equal-weight portfolio weights.

    On the last trading day of each month, rank assets by trailing
    ``lookback_days`` return, pick the top ``top_n`` (optionally requiring
    positive momentum), and equal-weight them. Non-selected assets get 0.
    Weights are held until the next rebalance.
    """
    if prices.empty:
        return prices.copy()

    momentum = prices.pct_change(lookback_days)
    rebal_dates = _rebalance_dates(prices.index)

    rebal_weights = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)
    for date in rebal_dates:
        mom = momentum.loc[date].dropna()
        if positive_momentum_filter:
            mom = mom[mom > 0]
        if mom.empty:
            continue  # everything stays in cash this month
        selected = mom.sort_values(ascending=False).head(top_n)
        weight = min(1.0 / len(selected), max_weight_per_asset)
        rebal_weights.loc[date, selected.index] = weight

    # Hold each rebalance's weights until the next one; 0 before the first.
    weights = rebal_weights.reindex(prices.index, method="ffill").fillna(0.0)
    return weights


def get_current_momentum_signals(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    lookback_days: int,
) -> pd.DataFrame:
    """Build the current signal/ranking table for the latest rebalance.

    Columns: Asset, Momentum, Model Weight, Signal, Strategy Role
    (Momentum and Model Weight are fractions, formatted by the dashboard.)
    """
    if prices.empty or weights.empty:
        return pd.DataFrame(
            columns=["Asset", "Momentum", "Model Weight", "Signal", "Strategy Role"]
        )

    latest_mom = prices.pct_change(lookback_days).iloc[-1]
    latest_weights = weights.iloc[-1]
    # Previous distinct weight vector to detect assets dropped at last rebalance.
    prev_weights = weights[(weights != latest_weights).any(axis=1)]
    prev_weights = prev_weights.iloc[-1] if not prev_weights.empty else latest_weights

    rows = []
    for asset in prices.columns:
        mom = latest_mom.get(asset, np.nan)
        weight = float(latest_weights.get(asset, 0.0))
        prev_w = float(prev_weights.get(asset, 0.0))

        if weight > 0:
            signal = "Buy / Hold" if (pd.notna(mom) and mom > 0) else "Hold"
            role = "Selected top momentum asset"
        else:
            if prev_w > 0:
                signal = "Sell / Reduce"
                role = "Dropped out of top N"
            elif pd.notna(mom) and mom <= 0:
                signal = "Cash / Avoid"
                role = "Weak momentum"
            else:
                signal = "Cash / Avoid"
                role = "Not selected this rebalance"

        rows.append(
            {
                "Asset": asset,
                "Momentum": float(mom) if pd.notna(mom) else np.nan,
                "Model Weight": weight,
                "Signal": signal,
                "Strategy Role": role,
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values("Momentum", ascending=False, na_position="last").reset_index(drop=True)
