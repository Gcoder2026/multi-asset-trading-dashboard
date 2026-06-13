"""Plotly chart builders for the Streamlit dashboard.

A small, restrained palette is used so the dashboard reads cleanly.
Each function returns a Plotly figure; the app handles layout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Restrained, consistent palette.
PRICE_COLOR = "#1f77b4"
SHORT_SMA_COLOR = "#ff7f0e"
LONG_SMA_COLOR = "#6c757d"
STRATEGY_COLOR = "#2ca02c"
BENCHMARK_COLOR = "#6c757d"
DRAWDOWN_COLOR = "#d62728"

_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
)


def plot_price_with_sma(signal_df: pd.DataFrame, title: str = "Price with Moving Averages") -> go.Figure:
    """Price line + short/long SMAs + Buy/Sell markers."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=signal_df.index, y=signal_df["price"], name="Price", line=dict(color=PRICE_COLOR, width=1.5))
    )
    fig.add_trace(
        go.Scatter(x=signal_df.index, y=signal_df["short_sma"], name="Short SMA", line=dict(color=SHORT_SMA_COLOR, width=1.2))
    )
    fig.add_trace(
        go.Scatter(x=signal_df.index, y=signal_df["long_sma"], name="Long SMA", line=dict(color=LONG_SMA_COLOR, width=1.2))
    )

    buys = signal_df[signal_df["trade_signal"] == "Buy"]
    sells = signal_df[signal_df["trade_signal"] == "Sell"]
    fig.add_trace(
        go.Scatter(
            x=buys.index, y=buys["price"], name="Buy", mode="markers",
            marker=dict(symbol="triangle-up", color=STRATEGY_COLOR, size=11, line=dict(width=1, color="white")),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index, y=sells["price"], name="Sell", mode="markers",
            marker=dict(symbol="triangle-down", color=DRAWDOWN_COLOR, size=11, line=dict(width=1, color="white")),
        )
    )
    fig.update_layout(title=title, yaxis_title="Price", **_LAYOUT)
    return fig


def plot_equity_curve(
    strategy_curve: pd.Series,
    benchmark_curve: pd.Series,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
) -> go.Figure:
    """Strategy vs benchmark cumulative growth of 1.0."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=strategy_curve.index, y=strategy_curve.values, name=strategy_name, line=dict(color=STRATEGY_COLOR, width=2))
    )
    fig.add_trace(
        go.Scatter(x=benchmark_curve.index, y=benchmark_curve.values, name=benchmark_name, line=dict(color=BENCHMARK_COLOR, width=1.6, dash="dash"))
    )
    fig.update_layout(title="Cumulative Return (growth of 1.0)", yaxis_title="Equity (×)", **_LAYOUT)
    return fig


def plot_drawdown(drawdown_series: pd.Series) -> go.Figure:
    """Filled drawdown area chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown_series.index, y=drawdown_series.values * 100.0, name="Drawdown",
            fill="tozeroy", line=dict(color=DRAWDOWN_COLOR, width=1),
        )
    )
    fig.update_layout(title="Drawdown", yaxis_title="Drawdown (%)", **_LAYOUT)
    return fig


def plot_asset_correlation(returns: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of daily returns."""
    corr = returns.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(title="Return Correlation", margin=dict(l=40, r=20, t=50, b=40), template="plotly_white")
    return fig


def plot_risk_return_scatter(summary_table: pd.DataFrame) -> go.Figure:
    """Scatter of annualised volatility (x) vs annualised return (y) per asset.

    Expects columns: 'Asset', 'Annualized Volatility', 'Annualized Return',
    and optionally 'Sharpe Ratio' (used for marker size/color).
    """
    df = summary_table.copy()
    size = None
    if "Sharpe Ratio" in df.columns:
        size = (df["Sharpe Ratio"].clip(lower=0) + 0.1) * 10
    fig = px.scatter(
        df,
        x="Annualized Volatility",
        y="Annualized Return",
        text="Asset",
        color="Sharpe Ratio" if "Sharpe Ratio" in df.columns else None,
        size=size,
        color_continuous_scale="Viridis",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        title="Risk vs Return",
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_portfolio_weights(weights: pd.DataFrame) -> go.Figure:
    """Stacked area chart of portfolio weights over time."""
    fig = go.Figure()
    # Only plot assets that ever receive weight, to keep the legend tidy.
    active = [c for c in weights.columns if weights[c].abs().sum() > 0]
    for col in active:
        fig.add_trace(
            go.Scatter(
                x=weights.index, y=weights[col] * 100.0, name=col,
                stackgroup="one", mode="none",
            )
        )
    fig.update_layout(title="Portfolio Weights Over Time", yaxis_title="Weight (%)", **_LAYOUT)
    return fig


def plot_current_signal_table(signal_table: pd.DataFrame) -> go.Figure:
    """Render the current signal table as a Plotly table (optional helper)."""
    display = signal_table.copy()
    if "Momentum" in display.columns:
        display["Momentum"] = (display["Momentum"] * 100).map(lambda v: f"{v:.1f}%" if pd.notna(v) else "n/a")
    if "Model Weight" in display.columns:
        display["Model Weight"] = (display["Model Weight"] * 100).map(lambda v: f"{v:.1f}%")
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(display.columns), fill_color="#f0f2f6", align="left"),
                cells=dict(values=[display[c] for c in display.columns], align="left"),
            )
        ]
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig
