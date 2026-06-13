"""LLM-Enhanced Multi-Asset Trading Strategy Backtesting Dashboard.

Run locally with:

    streamlit run app.py

This is a research-oriented backtesting and signal visualization dashboard,
not a live trading bot or automated execution system. The optional Gemini LLM
module only explains rule-based outputs; it does not make trading decisions.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src import llm_commentary, plots
from src.backtester import (
    backtest_portfolio_strategy,
    backtest_single_asset_strategy,
    calculate_buy_and_hold,
    calculate_equal_weight_benchmark,
)
from src.data_loader import get_close_prices, load_price_data
from src.indicators import calculate_daily_returns, calculate_drawdown
from src.metrics import (
    annualized_return,
    annualized_volatility,
    performance_summary,
    sharpe_ratio,
)
from src.strategies import (
    generate_ma_signals,
    generate_momentum_weights,
    get_current_ma_signal,
    get_current_momentum_signals,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ASSET_UNIVERSE: dict[str, dict[str, str]] = {
    "SPY": {"type": "Equity ETF", "purpose": "S&P 500 / US large-cap benchmark"},
    "QQQ": {"type": "Equity ETF", "purpose": "Nasdaq-100 / technology growth exposure"},
    "TLT": {"type": "Bond ETF", "purpose": "Long-duration US Treasury defensive asset"},
    "GLD": {"type": "Gold ETF", "purpose": "Gold / safe-haven asset proxy"},
    "BTC-USD": {"type": "Crypto", "purpose": "Bitcoin high-volatility risk asset"},
    "ETH-USD": {"type": "Crypto", "purpose": "Ethereum second crypto comparison"},
    "AAPL": {"type": "Stock", "purpose": "Large-cap technology stock"},
    "MSFT": {"type": "Stock", "purpose": "Large-cap technology stock"},
}
ALL_TICKERS = list(ASSET_UNIVERSE.keys())
DEFAULT_SELECTION = ["SPY", "QQQ", "GLD", "BTC-USD", "AAPL", "MSFT"]

DISCLAIMER = (
    "This project is a research-oriented backtesting and signal visualization "
    "dashboard, not a live trading bot or automated execution system. The "
    "Gemini LLM module is used only to explain rule-based outputs; it does not "
    "make trading decisions. Nothing here is financial advice."
)

st.set_page_config(
    page_title="Multi-Asset Trading Strategy Backtesting Dashboard",
    page_icon="📈",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Downloading market data from yfinance…")
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    data, failed = load_price_data(list(tickers), start, end)
    prices = get_close_prices(data)
    return prices, failed


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def fmt_pct(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.2%}"


def fmt_num(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.2f}"


def metrics_to_display(summary: dict) -> pd.DataFrame:
    """Format a performance_summary dict into a tidy two-column table."""
    pct_keys = {"Total Return", "Annualized Return", "Annualized Volatility", "Max Drawdown", "Win Rate"}
    rows = []
    for key, value in summary.items():
        formatted = fmt_pct(value) if key in pct_keys else fmt_num(value)
        rows.append({"Metric": key, "Value": formatted})
    return pd.DataFrame(rows)


def compute_asset_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-asset buy-and-hold risk/return summary for the scatter plot."""
    returns = calculate_daily_returns(prices)
    rows = []
    for asset in prices.columns:
        r = returns[asset].dropna()
        rows.append(
            {
                "Asset": asset,
                "Annualized Return": annualized_return(r),
                "Annualized Volatility": annualized_volatility(r),
                "Sharpe Ratio": sharpe_ratio(r),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio(
    "Mode",
    ["Single Asset Mode", "Multi-Asset Mode"],
    help="Single Asset: analyse one asset with a moving-average strategy. "
    "Multi-Asset: rank several assets with a momentum portfolio strategy.",
)

selected_assets = st.sidebar.multiselect(
    "Asset universe",
    options=ALL_TICKERS,
    default=DEFAULT_SELECTION,
    help="Choose the assets to analyse from the fixed universe.",
)
if not selected_assets:
    st.sidebar.warning("Select at least one asset.")
    selected_assets = ["SPY"]

single_asset = None
if mode == "Single Asset Mode":
    single_asset = st.sidebar.selectbox("Asset (single)", options=selected_assets)

st.sidebar.subheader("Date range")
default_start = dt.date(2018, 1, 1)
today = dt.date.today()
col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("Start", value=default_start, min_value=dt.date(2010, 1, 1), max_value=today)
end_date = col_b.date_input("End", value=today, min_value=default_start, max_value=today)

st.sidebar.subheader("Strategy parameters")
short_window = st.sidebar.number_input("Short SMA window", min_value=2, max_value=200, value=20, step=1)
long_window = st.sidebar.number_input("Long SMA window", min_value=5, max_value=400, value=50, step=1)
lookback_days = st.sidebar.number_input("Momentum lookback (days)", min_value=10, max_value=252, value=63, step=1)
top_n = st.sidebar.number_input("Top N assets", min_value=1, max_value=len(ALL_TICKERS), value=3, step=1)
positive_filter = st.sidebar.checkbox("Positive momentum filter", value=True)
tx_cost_bps = st.sidebar.number_input("Transaction cost (bps)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
initial_capital = st.sidebar.number_input("Initial capital", min_value=100, max_value=10_000_000, value=10_000, step=1000)

if short_window >= long_window:
    st.sidebar.warning("Short SMA window should be smaller than the long SMA window.")

st.sidebar.subheader("AI commentary")
enable_ai = st.sidebar.checkbox("Enable AI commentary", value=False)
gemini_available = llm_commentary.is_gemini_available()
if gemini_available:
    st.sidebar.success("Gemini API key detected.")
else:
    st.sidebar.info("No Gemini API key found — AI commentary disabled.")


# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
prices, failed = load_prices(tuple(selected_assets), start_date.isoformat(), end_date.isoformat())

if failed:
    st.warning(f"Could not download data for: {', '.join(failed)}. Continuing with the rest.")

if prices.empty:
    st.error(
        "No price data could be loaded. This is usually a temporary yfinance / "
        "network issue. Please check your connection and try again."
    )
    st.stop()

returns = calculate_daily_returns(prices)


# --------------------------------------------------------------------------- #
# Run the selected strategy
# --------------------------------------------------------------------------- #
results: dict = {"mode": mode}

if mode == "Single Asset Mode":
    if single_asset not in prices.columns:
        single_asset = prices.columns[0]
    price_series = prices[single_asset]

    signal_df = generate_ma_signals(price_series, int(short_window), int(long_window))
    current = get_current_ma_signal(signal_df)
    bt = backtest_single_asset_strategy(price_series, signal_df["position"], float(tx_cost_bps))

    strategy_equity = bt["equity_curve"]
    benchmark_equity = calculate_buy_and_hold(price_series)
    strategy_summary = performance_summary(bt["net_return"], strategy_equity)
    benchmark_summary = performance_summary(price_series.pct_change().fillna(0.0), benchmark_equity)
    drawdown = calculate_drawdown(strategy_equity)

    results.update(
        asset=single_asset,
        signal_df=signal_df,
        current=current,
        strategy_equity=strategy_equity,
        benchmark_equity=benchmark_equity,
        strategy_summary=strategy_summary,
        benchmark_summary=benchmark_summary,
        drawdown=drawdown,
    )
else:
    weights = generate_momentum_weights(
        prices,
        int(lookback_days),
        int(top_n),
        positive_momentum_filter=bool(positive_filter),
    )
    signal_table = get_current_momentum_signals(prices, weights, int(lookback_days))
    bt = backtest_portfolio_strategy(prices, weights, float(tx_cost_bps))

    strategy_equity = bt["equity_curve"]
    benchmark_equity = calculate_equal_weight_benchmark(prices)
    strategy_summary = performance_summary(bt["net_return"], strategy_equity)
    benchmark_summary = performance_summary(benchmark_equity.pct_change().fillna(0.0), benchmark_equity)
    drawdown = calculate_drawdown(strategy_equity)
    allocation = signal_table[signal_table["Model Weight"] > 0].copy()

    results.update(
        weights=weights,
        signal_table=signal_table,
        allocation=allocation,
        strategy_equity=strategy_equity,
        benchmark_equity=benchmark_equity,
        strategy_summary=strategy_summary,
        benchmark_summary=benchmark_summary,
        drawdown=drawdown,
    )


# --------------------------------------------------------------------------- #
# Header + headline metric cards
# --------------------------------------------------------------------------- #
st.title("📈 Multi-Asset Trading Strategy Backtesting Dashboard")
st.caption("Research-oriented backtesting and signal visualization for ETFs, stocks, and crypto.")

card_cols = st.columns(5)
if mode == "Single Asset Mode":
    headline_signal = results["current"]["signal"]
else:
    headline_signal = f"{int((results['signal_table']['Model Weight'] > 0).sum())} held"

card_cols[0].metric("Current Signal", headline_signal)
card_cols[1].metric("Strategy Return", fmt_pct(results["strategy_summary"]["Total Return"]))
card_cols[2].metric("Benchmark Return", fmt_pct(results["benchmark_summary"]["Total Return"]))
card_cols[3].metric("Sharpe Ratio", fmt_num(results["strategy_summary"]["Sharpe Ratio"]))
card_cols[4].metric("Max Drawdown", fmt_pct(results["strategy_summary"]["Max Drawdown"]))


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_overview, tab_signals, tab_market, tab_trading, tab_backtest, tab_portfolio, tab_ai, tab_method = st.tabs(
    [
        "Overview",
        "Current Signals",
        "Market Data",
        "Trading Signals",
        "Backtest Results",
        "Portfolio Analysis",
        "AI Commentary",
        "Methodology",
    ]
)

# ---- Overview ------------------------------------------------------------- #
with tab_overview:
    st.subheader("Project")
    st.write(
        "This dashboard generates **rule-based** Buy / Hold / Sell / Cash signals for a "
        "fixed universe of ETFs, stocks, and crypto, backtests them with transaction "
        "costs, and compares them against a benchmark. It supports two modes: a "
        "single-asset **moving-average crossover** strategy and a multi-asset **momentum "
        "ranking portfolio**."
    )
    st.info(f"**Current mode:** {mode}")
    st.warning(DISCLAIMER)

    st.subheader("Asset universe")
    universe_df = pd.DataFrame(
        [{"Ticker": t, "Type": v["type"], "Purpose": v["purpose"]} for t, v in ASSET_UNIVERSE.items()]
    )
    st.dataframe(universe_df, hide_index=True, width="stretch")

# ---- Current Signals ------------------------------------------------------ #
with tab_signals:
    if mode == "Single Asset Mode":
        current = results["current"]
        signal = current["signal"]
        color = {"Buy": "🟢", "Hold": "🔵", "Sell": "🔴", "Cash": "⚪"}.get(signal, "🔵")
        st.subheader(f"Current signal for {results['asset']}: {color} {signal.upper()}")
        st.write(f"**Reason:** {current['reason']}")
        info_cols = st.columns(3)
        info_cols[0].metric("Last price", fmt_num(current.get("price")))
        info_cols[1].metric(f"Short SMA ({int(short_window)})", fmt_num(current.get("short_sma")))
        info_cols[2].metric(f"Long SMA ({int(long_window)})", fmt_num(current.get("long_sma")))
        st.caption("Signals are rule-based and shifted to avoid look-ahead bias.")
    else:
        st.subheader("Current signals across selected assets")
        display = results["signal_table"].copy()
        display["Momentum"] = display["Momentum"].map(fmt_pct)
        display["Model Weight"] = display["Model Weight"].map(fmt_pct)
        st.dataframe(display, hide_index=True, width="stretch")

        st.subheader("Model allocation (current rebalance)")
        alloc = results["allocation"].copy()
        if alloc.empty:
            st.info("No assets currently selected — the model is fully in cash this rebalance.")
        else:
            alloc["Momentum"] = alloc["Momentum"].map(fmt_pct)
            alloc["Model Weight"] = alloc["Model Weight"].map(fmt_pct)
            st.dataframe(
                alloc[["Asset", "Momentum", "Model Weight", "Signal", "Strategy Role"]],
                hide_index=True,
                width="stretch",
            )
        st.caption("Use terms like *model output*, *strategy signal*, or *model allocation* — not financial advice.")

# ---- Market Data ---------------------------------------------------------- #
with tab_market:
    st.subheader("Price history")
    if mode == "Single Asset Mode":
        st.line_chart(prices[results["asset"]], height=380)
    else:
        normalized = prices / prices.iloc[0]
        st.caption("Prices normalised to 1.0 at the start of the period for comparison.")
        st.line_chart(normalized, height=380)

    st.subheader("Daily returns summary")
    summary_stats = pd.DataFrame(
        {
            "Annualized Return": {a: annualized_return(returns[a].dropna()) for a in prices.columns},
            "Annualized Volatility": {a: annualized_volatility(returns[a].dropna()) for a in prices.columns},
            "Sharpe Ratio": {a: sharpe_ratio(returns[a].dropna()) for a in prices.columns},
        }
    )
    st.dataframe(
        summary_stats.style.format(
            {
                "Annualized Return": "{:.2%}",
                "Annualized Volatility": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
            }
        ),
        width="stretch",
    )

    if len(prices.columns) > 1:
        st.subheader("Return correlation")
        st.plotly_chart(plots.plot_asset_correlation(returns.dropna()), width="stretch")
    else:
        st.info("Select more than one asset to see the correlation heatmap.")

# ---- Trading Signals ------------------------------------------------------ #
with tab_trading:
    if mode == "Single Asset Mode":
        st.subheader(f"Moving averages & signals — {results['asset']}")
        st.plotly_chart(
            plots.plot_price_with_sma(
                results["signal_df"],
                title=f"{results['asset']} price with {int(short_window)}/{int(long_window)} SMA",
            ),
            width="stretch",
        )
        st.subheader("Historical signal table (most recent first)")
        hist = results["signal_df"][["price", "short_sma", "long_sma", "position", "trade_signal"]].copy()
        hist = hist.dropna(subset=["short_sma", "long_sma"]).tail(250).iloc[::-1]
        st.dataframe(hist.style.format({"price": "{:.2f}", "short_sma": "{:.2f}", "long_sma": "{:.2f}"}), width="stretch")
    else:
        st.subheader("Momentum ranking")
        table = results["signal_table"].copy()
        chart_df = table.dropna(subset=["Momentum"]).set_index("Asset")["Momentum"] * 100.0
        st.bar_chart(chart_df, height=360)
        st.caption(f"{int(lookback_days)}-day trailing momentum per asset (most recent observation).")
        display = table.copy()
        display["Momentum"] = display["Momentum"].map(fmt_pct)
        display["Model Weight"] = display["Model Weight"].map(fmt_pct)
        st.dataframe(display, hide_index=True, width="stretch")

# ---- Backtest Results ----------------------------------------------------- #
with tab_backtest:
    bench_name = "Buy & Hold" if mode == "Single Asset Mode" else "Equal-Weight Benchmark"
    strat_name = "MA Strategy" if mode == "Single Asset Mode" else "Momentum Portfolio"
    st.subheader("Cumulative return vs benchmark")
    st.plotly_chart(
        plots.plot_equity_curve(results["strategy_equity"], results["benchmark_equity"], strat_name, bench_name),
        width="stretch",
    )

    final_value = results["strategy_equity"].iloc[-1] * initial_capital
    st.caption(f"Starting from {initial_capital:,.0f}, the strategy ends at **{final_value:,.0f}** (before taxes).")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"**{strat_name} metrics**")
        st.dataframe(metrics_to_display(results["strategy_summary"]), hide_index=True, width="stretch")
    with col_right:
        st.markdown(f"**{bench_name} metrics**")
        st.dataframe(metrics_to_display(results["benchmark_summary"]), hide_index=True, width="stretch")

    st.subheader("Drawdown")
    st.plotly_chart(plots.plot_drawdown(results["drawdown"]), width="stretch")

# ---- Portfolio Analysis --------------------------------------------------- #
with tab_portfolio:
    if mode == "Multi-Asset Mode":
        st.subheader("Selected assets (current rebalance)")
        alloc = results["allocation"]
        if alloc.empty:
            st.info("No assets currently selected — fully in cash.")
        else:
            st.write(", ".join(alloc["Asset"].tolist()))
        st.subheader("Portfolio weights over time")
        st.plotly_chart(plots.plot_portfolio_weights(results["weights"]), width="stretch")
    else:
        st.info(
            "Portfolio analytics are most meaningful in Multi-Asset Mode. The risk-return "
            "scatter below covers all currently selected assets."
        )

    if len(prices.columns) > 1:
        st.subheader("Risk vs return (buy-and-hold, per asset)")
        st.plotly_chart(plots.plot_risk_return_scatter(compute_asset_summary(prices)), width="stretch")

# ---- AI Commentary -------------------------------------------------------- #
with tab_ai:
    st.subheader("AI Commentary (optional)")
    st.caption(
        "Gemini explains the rule-based signals and metrics. It does **not** generate or "
        "override the signals. The rule-based strategy is the source of truth."
    )

    if not enable_ai:
        st.info("Enable **AI commentary** in the sidebar to use this feature.")
    elif not gemini_available:
        st.warning(llm_commentary.NO_KEY_MESSAGE)
        st.caption("Add GEMINI_API_KEY to a .env file, environment variable, or Streamlit secrets.")
    else:
        if mode == "Single Asset Mode":
            current = results["current"]
            st.markdown(
                f"- **Asset:** {results['asset']}\n"
                f"- **Current signal:** {current['signal']}\n"
                f"- **Reason:** {current['reason']}\n"
                f"- **Strategy return:** {fmt_pct(results['strategy_summary']['Total Return'])} "
                f"vs buy-and-hold {fmt_pct(results['benchmark_summary']['Total Return'])}\n"
                f"- **Sharpe:** {fmt_num(results['strategy_summary']['Sharpe Ratio'])} · "
                f"**Volatility:** {fmt_pct(results['strategy_summary']['Annualized Volatility'])} · "
                f"**Max drawdown:** {fmt_pct(results['strategy_summary']['Max Drawdown'])}"
            )
            if st.button("Generate AI Commentary", type="primary"):
                with st.spinner("Asking Gemini to explain the model output…"):
                    prompt = llm_commentary.build_signal_commentary_prompt(
                        asset=results["asset"],
                        signal=current["signal"],
                        signal_reason=current["reason"],
                        metrics=results["strategy_summary"],
                        benchmark_metrics=results["benchmark_summary"],
                        strategy_type="Moving Average Crossover",
                    )
                    st.session_state["ai_single"] = llm_commentary.generate_signal_commentary(prompt)
            if "ai_single" in st.session_state:
                st.markdown("---")
                st.markdown(st.session_state["ai_single"])
        else:
            alloc = results["allocation"]
            st.markdown(
                f"- **Selected assets:** {', '.join(alloc['Asset'].tolist()) if not alloc.empty else 'none (all cash)'}\n"
                f"- **Portfolio return:** {fmt_pct(results['strategy_summary']['Total Return'])} "
                f"vs equal-weight {fmt_pct(results['benchmark_summary']['Total Return'])}\n"
                f"- **Sharpe:** {fmt_num(results['strategy_summary']['Sharpe Ratio'])} · "
                f"**Volatility:** {fmt_pct(results['strategy_summary']['Annualized Volatility'])} · "
                f"**Max drawdown:** {fmt_pct(results['strategy_summary']['Max Drawdown'])}"
            )
            if st.button("Generate AI Commentary", type="primary"):
                with st.spinner("Asking Gemini to explain the model output…"):
                    table_text = results["signal_table"].assign(
                        Momentum=lambda d: d["Momentum"].map(fmt_pct),
                        **{"Model Weight": lambda d: d["Model Weight"].map(fmt_pct)},
                    ).to_string(index=False)
                    prompt = llm_commentary.build_portfolio_commentary_prompt(
                        assets=selected_assets,
                        signal_table_text=table_text,
                        metrics=results["strategy_summary"],
                        benchmark_metrics=results["benchmark_summary"],
                        strategy_type="Momentum Ranking Portfolio",
                    )
                    st.session_state["ai_multi"] = llm_commentary.generate_signal_commentary(prompt)
            if "ai_multi" in st.session_state:
                st.markdown("---")
                st.markdown(st.session_state["ai_multi"])

    st.caption(llm_commentary.DISCLAIMER)

# ---- Methodology ---------------------------------------------------------- #
with tab_method:
    st.subheader("Methodology")
    st.markdown(
        """
**Signal definitions**

- *Moving Average Crossover (single asset):* `raw_signal = 1` when the short SMA is
  above the long SMA, else `0`. **Buy** when the position turns on, **Sell** when it
  turns off, **Hold** while invested, **Cash** while flat.
- *Momentum Ranking (multi-asset):* each month, rank assets by trailing momentum,
  select the top *N* (optionally requiring positive momentum), and equal-weight them.
  Others receive 0% weight.

**Look-ahead bias prevention**

- Positions and weights are **shifted by one day** before earning returns: a signal
  observed at today's close is only acted on from the next trading day.

**Transaction costs**

- Charged on traded notional (turnover) at the configured basis-point rate
  (default 10 bps per trade).

**Annualisation**

- A consistent **252 trading days/year** is used for ETFs, stocks, and crypto alike.
  This is a simplifying assumption (crypto trades 365 days/year).

**Limitations**

- Historical backtesting does not guarantee future performance.
- Slippage and liquidity are not fully modelled; costs are estimates.
- yfinance data is suitable for education/research, not institutional trading.
- The system does not connect to brokers or execute trades.
- Gemini commentary may be inaccurate; it is an explanatory layer, not a decision engine.
        """
    )
    st.warning(DISCLAIMER)
