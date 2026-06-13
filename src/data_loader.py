"""Market data loading and cleaning using yfinance.

The MVP uses yfinance as the only market data source. No API key is required.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_price_data(tickers: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Download daily OHLCV data for each ticker.

    Returns a dict mapping ticker -> cleaned DataFrame and a list of tickers
    that failed to download. A single failing ticker never crashes the load.
    """
    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                # yfinance returns adjusted close in a separate column when
                # auto_adjust is False, which is what the plan asks for.
            )
            if raw is None or raw.empty:
                failed.append(ticker)
                continue
            data[ticker] = clean_price_data(raw)
        except Exception:  # noqa: BLE001 - we never want one ticker to break the app
            failed.append(ticker)

    return data, failed


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a yfinance DataFrame to columns: Open/High/Low/Close/AdjClose/Volume.

    Handles yfinance's MultiIndex columns, prefers adjusted close, and
    forward-fills only small gaps.
    """
    df = df.copy()

    # yfinance can return MultiIndex columns like ('Close', 'SPY'). Flatten to
    # the price-field level so downstream code sees plain column names.
    if isinstance(df.columns, pd.MultiIndex):
        # Level 0 holds the price field (Open, High, ...); level 1 the ticker.
        df.columns = df.columns.get_level_values(0)

    # Normalise column names.
    rename = {c: c.strip().title().replace(" ", "") for c in df.columns}
    df = df.rename(columns=rename)
    # 'AdjClose' is the title-cased form of 'Adj Close'.

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Drop fully empty rows then forward-fill small interior gaps.
    df = df.dropna(how="all")
    df = df.ffill(limit=5)
    df = df.dropna(subset=[c for c in ["Close", "AdjClose"] if c in df.columns], how="all")

    return df


def _select_price_column(df: pd.DataFrame) -> pd.Series:
    """Return adjusted close if available, otherwise close."""
    if "AdjClose" in df.columns:
        series = df["AdjClose"]
    elif "Close" in df.columns:
        series = df["Close"]
    else:  # pragma: no cover - defensive
        raise ValueError("No Close or AdjClose column found.")
    return series


def get_close_prices(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a single price DataFrame (adjusted close), aligned by date.

    Columns are tickers. Assets are aligned on their common dates and small
    gaps are forward-filled. Crypto trades on weekends, so we keep only dates
    where at least one asset has data and forward-fill the rest.
    """
    if not data:
        return pd.DataFrame()

    series_map = {ticker: _select_price_column(df) for ticker, df in data.items()}
    prices = pd.DataFrame(series_map)
    prices = prices.sort_index()

    # Align: forward-fill small holes (e.g. holidays) but require every column
    # to have a real value to be usable, so drop leading rows with NaNs.
    prices = prices.ffill(limit=5)
    prices = prices.dropna(how="any")

    return prices
