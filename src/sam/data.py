"""Market data ingestion and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sam.config import Market, normalize_ticker
from sam.io import write_partitioned_table, write_table

CANONICAL_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]


def fetch_prices(
    symbols: list[str],
    *,
    start: str,
    end: str | None = None,
    market: Market = "us",
    out: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch adjusted OHLCV data from yfinance and optionally write long-form Parquet."""

    import yfinance as yf

    normalized = [normalize_ticker(symbol, market) for symbol in symbols]
    raw = yf.download(
        normalized,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    frame = _yfinance_to_long(raw, normalized)
    if out is not None:
        write_table(frame, out)
    return frame


def write_price_dataset(
    prices: pd.DataFrame,
    root: str | Path,
    *,
    table: str = "raw_ohlcv",
) -> Path:
    """Write canonical prices as a partitioned Parquet dataset for DuckDB research."""

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year.astype("int16")
    frame["date"] = frame["date"].dt.date
    return write_partitioned_table(frame, Path(root) / table, partition_cols=["symbol", "year"])


def write_return_dataset(
    returns: pd.DataFrame,
    root: str | Path,
    *,
    table: str = "returns",
) -> Path:
    frame = returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year.astype("int16")
    frame["date"] = frame["date"].dt.date
    return write_partitioned_table(frame, Path(root) / table, partition_cols=["symbol", "year"])


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "adj_close"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"price data missing columns: {missing}")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.sort_values(["symbol", "date"])
    duplicate_keys = (
        frame.duplicated(["symbol", "date"], keep=False)
        .groupby(frame["symbol"])
        .sum()
        .rename("duplicate_date_rows")
    )
    diagnostics = (
        frame.groupby("symbol", as_index=False)
        .agg(
            rows=("adj_close", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            missing_adj_close=("adj_close", lambda item: int(item.isna().sum())),
            non_positive_adj_close=("adj_close", lambda item: int((item <= 0).sum())),
        )
        .sort_values("symbol")
    )
    diagnostics = diagnostics.merge(
        duplicate_keys.reset_index(),
        on="symbol",
        how="left",
    )
    diagnostics["duplicate_date_rows"] = diagnostics["duplicate_date_rows"].fillna(0).astype(int)
    diagnostics["status"] = diagnostics.apply(
        lambda row: "fail"
        if row["missing_adj_close"] > 0
        or row["non_positive_adj_close"] > 0
        or row["duplicate_date_rows"] > 0
        else "pass",
        axis=1,
    )
    return diagnostics


def price_panel(prices: pd.DataFrame, value: str = "adj_close") -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    panel = frame.pivot(index="date", columns="symbol", values=value)
    return panel.sort_index()


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    panel = price_panel(prices)
    return panel.pct_change(fill_method=None).dropna(how="all")


def long_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    returns = returns_from_prices(prices)
    return (
        returns.reset_index()
        .melt(id_vars="date", var_name="symbol", value_name="return")
        .dropna(subset=["return"])
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def _yfinance_to_long(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    if isinstance(raw.columns, pd.MultiIndex):
        if raw.columns.names[0] in {"Price", None}:
            stacked = raw.stack(level=1, future_stack=True).reset_index()
        else:
            stacked = raw.stack(level=0, future_stack=True).reset_index()
        stacked = stacked.rename(columns={"Date": "date", "Ticker": "symbol", "level_1": "symbol"})
    else:
        stacked = raw.reset_index()
        stacked["symbol"] = symbols[0]
        stacked = stacked.rename(columns={"Date": "date", "index": "date"})

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    stacked = stacked.rename(columns=rename)
    for column in CANONICAL_COLUMNS:
        if column not in stacked.columns:
            stacked[column] = pd.NA
    frame = stacked[CANONICAL_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["symbol"] = frame["symbol"].astype(str)
    numeric = ["open", "high", "low", "close", "adj_close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return (
        frame.dropna(subset=["date", "symbol"])
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
