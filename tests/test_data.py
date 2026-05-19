from __future__ import annotations

import sys
import types

import pandas as pd

from sam.data import fetch_prices, long_returns_from_prices, validate_prices, write_price_dataset
from sam.io import query_parquet


def test_validate_prices_flags_bad_values(sample_prices: pd.DataFrame) -> None:
    broken = sample_prices.copy()
    broken.loc[0, "adj_close"] = -1
    diagnostics = validate_prices(broken)
    assert diagnostics.loc[diagnostics["symbol"] == "AAA", "status"].iloc[0] == "fail"


def test_validate_prices_flags_duplicate_symbol_dates(sample_prices: pd.DataFrame) -> None:
    duplicated = pd.concat([sample_prices, sample_prices.iloc[[0]]], ignore_index=True)
    diagnostics = validate_prices(duplicated)
    aaa = diagnostics.loc[diagnostics["symbol"] == "AAA"].iloc[0]
    assert aaa["duplicate_date_rows"] == 2
    assert aaa["status"] == "fail"


def test_fetch_prices_uses_yfinance_download(monkeypatch) -> None:
    dates = pd.date_range("2020-01-01", periods=2)
    raw = pd.DataFrame(
        {
            "Open": [1.0, 1.1],
            "High": [1.2, 1.3],
            "Low": [0.9, 1.0],
            "Close": [1.1, 1.2],
            "Adj Close": [1.1, 1.2],
            "Volume": [100, 200],
        },
        index=dates,
    )
    fake = types.SimpleNamespace(download=lambda *args, **kwargs: raw)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    frame = fetch_prices(["AAA"], start="2020-01-01")
    assert list(frame["symbol"].unique()) == ["AAA"]
    assert frame["adj_close"].iloc[-1] == 1.2


def test_write_price_dataset_is_queryable_with_duckdb(
    tmp_path, sample_prices: pd.DataFrame
) -> None:
    dataset = write_price_dataset(sample_prices, tmp_path)
    result = query_parquet(
        "SELECT symbol, count(*) AS rows FROM parquet_data GROUP BY symbol",
        dataset,
    )
    assert set(result["symbol"]) == {"AAA", "BBB", "CCC", "SPY"}

    returns = long_returns_from_prices(sample_prices)
    assert {"date", "symbol", "return"}.issubset(returns.columns)
