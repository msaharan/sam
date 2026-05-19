from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=420, freq="B")
    symbols = ["AAA", "BBB", "CCC", "SPY"]
    rows = []
    for idx, symbol in enumerate(symbols):
        drift = 0.0002 + idx * 0.00005
        vol = 0.01 + idx * 0.001
        returns = drift + np.sin(np.arange(len(dates)) / 20 + idx) * vol / 5
        prices = 100 * (1 + pd.Series(returns)).cumprod()
        for date, price in zip(dates, prices, strict=True):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)
