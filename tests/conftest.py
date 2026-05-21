from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SAM_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _sam_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests assume paths relative to the SAM repository root."""
    monkeypatch.chdir(SAM_ROOT)


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
