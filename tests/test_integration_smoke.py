from pathlib import Path

import pandas as pd

from sam.backtest import run_backtest
from sam.config import BacktestConfig, PortfolioConfig
from sam.io import write_table
from sam.portfolio import build_portfolio
from sam.stress import run_stress_tests


def test_full_research_smoke(tmp_path: Path, sample_prices: pd.DataFrame) -> None:
    prices_path = tmp_path / "prices.parquet"
    write_table(sample_prices, prices_path)
    prices = pd.read_parquet(prices_path)
    weights = build_portfolio(prices, method="equal_weight")
    result = run_backtest(
        prices,
        config=BacktestConfig(
            portfolio=PortfolioConfig(method="equal_weight", min_history_days=20, lookback_days=60)
        ),
    )
    stress = run_stress_tests(prices, weights)
    assert not weights.empty
    assert not result.returns.empty
    assert not stress.empty
