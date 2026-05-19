import pandas as pd

from sam.backtest import run_backtest
from sam.config import BacktestConfig, PortfolioConfig
from sam.stress import run_stress_tests


def test_backtest_writes_returns_and_metrics(sample_prices: pd.DataFrame) -> None:
    config = BacktestConfig(
        portfolio=PortfolioConfig(method="equal_weight", min_history_days=20, lookback_days=60)
    )
    result = run_backtest(sample_prices, config=config)
    assert not result.returns.empty
    assert result.returns["equity"].iloc[-1] > 0
    assert "sharpe" in result.metrics


def test_backtest_portfolio_translation_costs_turnover_and_benchmark(
    sample_prices: pd.DataFrame,
) -> None:
    config = BacktestConfig(
        portfolio=PortfolioConfig(
            method="equal_weight",
            min_history_days=20,
            lookback_days=60,
            rebalance="M",
            transaction_cost_bps=10,
            max_weight=0.5,
        ),
        slippage_bps=5,
    )
    benchmark = sample_prices[sample_prices["symbol"] == "SPY"]
    result = run_backtest(sample_prices, config=config, benchmark_prices=benchmark)
    rebalanced = result.returns[result.returns["turnover"] > 0]

    assert not rebalanced.empty
    assert result.returns["cost"].sum() > 0
    assert "average_turnover" in result.metrics
    assert "benchmark_return" in result.returns.columns
    assert "active_return" in result.returns.columns
    assert result.weights.groupby("date")["weight"].max().max() <= 0.5


def test_backtest_rebalance_does_not_use_same_day_return() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-02-03")
    levels = {
        "AAA": [100.0] * (len(dates) - 1) + [200.0],
        "BBB": [100.0 + idx for idx in range(len(dates) - 1)] + [100.0 + len(dates) - 2],
    }
    rows = []
    for symbol, prices in levels.items():
        for date, price in zip(dates, prices, strict=True):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000,
                }
            )
    result = run_backtest(
        pd.DataFrame(rows),
        config=BacktestConfig(
                portfolio=PortfolioConfig(
                    method="momentum_tilt",
                    min_history_days=20,
                    lookback_days=20,
                rebalance="M",
                max_weight=1.0,
            )
        ),
    )
    feb_weights = result.weights[result.weights["date"] == pd.Timestamp("2020-02-03")]
    weights = feb_weights.set_index("symbol")["weight"]
    assert weights["BBB"] > weights["AAA"]


def test_stress_tests_include_default_scenarios(sample_prices: pd.DataFrame) -> None:
    weights = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "weight": [0.4, 0.3, 0.3]})
    benchmark = sample_prices[sample_prices["symbol"] == "SPY"]
    result = run_stress_tests(sample_prices, weights, benchmark_prices=benchmark)
    assert {"covid_crash", "broad_equity_selloff"}.issubset(set(result["scenario"]))
    assert {"relative_return", "worst_contribution"}.issubset(result.columns)
