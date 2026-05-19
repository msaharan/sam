import pandas as pd

from sam.metrics import calculate_metrics
from sam.portfolio import build_portfolio, portfolio_diagnostics


def test_build_portfolio_weights_sum_to_one(sample_prices: pd.DataFrame) -> None:
    weights = build_portfolio(sample_prices, method="inverse_volatility")
    assert round(weights["weight"].sum(), 10) == 1
    assert (weights["weight"] >= 0).all()


def test_portfolio_diagnostics_include_eligibility(sample_prices: pd.DataFrame) -> None:
    weights = build_portfolio(sample_prices, method="equal_weight")
    diagnostics = portfolio_diagnostics(sample_prices, weights)
    assert {"history_days", "lookback_volatility", "eligible"}.issubset(diagnostics.columns)
    assert diagnostics["eligible"].all()


def test_etf_baseline_only_uses_default_etfs(sample_prices: pd.DataFrame) -> None:
    weights = build_portfolio(sample_prices, method="etf_baseline")
    assert set(weights["symbol"]) == {"SPY"}


def test_calculate_metrics_includes_drawdown_and_sharpe() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, 0.005], name="portfolio_return")
    metrics = calculate_metrics(returns)
    assert "sharpe" in metrics
    assert metrics["max_drawdown"] < 0
