"""Vectorized portfolio backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sam.config import DEFAULT_ETFS, BacktestConfig
from sam.metrics import calculate_metrics
from sam.portfolio import _price_panel, _weights_for_returns


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(
    prices: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    benchmark_prices: pd.DataFrame | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    panel = _price_panel(prices)
    if cfg.portfolio.method == "etf_baseline":
        etfs = [symbol for symbol in DEFAULT_ETFS if symbol in panel.columns]
        if not etfs:
            raise ValueError("ETF baseline requires at least one default ETF in price data")
        panel = panel[etfs]
    if cfg.start:
        panel = panel.loc[pd.to_datetime(cfg.start) :]
    if cfg.end:
        panel = panel.loc[: pd.to_datetime(cfg.end)]
    returns = panel.pct_change(fill_method=None).dropna(how="all")
    if returns.empty:
        raise ValueError("not enough price data to backtest")
    rebalance_dates = _rebalance_dates(returns, cfg.portfolio.rebalance)
    portfolio_returns = []
    weight_rows = []
    current_weights = pd.Series(0.0, index=returns.columns)
    cost_rate = (cfg.portfolio.transaction_cost_bps + cfg.slippage_bps) / 10_000
    for date, row in returns.iterrows():
        if date in rebalance_dates:
            history = panel.loc[:date].iloc[:-1].tail(cfg.portfolio.lookback_days + 1)
            history_returns = history.pct_change(fill_method=None).dropna(how="any")
            if len(history_returns) >= cfg.portfolio.min_history_days:
                target = _weights_for_returns(
                    history_returns,
                    cfg.portfolio.method,
                    cfg.portfolio.max_weight,
                ).reindex(returns.columns, fill_value=0.0)
            else:
                eligible = row.dropna().index
                target = pd.Series(0.0, index=returns.columns)
                if len(eligible):
                    target.loc[eligible] = 1 / len(eligible)
            turnover = float((target - current_weights).abs().sum())
            current_weights = target
            cost = turnover * cost_rate
        else:
            turnover = 0.0
            cost = 0.0
        daily_return = float((current_weights * row.fillna(0.0)).sum() - cost)
        portfolio_returns.append(
            {
                "date": date,
                "portfolio_return": daily_return,
                "turnover": turnover,
                "cost": cost,
            }
        )
        for symbol, weight in current_weights.items():
            weight_rows.append({"date": date, "symbol": symbol, "weight": float(weight)})
    returns_frame = pd.DataFrame(portfolio_returns)
    returns_frame["equity"] = (1.0 + returns_frame["portfolio_return"]).cumprod()
    benchmark_returns = (
        _benchmark_returns(benchmark_prices, returns_frame["date"])
        if benchmark_prices is not None
        else None
    )
    if benchmark_returns is not None:
        returns_frame["benchmark_return"] = benchmark_returns.to_numpy()
        returns_frame["active_return"] = (
            returns_frame["portfolio_return"] - returns_frame["benchmark_return"]
        )
    metrics = calculate_metrics(
        returns_frame.set_index("date")["portfolio_return"],
        benchmark=benchmark_returns,
        risk_free_rate=cfg.portfolio.risk_free_rate,
    )
    metrics["average_turnover"] = float(returns_frame["turnover"].mean())
    return BacktestResult(returns=returns_frame, weights=pd.DataFrame(weight_rows), metrics=metrics)


def compare_backtests(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = [{"name": name, **result.metrics} for name, result in results.items()]
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def _rebalance_dates(returns: pd.DataFrame, frequency: str) -> set[pd.Timestamp]:
    pandas_frequency = {"M": "ME", "Q": "QE", "W": "W"}.get(frequency, frequency)
    grouped = returns.groupby(pd.Grouper(freq=pandas_frequency))
    return {group.index[0] for _, group in grouped if not group.empty}


def _benchmark_returns(benchmark_prices: pd.DataFrame, dates: pd.Series) -> pd.Series:
    panel = _price_panel(benchmark_prices)
    series = panel.iloc[:, 0].pct_change(fill_method=None)
    return series.reindex(pd.to_datetime(dates)).rename("benchmark")
