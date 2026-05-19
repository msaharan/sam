"""Historical and parametric portfolio stress testing."""

from __future__ import annotations

import pandas as pd

from sam.data import price_panel

DEFAULT_SCENARIOS = [
    {"name": "covid_crash", "kind": "historical", "start": "2020-02-19", "end": "2020-03-23"},
    {
        "name": "inflation_rate_shock_2022",
        "kind": "historical",
        "start": "2022-01-03",
        "end": "2022-10-14",
    },
    {"name": "broad_equity_selloff", "kind": "parametric", "equity_shock": -0.20},
    {
        "name": "volatility_spike",
        "kind": "parametric",
        "equity_shock": -0.12,
        "vol_multiplier": 2.0,
    },
    {
        "name": "liquidity_cost_shock",
        "kind": "parametric",
        "equity_shock": -0.08,
        "cost_shock": -0.01,
    },
]


def run_stress_tests(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    scenarios: list[dict] | None = None,
    benchmark_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    panel = price_panel(prices)
    benchmark_panel = price_panel(benchmark_prices) if benchmark_prices is not None else None
    weight_series = _latest_weights(weights)
    scenarios = scenarios or DEFAULT_SCENARIOS
    rows = []
    for scenario in scenarios:
        if scenario["kind"] == "historical":
            row = _historical_stress(panel, weight_series, scenario, benchmark_panel)
        else:
            row = _parametric_stress(weight_series, scenario)
        rows.append(row)
    return pd.DataFrame(rows)


def _historical_stress(
    panel: pd.DataFrame,
    weights: pd.Series,
    scenario: dict,
    benchmark_panel: pd.DataFrame | None,
) -> dict:
    window = panel.loc[pd.to_datetime(scenario["start"]) : pd.to_datetime(scenario["end"])]
    aligned = window.dropna(axis=1, how="any")
    aligned_weights = weights.reindex(aligned.columns).fillna(0.0)
    benchmark_return = _historical_benchmark_return(benchmark_panel, scenario)
    if aligned.empty or aligned_weights.sum() <= 0:
        loss = float("nan")
        recovery_days = float("nan")
        worst_symbol = ""
        worst_contribution = float("nan")
    else:
        aligned_weights = aligned_weights / aligned_weights.sum()
        asset_returns = aligned.iloc[-1] / aligned.iloc[0] - 1.0
        contributions = aligned_weights * asset_returns
        loss = float(contributions.sum())
        daily_returns = (
            aligned.pct_change(fill_method=None)
            .fillna(0)
            .mul(aligned_weights, axis=1)
            .sum(axis=1)
        )
        path = (1 + daily_returns).cumprod()
        recovery_days = _recovery_days(path)
        worst_symbol = str(contributions.idxmin())
        worst_contribution = float(contributions.min())
    return {
        "scenario": scenario["name"],
        "kind": "historical",
        "portfolio_return": loss,
        "benchmark_return": benchmark_return,
        "relative_return": loss - benchmark_return
        if pd.notna(loss) and pd.notna(benchmark_return)
        else float("nan"),
        "recovery_days": recovery_days,
        "worst_contributor": worst_symbol,
        "worst_contribution": worst_contribution,
    }


def _parametric_stress(weights: pd.Series, scenario: dict) -> dict:
    contributions = pd.Series(
        {
            symbol: weight * _asset_shock(symbol, scenario)
            for symbol, weight in weights.items()
        }
    )
    if contributions.empty:
        portfolio_return = float("nan")
        worst_contributor = ""
        worst_contribution = float("nan")
    else:
        portfolio_return = float(contributions.sum())
        worst_contributor = str(contributions.idxmin())
        worst_contribution = float(contributions.min())
    return {
        "scenario": scenario["name"],
        "kind": "parametric",
        "portfolio_return": portfolio_return,
        "benchmark_return": float("nan"),
        "relative_return": float("nan"),
        "recovery_days": float("nan"),
        "worst_contributor": worst_contributor,
        "worst_contribution": worst_contribution,
    }


def _recovery_days(path: pd.Series) -> float:
    start = float(path.iloc[0])
    trough_date = path.idxmin()
    recovered = path.loc[trough_date:][path.loc[trough_date:] >= start]
    if recovered.empty:
        return float("nan")
    return float((recovered.index[0] - path.index[0]).days)


def _latest_weights(weights: pd.DataFrame) -> pd.Series:
    frame = weights.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.loc[frame["date"] == frame["date"].max()]
    return frame.groupby("symbol")["weight"].sum().astype(float)


def _historical_benchmark_return(benchmark_panel: pd.DataFrame | None, scenario: dict) -> float:
    if benchmark_panel is None or benchmark_panel.empty:
        return float("nan")
    window = benchmark_panel.loc[
        pd.to_datetime(scenario["start"]) : pd.to_datetime(scenario["end"])
    ]
    series = window.iloc[:, 0].dropna()
    if len(series) < 2:
        return float("nan")
    return float(series.iloc[-1] / series.iloc[0] - 1.0)


def _asset_shock(symbol: str, scenario: dict) -> float:
    equity_shock = float(scenario.get("equity_shock", 0.0))
    vol_multiplier = float(scenario.get("vol_multiplier", 1.0))
    return equity_shock * vol_multiplier + float(scenario.get("cost_shock", 0.0))
