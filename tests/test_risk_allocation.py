from __future__ import annotations

import pandas as pd

from sam.allocation import allocation_turnover, build_score_allocation, scores_to_weights
from sam.risk import alert_operating_points, build_volatility_regime_frame


def test_volatility_regime_frame_uses_future_target_dates(sample_prices: pd.DataFrame) -> None:
    result = build_volatility_regime_frame(
        sample_prices,
        symbol="SPY",
        horizon_days=20,
        threshold_quantile=0.8,
        threshold_end="2020-12-31",
        windows=(5, 20),
    )
    frame = result.frame
    assert {"realized_vol_20d", "parkinson_vol_20d", "garman_klass_vol_20d"}.issubset(
        frame.columns
    )
    assert result.threshold > 0
    assert frame["target_high_vol"].isin([0, 1]).all()
    assert (pd.to_datetime(frame["target_date"]) > pd.to_datetime(frame["date"])).all()
    assert (pd.to_datetime(frame["feature_available_date"]) <= pd.to_datetime(frame["date"])).all()


def test_volatility_regime_frame_can_keep_latest_unlabeled_row(
    sample_prices: pd.DataFrame,
) -> None:
    result = build_volatility_regime_frame(
        sample_prices,
        symbol="SPY",
        horizon_days=20,
        threshold_quantile=0.8,
        threshold_end="2020-12-31",
        windows=(5, 20),
        include_unlabeled=True,
    )
    latest = result.frame.tail(1).iloc[0]
    assert pd.Timestamp(latest["date"]) == pd.to_datetime(sample_prices["date"]).max()
    assert pd.isna(latest["future_realized_vol_20d"])
    assert pd.isna(latest["target_high_vol"])
    assert latest["feature_available_date"] <= latest["date"]


def test_alert_operating_points_prioritize_high_scores() -> None:
    scores = pd.DataFrame(
        {
            "score": [0.9, 0.8, 0.2, 0.1],
            "target_high_vol": [1, 1, 0, 0],
        }
    )
    result = alert_operating_points(scores, score_columns=["score"], recalls=[0.5, 1.0])
    assert result.loc[result["target_recall"] == 0.5, "alerts"].iloc[0] == 1
    assert result.loc[result["target_recall"] == 1.0, "precision"].iloc[0] == 1.0


def test_scores_to_weights_respects_top_k_and_max_weight() -> None:
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 5),
            "symbol": ["A", "B", "C", "D", "E"],
            "prediction": [5, 4, 3, 2, 1],
        }
    )
    weights = scores_to_weights(scores, top_k=3, max_weight=0.5)
    assert set(weights["symbol"]) == {"A", "B", "C"}
    assert round(weights["weight"].sum(), 10) == 1.0
    assert weights["weight"].max() <= 0.5


def test_allocation_turnover_includes_initial_entry_and_costs() -> None:
    weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]),
            "symbol": ["A", "B", "A", "C"],
            "weight": [0.5, 0.5, 0.4, 0.6],
        }
    )
    turnover = allocation_turnover(weights, transaction_cost_bps=10, slippage_bps=5)
    assert turnover["turnover"].iloc[0] == 1.0
    assert turnover["turnover"].iloc[1] == 1.2
    assert turnover["cost"].sum() > 0
    assert (turnover["gross_exposure"] == 1.0).all()


def test_allocation_turnover_flags_turnover_limit_breaches() -> None:
    weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "symbol": ["A", "B"],
            "weight": [1.0, 1.0],
        }
    )
    turnover = allocation_turnover(weights, turnover_limit=0.5)
    assert turnover["turnover_breach"].tolist() == [True, True]


def test_build_score_allocation_returns_weights_and_turnover() -> None:
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 4 + ["2020-02-29"] * 4),
            "symbol": ["A", "B", "C", "D"] * 2,
            "prediction": [4, 3, 2, 1, 1, 4, 3, 2],
        }
    )
    result = build_score_allocation(
        scores,
        top_k=2,
        max_weight=0.6,
        transaction_cost_bps=5,
        slippage_bps=1,
    )
    assert not result.weights.empty
    assert not result.turnover.empty
    assert result.turnover["cost"].sum() > 0
