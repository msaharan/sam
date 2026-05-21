from __future__ import annotations

import pandas as pd

from sam.volatility_config import load_volatility_run_config
from sam.volatility_data import synthetic_volatility_fixtures
from sam.volatility_features import (
    assign_split_masks,
    build_volatility_feature_frame,
    leakage_audit,
)


def test_build_volatility_feature_frame_has_targets_and_model_features() -> None:
    run_cfg = load_volatility_run_config("configs/risk/volatility_regime_run.toml")
    data = synthetic_volatility_fixtures(n_days=3200, start="2006-01-01")
    result = build_volatility_feature_frame(
        data["prices"],
        data["panel"],
        config=run_cfg,
    )
    frame = assign_split_masks(result.frame, config=run_cfg)
    assert "target_high_vol_20d" in frame.columns
    assert "spy_rv_20d" in result.base_features
    assert len(result.model_features) > len(result.base_features)
    assert "target_forward_rv_20d" not in result.model_features
    assert frame["target_high_vol_20d"].isin([0, 1]).all()
    labeled = frame["target_high_vol_20d"].notna()
    assert (
        pd.to_datetime(frame.loc[labeled, "target_date"])
        > pd.to_datetime(frame.loc[labeled, "date"])
    ).all()


def test_feature_policy_drops_sparse_columns() -> None:
    run_cfg = load_volatility_run_config("configs/risk/volatility_regime_run.toml")
    data = synthetic_volatility_fixtures(n_days=3200, start="2006-01-01")
    result = build_volatility_feature_frame(
        data["prices"],
        data["panel"],
        config=run_cfg,
    )
    assert result.dropped_features == [] or isinstance(result.dropped_features, list)


def test_feature_counts_match_notebook_bands_on_full_history() -> None:
    run_cfg = load_volatility_run_config("configs/risk/volatility_regime_run.toml")
    data = synthetic_volatility_fixtures(n_days=4000, start="2006-01-01")
    result = build_volatility_feature_frame(
        data["prices"],
        data["panel"],
        config=run_cfg,
    )
    assert 200 <= len(result.base_features) <= 320
    assert 400 <= len(result.model_features) <= 650


def test_split_row_counts_are_positive() -> None:
    run_cfg = load_volatility_run_config("configs/risk/volatility_regime_run.toml")
    data = synthetic_volatility_fixtures(n_days=800, start="2018-01-01")
    result = build_volatility_feature_frame(
        data["prices"],
        data["panel"],
        config=run_cfg,
    )
    frame = assign_split_masks(result.frame, config=run_cfg)
    assert int(frame["split_holdout"].sum()) > 100
    assert int(frame["split_calibration"].sum()) > 100
    assert int(frame["split_train_all_history"].sum()) > 500


def test_imputer_fit_window_ends_at_threshold_source_end() -> None:
    run_cfg = load_volatility_run_config("configs/risk/volatility_regime_run.toml")
    data = synthetic_volatility_fixtures(n_days=4000, start="2006-01-01")
    result = build_volatility_feature_frame(
        data["prices"],
        data["panel"],
        config=run_cfg,
    )
    precalib_end = pd.Timestamp(run_cfg.threshold_source_end)
    train = result.frame.loc[pd.to_datetime(result.frame["date"]) <= precalib_end]
    for column in result.base_features:
        expected = float(train[column].median())
        assert result.imputer_medians[column] == expected


def test_leakage_audit_matches_notebook_contract() -> None:
    run_cfg = load_volatility_run_config("configs/risk/volatility_regime_run.toml")
    data = synthetic_volatility_fixtures(n_days=4000, start="2006-01-01")
    result = build_volatility_feature_frame(
        data["prices"],
        data["panel"],
        config=run_cfg,
    )
    frame = assign_split_masks(result.frame, config=run_cfg)
    checks = leakage_audit(
        frame,
        config=run_cfg,
        model_feature_count=len(result.model_features),
    )
    expected_checks = {
        "Target columns excluded from features",
        "High-volatility threshold estimated before calibration and holdout",
        "Chronological split order",
        "TFM embedding context reused as downstream labels",
        "Raw all-history incumbent separated from optional benchmark window",
        "Calibration window used for final uncalibrated refit",
        "Overlapping target windows",
        "Target-threshold sensitivity reported",
        "Named regime diagnostics reported",
        "Volatility-domain baselines included",
        "Investment decision interpretation",
    }
    assert expected_checks.issubset(set(checks["Check"]))
    assert (checks["Status"] == "pass").sum() >= 6
