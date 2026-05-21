"""Run configuration for the volatility-regime scoring experiment."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from sam.config import load_toml


class VolatilityRegimeRunConfig(BaseModel):
    target_asset: str = "SPY"
    target_horizon_days: int = Field(default=20, ge=2)
    high_vol_quantile: float = Field(default=0.8, gt=0, lt=1)
    threshold_source_end: str = "2017-12-29"
    data_start: str = "2006-01-01"
    data_end: str = "2026-04-02"
    feature_max_missing_rate: float = Field(default=0.4, ge=0, le=1)
    feature_min_precalibration_observations: int = Field(default=30, ge=1)
    context_end: str = "2009-12-31"
    tuning_end: str = "2017-12-29"
    calibration_end: str = "2019-12-31"
    holdout_start: str = "2020-01-02"
    bootstrap_iterations: int = Field(default=300, ge=10)
    threshold_sensitivity_quantiles: list[float] = Field(
        default_factory=lambda: [0.7, 0.75, 0.8, 0.85, 0.9]
    )
    xgboost_tuning_iterations: int = Field(default=128, ge=1)
    xgboost_cv_splits: int = Field(default=6, ge=1)
    run_xgboost: bool = True
    run_tabpfn: bool = True
    run_tabicl: bool = True
    risk_policy_action_rate: float = Field(default=0.2, gt=0, lt=1)
    risk_policy_high_risk_weight: float = Field(default=0.5, ge=0, le=1)
    risk_policy_cost_bps: float = Field(default=1.0, ge=0)
    fast_mode: bool = False
    fast_bootstrap_iterations: int = Field(default=50, ge=10)
    fast_xgboost_tuning_iterations: int = Field(default=8, ge=1)
    holdout_regimes: dict[str, list[str]] = Field(default_factory=dict)
    fred_series: dict[str, str] = Field(default_factory=dict)


def load_volatility_run_config(path: str | Path) -> VolatilityRegimeRunConfig:
    payload = load_toml(path)
    return VolatilityRegimeRunConfig.model_validate(payload)


__all__ = ["VolatilityRegimeRunConfig", "load_volatility_run_config"]
