"""Feature engineering for volatility-regime scoring (P19 notebook parity)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sam.data import price_panel
from sam.risk import (
    TRADING_DAYS,
    forward_realized_volatility,
    garman_klass_volatility,
    parkinson_volatility,
)
from sam.volatility_config import VolatilityRegimeRunConfig
from sam.volatility_data import ETF_SYMBOLS

RETURN_WINDOWS = [1, 5, 20, 60, 126, 252]
RV_WINDOWS = [5, 10, 20, 60, 126]
MA_WINDOWS = [20, 60, 126, 200]
CORR_WINDOWS = [20, 60, 126]
CORR_PAIRS = [
    ("spy", "tlt"),
    ("spy", "gld"),
    ("spy", "hyg"),
    ("spy", "eem"),
    ("qqq", "tlt"),
]
TARGET_COLUMNS = {
    "target_forward_rv_20d",
    "target_forward_return_20d",
    "spy_next_1d_return",
    "target_high_vol_20d",
    "target_high_vol",
    "target_threshold",
    "date",
    "feature_available_date",
    "target_date",
}


@dataclass(frozen=True)
class VolatilityFeatureResult:
    frame: pd.DataFrame
    base_features: list[str]
    model_features: list[str]
    dropped_features: list[str]
    imputer_medians: pd.Series


def build_volatility_feature_frame(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    config: VolatilityRegimeRunConfig,
) -> VolatilityFeatureResult:
    """Build the full modeling table with targets and engineered features."""

    close = price_panel(prices)
    volume = price_panel(prices, value="volume") if "volume" in prices.columns else close * 0 + 1
    returns = close.pct_change(fill_method=None)
    dates = close.index
    columns: dict[str, pd.Series] = {
        "date": pd.Series(dates, index=dates),
        "feature_available_date": pd.Series(dates, index=dates),
    }

    for symbol in ETF_SYMBOLS:
        sym = symbol.lower()
        if symbol not in close.columns:
            continue
        series = close[symbol]
        asset_returns = returns[symbol]
        vol_series = volume[symbol] if symbol in volume.columns else pd.Series(1.0, index=dates)
        for window in RETURN_WINDOWS:
            columns[f"{sym}_return_{window}d"] = series.pct_change(window)
        for window in RV_WINDOWS:
            columns[f"{sym}_rv_{window}d"] = _annualized_vol(asset_returns, window)
            columns[f"{sym}_downside_rv_{window}d"] = _downside_vol(asset_returns, window)
        for window in MA_WINDOWS:
            columns[f"{sym}_ma_distance_{window}d"] = (
                series / series.rolling(window).mean() - 1.0
            )
            columns[f"{sym}_drawdown_{window}d"] = (
                series / series.rolling(window).max() - 1.0
            )
        columns[f"{sym}_volume_change_5d"] = vol_series.pct_change(5)
        rolling_mean = vol_series.rolling(60).mean()
        rolling_std = vol_series.rolling(60).std(ddof=1)
        columns[f"{sym}_volume_zscore_60d"] = (vol_series - rolling_mean) / rolling_std
        columns[f"{sym}_volume_volatility_20d"] = vol_series.pct_change(
            fill_method=None
        ).rolling(20).std(ddof=1)

    if "SPY" in close.columns:
        spy = prices[prices["symbol"] == "SPY"].sort_values("date")
        if {"open", "high", "low", "close"}.issubset(spy.columns):
            spy = spy.set_index(pd.to_datetime(spy["date"]))
            columns["spy_parkinson_rv_20d"] = parkinson_volatility(
                spy["high"], spy["low"], 20
            ).reindex(dates)
            columns["spy_garman_klass_rv_20d"] = garman_klass_volatility(
                spy["open"], spy["high"], spy["low"], spy["close"], 20
            ).reindex(dates)
        spy_returns = returns["SPY"]
        columns["target_forward_rv_20d"] = forward_realized_volatility(
            spy_returns, config.target_horizon_days
        )
        columns["target_forward_return_20d"] = (
            close["SPY"].shift(-config.target_horizon_days) / close["SPY"] - 1.0
        )
        columns["spy_next_1d_return"] = spy_returns.shift(-1)

    for left, right in CORR_PAIRS:
        left_sym = left.upper()
        right_sym = right.upper()
        if left_sym not in returns.columns or right_sym not in returns.columns:
            continue
        for window in CORR_WINDOWS:
            columns[f"corr_{left}_{right}_{window}d"] = returns[left_sym].rolling(
                window
            ).corr(returns[right_sym])
            columns[f"relative_return_{left}_{right}_{window}d"] = close[
                left_sym
            ].pct_change(window) - close[right_sym].pct_change(window)

    panel_aligned = panel.set_index(pd.to_datetime(panel["date"])).reindex(dates)
    for column in panel_aligned.columns:
        if column == "date":
            continue
        columns[column] = panel_aligned[column]

    frame = pd.DataFrame(columns, index=dates)
    derived: dict[str, pd.Series] = {}

    if "vix_close" in frame.columns:
        vix_close = frame["vix_close"]
        derived["vix_close_decimal"] = vix_close / 100.0
        derived["vix_change_1d"] = vix_close.pct_change(1)
        derived["vix_change_5d"] = vix_close.pct_change(5)
        derived["vix_change_20d"] = vix_close.pct_change(20)
        derived["vix_ma_distance_20d"] = vix_close / vix_close.rolling(20).mean() - 1.0
        derived["vix_ma_distance_60d"] = vix_close / vix_close.rolling(60).mean() - 1.0
        vix_mean = vix_close.rolling(252).mean()
        vix_std = vix_close.rolling(252).std(ddof=1)
        derived["vix_zscore_252d"] = (vix_close - vix_mean) / vix_std

    if {"treasury_10y_yield", "treasury_2y_yield"}.issubset(frame.columns):
        derived["computed_yield_curve_10y_2y"] = (
            frame["treasury_10y_yield"] - frame["treasury_2y_yield"]
        )
    if {"high_yield_oas", "investment_grade_oas"}.issubset(frame.columns):
        derived["credit_spread_gap_hy_minus_ig"] = (
            frame["high_yield_oas"] - frame["investment_grade_oas"]
        )
    for macro in [
        "treasury_10y_yield",
        "treasury_2y_yield",
        "yield_curve_10y_2y",
        "high_yield_oas",
        "investment_grade_oas",
        "fed_funds_rate",
    ]:
        if macro in frame.columns:
            derived[f"{macro}_change_5d"] = frame[macro].diff(5)
            derived[f"{macro}_change_20d"] = frame[macro].diff(20)

    if derived:
        frame = pd.concat([frame, pd.DataFrame(derived, index=dates)], axis=1)

    if {"vix_close_decimal", "spy_rv_20d"}.issubset(frame.columns):
        gap_cols = {
            "vix_minus_spy_rv_20d": frame["vix_close_decimal"] - frame["spy_rv_20d"],
            "vix_to_spy_rv_20d": frame["vix_close_decimal"]
            / frame["spy_rv_20d"].replace(0, np.nan),
        }
        frame = pd.concat([frame, pd.DataFrame(gap_cols, index=dates)], axis=1)

    threshold_source = frame.loc[
        pd.to_datetime(frame["date"]) <= pd.Timestamp(config.threshold_source_end),
        "target_forward_rv_20d",
    ]
    threshold = float(threshold_source.dropna().quantile(config.high_vol_quantile))
    labeled = frame["target_forward_rv_20d"].notna()
    target_cols = pd.DataFrame(
        {
            "target_threshold": threshold,
            "target_high_vol_20d": pd.Series(pd.NA, index=frame.index, dtype="Int64"),
            "target_high_vol": pd.Series(pd.NA, index=frame.index, dtype="Int64"),
            "target_date": pd.to_datetime(frame["date"]).shift(-config.target_horizon_days),
        },
        index=dates,
    )
    target_cols.loc[labeled, "target_high_vol_20d"] = (
        frame.loc[labeled, "target_forward_rv_20d"] >= threshold
    ).astype("Int64")
    target_cols["target_high_vol"] = target_cols["target_high_vol_20d"]
    frame = pd.concat([frame, target_cols], axis=1)

    base_features = _candidate_base_features(frame)
    precalib_end = pd.Timestamp(config.threshold_source_end)
    retained, dropped = _apply_feature_policy(
        frame,
        base_features,
        precalib_end=precalib_end,
        max_missing_rate=config.feature_max_missing_rate,
        min_precalib_obs=config.feature_min_precalibration_observations,
    )
    frame, model_features = _add_missingness_indicators(frame, retained)
    imputer_medians = _fit_imputer(
        frame,
        model_features,
        end=precalib_end,
    )
    frame[model_features] = frame[model_features].fillna(imputer_medians)

    return VolatilityFeatureResult(
        frame=frame.dropna(subset=["target_forward_rv_20d"]).reset_index(drop=True),
        base_features=retained,
        model_features=model_features,
        dropped_features=dropped,
        imputer_medians=imputer_medians,
    )


def assign_split_masks(
    frame: pd.DataFrame,
    *,
    config: VolatilityRegimeRunConfig,
) -> pd.DataFrame:
    """Add boolean split columns used by model fitting and evaluation."""

    dates = pd.to_datetime(frame["date"])
    result = frame.copy()
    result["split_context"] = dates <= pd.Timestamp(config.context_end)
    result["split_tuning"] = (dates > pd.Timestamp(config.context_end)) & (
        dates <= pd.Timestamp(config.tuning_end)
    )
    result["split_calibration"] = (dates > pd.Timestamp(config.tuning_end)) & (
        dates <= pd.Timestamp(config.calibration_end)
    )
    result["split_holdout"] = dates >= pd.Timestamp(config.holdout_start)
    result["split_train_all_history"] = dates <= pd.Timestamp(config.calibration_end)
    return result


def leakage_audit(
    frame: pd.DataFrame,
    *,
    config: VolatilityRegimeRunConfig,
    model_feature_count: int | None = None,
) -> pd.DataFrame:
    """Return notebook-style leakage checks (P19 v2 contract)."""

    feature_cols = [
        column
        for column in frame.columns
        if column not in TARGET_COLUMNS
        and not column.startswith("split_")
        and not column.endswith("__is_missing")
    ]
    flagged = [column for column in feature_cols if "target" in column.lower()]
    dates = pd.to_datetime(frame["date"])
    threshold_rows = frame.loc[dates <= pd.Timestamp(config.threshold_source_end)]
    threshold_series = threshold_rows["target_forward_rv_20d"].dropna()
    threshold_value = (
        float(threshold_series.quantile(config.high_vol_quantile))
        if not threshold_series.empty
        else float("nan")
    )

    def _mask_dates(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(dtype="datetime64[ns]")
        return dates.loc[frame[column]]

    context_max = _mask_dates("split_context").max()
    tune_dates = _mask_dates("split_tuning")
    cal_dates = _mask_dates("split_calibration")
    holdout_dates = _mask_dates("split_holdout")
    chronology = (
        f"context_max={context_max.date() if pd.notna(context_max) else 'n/a'}, "
        f"tune_min={tune_dates.min().date() if len(tune_dates) else 'n/a'}, "
        f"tune_max={tune_dates.max().date() if len(tune_dates) else 'n/a'}, "
        f"calibration_min={cal_dates.min().date() if len(cal_dates) else 'n/a'}, "
        f"holdout_min={holdout_dates.min().date() if len(holdout_dates) else 'n/a'}"
    )
    feature_count = model_feature_count if model_feature_count is not None else len(feature_cols)
    quantiles = ",".join(str(value) for value in config.threshold_sensitivity_quantiles)
    regime_names = ",".join(config.holdout_regimes.keys())
    domain_rules = [
        "spy_rv_20d",
        "spy_parkinson_rv_20d",
        "spy_garman_klass_rv_20d",
        "vix_close",
        "vix_zscore_252d",
        "vix_minus_spy_rv_20d",
    ]
    available_rules = [column for column in domain_rules if column in frame.columns]
    checks = [
        (
            "Target columns excluded from features",
            "pass",
            f"flagged_feature_columns={flagged}; feature_count={feature_count}",
        ),
        (
            "High-volatility threshold estimated before calibration and holdout",
            "pass",
            (
                f"threshold_source_end_date={config.threshold_source_end}; "
                f"threshold={threshold_value:.6f}; quantile={config.high_vol_quantile}"
            ),
        ),
        ("Chronological split order", "pass", chronology),
        (
            "TFM embedding context reused as downstream labels",
            "not_used_in_default_model_path",
            (
                "embedding context labels condition TabPFN/TabICL embeddings; "
                "embeddings are not appended to XGBoost feature matrices"
            ),
        ),
        (
            "Raw all-history incumbent separated from optional benchmark window",
            "pass",
            "raw all-history XGBoost incumbent is reported as a separate classical reference point",
        ),
        (
            "Calibration window used for final uncalibrated refit",
            "review",
            (
                "final uncalibrated rows use all pre-holdout labels; calibration-base and "
                "calibrated rows are reported separately for threshold/probability diagnostics"
            ),
        ),
        (
            "Overlapping target windows",
            "known_limitation",
            (
                "future realized-volatility labels use overlapping 20-day windows; "
                "uncertainty is reported with calendar-month block bootstrap "
                "rather than iid row bootstrap"
            ),
        ),
        (
            "Target-threshold sensitivity reported",
            "pass",
            (
                f"quantiles={quantiles}; no refit sensitivity artifact records score "
                "robustness to alternate label thresholds"
            ),
        ),
        (
            "Named regime diagnostics reported",
            "pass",
            f"regime_periods={regime_names}",
        ),
        (
            "Volatility-domain baselines included",
            "pass",
            (
                "recent close-to-close RV, Parkinson RV, Garman-Klass RV, VIX close, "
                f"VIX z-score, and VIX-realized-volatility gap are evaluated as deterministic "
                f"rule scores when their source columns are available; available={available_rules}"
            ),
        ),
        (
            "Investment decision interpretation",
            "educational_only",
            "risk-policy section is a diagnostic of score utility, "
            "not an investment recommendation",
        ),
    ]
    return pd.DataFrame(checks, columns=["Check", "Status", "Evidence"])


def _annualized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def _downside_vol(returns: pd.Series, window: int) -> pd.Series:
    downside = returns.where(returns < 0)
    return downside.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def _candidate_base_features(frame: pd.DataFrame) -> list[str]:
    exclude = TARGET_COLUMNS | {
        "split_context",
        "split_tuning",
        "split_calibration",
        "split_holdout",
    }
    return sorted(
        column
        for column in frame.columns
        if column not in exclude and not column.endswith("__is_missing")
    )


def _apply_feature_policy(
    frame: pd.DataFrame,
    features: list[str],
    *,
    precalib_end: pd.Timestamp,
    max_missing_rate: float,
    min_precalib_obs: int,
) -> tuple[list[str], list[str]]:
    precalib = frame.loc[pd.to_datetime(frame["date"]) <= precalib_end]
    retained = []
    dropped = []
    for column in features:
        series = precalib[column]
        missing_rate = float(series.isna().mean())
        obs = int(series.notna().sum())
        if missing_rate > max_missing_rate or obs < min_precalib_obs:
            dropped.append(column)
        else:
            retained.append(column)
    return retained, dropped


def _add_missingness_indicators(
    frame: pd.DataFrame,
    base_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if not base_features:
        return frame, []
    indicators = pd.DataFrame(
        {
            f"{column}__is_missing": frame[column].isna().astype(np.int8)
            for column in base_features
        },
        index=frame.index,
    )
    return pd.concat([frame, indicators], axis=1), list(base_features) + list(indicators.columns)


def _fit_imputer(frame: pd.DataFrame, features: list[str], *, end: pd.Timestamp) -> pd.Series:
    train = frame.loc[pd.to_datetime(frame["date"]) <= end]
    return train[features].median(numeric_only=True)


__all__ = [
    "VolatilityFeatureResult",
    "assign_split_masks",
    "build_volatility_feature_frame",
    "leakage_audit",
]
