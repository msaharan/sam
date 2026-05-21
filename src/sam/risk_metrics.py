"""Classification and ranking metrics for volatility-regime scoring."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from sam.risk import alert_operating_points


def safe_roc_auc(y_true: pd.Series, scores: pd.Series) -> float:
    y = y_true.astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_average_precision(y_true: pd.Series, scores: pd.Series) -> float:
    y = y_true.astype(int)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, scores))


def expected_calibration_error(
    y_true: pd.Series,
    probabilities: pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    frame = pd.DataFrame({"y": y_true.astype(int), "p": probabilities.clip(0, 1)}).dropna()
    if frame.empty:
        return float("nan")
    frame["bin"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True)
    ece = 0.0
    for _, group in grouped:
        weight = len(group) / len(frame)
        ece += weight * abs(group["p"].mean() - group["y"].mean())
    return float(ece)


def classification_metrics(
    y_true: pd.Series,
    scores: pd.Series,
    *,
    probabilities: bool = True,
) -> dict[str, float]:
    y = y_true.astype(int)
    pred = scores.astype(float)
    payload = {
        "average_precision": safe_average_precision(y, pred),
        "roc_auc": safe_roc_auc(y, pred),
    }
    if probabilities:
        clipped = pred.clip(1e-6, 1 - 1e-6)
        payload["brier_score"] = float(brier_score_loss(y, clipped))
        payload["log_loss"] = float(log_loss(y, clipped))
        payload["ece"] = expected_calibration_error(y, clipped)
    return payload


def top_k_alert_metrics(
    y_true: pd.Series,
    scores: pd.Series,
    *,
    fractions: Iterable[float] = (0.05, 0.1, 0.2),
) -> dict[str, float]:
    frame = pd.DataFrame({"y": y_true.astype(int), "score": scores}).dropna().sort_values(
        "score", ascending=False
    )
    rows = len(frame)
    positives = int(frame["y"].sum())
    payload: dict[str, float] = {}
    for fraction in fractions:
        alerts = max(1, int(math.ceil(rows * fraction)))
        selected = frame.head(alerts)
        payload[f"top_{int(fraction * 100)}pct_alerts"] = float(alerts)
        payload[f"top_{int(fraction * 100)}pct_precision"] = float(selected["y"].mean())
        payload[f"top_{int(fraction * 100)}pct_recall"] = (
            float(selected["y"].sum() / positives) if positives else float("nan")
        )
    return payload


def score_decile_summary(y_true: pd.Series, scores: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"y": y_true.astype(int), "score": scores}).dropna()
    if frame.empty:
        return {
            "top_decile_high_vol_rate": float("nan"),
            "bottom_decile_high_vol_rate": float("nan"),
        }
    frame["decile"] = pd.qcut(frame["score"], 10, labels=False, duplicates="drop")
    rates = frame.groupby("decile")["y"].mean()
    return {
        "top_decile_high_vol_rate": float(rates.max()),
        "bottom_decile_high_vol_rate": float(rates.min()),
    }


def threshold_sensitivity_table(
    frame: pd.DataFrame,
    *,
    score_column: str,
    rv_column: str = "target_forward_rv_20d",
    quantiles: Iterable[float],
    holdout_mask: pd.Series | None = None,
) -> pd.DataFrame:
    if holdout_mask is None:
        holdout = frame.copy()
    else:
        holdout = frame.loc[holdout_mask].copy()
    if score_column not in holdout.columns:
        raise KeyError(f"missing score column: {score_column}")
    rows = []
    for quantile in quantiles:
        threshold = float(holdout[rv_column].quantile(quantile))
        labels = (holdout[rv_column] >= threshold).astype(int)
        valid = labels.notna() & holdout[score_column].notna()
        metrics = classification_metrics(
            labels.loc[valid],
            holdout.loc[valid, score_column],
            probabilities=False,
        )
        rows.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "holdout_high_vol_rate": float(labels.mean()),
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "score_column": score_column,
            }
        )
    return pd.DataFrame(rows)


def regime_metrics(
    frame: pd.DataFrame,
    *,
    score_columns: dict[str, str],
    regimes: dict[str, list[str]],
    target_column: str = "target_high_vol_20d",
) -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(frame["date"])
    for regime_name, bounds in regimes.items():
        start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        mask = (dates >= start) & (dates <= end)
        subset = frame.loc[mask]
        positives = int(subset[target_column].sum()) if target_column in subset else 0
        if positives == 0:
            continue
        for model, score_col in score_columns.items():
            metrics = classification_metrics(
                subset[target_column], subset[score_col], probabilities=False
            )
            rows.append(
                {
                    "regime": regime_name,
                    "rows": int(len(subset)),
                    "high_vol_rows": positives,
                    "high_vol_rate": float(subset[target_column].mean()),
                    "model": model,
                    "average_precision": metrics["average_precision"],
                    "roc_auc": metrics["roc_auc"],
                }
            )
    return pd.DataFrame(rows)


def month_block_bootstrap_ap(
    frame: pd.DataFrame,
    *,
    score_column: str,
    target_column: str = "target_high_vol_20d",
    n_iterations: int = 300,
    random_state: int = 7,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    dates = pd.to_datetime(frame["date"])
    months = pd.Series(dates).dt.to_period("M")
    unique_months = months.unique()
    from sam.progress import progress_range

    scores = []
    desc = progress_desc or "Month-block bootstrap"
    for _ in progress_range(
        n_iterations,
        desc=desc,
        enabled=show_progress,
        unit="draw",
    ):
        sampled_months = rng.choice(unique_months, size=len(unique_months), replace=True)
        mask = months.isin(sampled_months)
        subset = frame.loc[mask]
        if subset[target_column].nunique() < 2:
            continue
        scores.append(
            safe_average_precision(subset[target_column], subset[score_column])
        )
    if not scores:
        return {
            "ap_point": float("nan"),
            "ap_median": float("nan"),
            "ap_low": float("nan"),
            "ap_high": float("nan"),
        }
    series = pd.Series(scores)
    point = safe_average_precision(frame[target_column], frame[score_column])
    return {
        "ap_point": point,
        "ap_median": float(series.median()),
        "ap_low": float(series.quantile(0.025)),
        "ap_high": float(series.quantile(0.975)),
    }


def population_stability_index(
    reference: pd.Series,
    comparison: pd.Series,
    *,
    bins: int = 10,
) -> float:
    """PSI between reference (calibration) and comparison (holdout) distributions."""

    ref = reference.dropna()
    cmp = comparison.dropna()
    if len(ref) < bins or len(cmp) < bins or ref.nunique() < 2:
        return float("nan")
    try:
        _, edges = pd.qcut(ref, q=bins, retbins=True, duplicates="drop")
    except ValueError:
        return float("nan")
    ref_pct = pd.cut(ref, bins=edges).value_counts(normalize=True).sort_index()
    cmp_pct = pd.cut(cmp, bins=edges).value_counts(normalize=True).sort_index()
    ref_pct = ref_pct.reindex(ref_pct.index.union(cmp_pct.index), fill_value=0).clip(lower=1e-6)
    cmp_pct = cmp_pct.reindex(ref_pct.index, fill_value=0).clip(lower=1e-6)
    return float(((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)).sum())


def feature_drift_psi_summary(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    reference_mask: pd.Series,
    comparison_mask: pd.Series,
    bins: int = 10,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Summarize feature drift (PSI) between calibration and holdout windows."""

    rows = []
    from sam.progress import progress_iter

    reference = frame.loc[reference_mask]
    comparison = frame.loc[comparison_mask]
    for column in progress_iter(
        feature_columns,
        desc="Feature drift (PSI)",
        total=len(feature_columns),
        enabled=show_progress,
        unit="feat",
    ):
        if column not in frame.columns:
            continue
        psi = population_stability_index(
            reference[column],
            comparison[column],
            bins=bins,
        )
        rows.append(
            {
                "feature": column,
                "psi": psi,
                "reference_rows": int(reference[column].notna().sum()),
                "comparison_rows": int(comparison[column].notna().sum()),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def build_operating_summary(
    frame: pd.DataFrame,
    *,
    score_columns: dict[str, str],
    target_column: str = "target_high_vol_20d",
    recalls: Iterable[float] = (0.5, 0.8, 0.9),
) -> pd.DataFrame:
    scores = frame[list(score_columns.values()) + [target_column]].copy()
    scores = scores.rename(columns={value: key for key, value in score_columns.items()})
    return alert_operating_points(
        scores,
        score_columns=list(score_columns.keys()),
        target_column=target_column,
        recalls=recalls,
    )


__all__ = [
    "build_operating_summary",
    "classification_metrics",
    "expected_calibration_error",
    "feature_drift_psi_summary",
    "month_block_bootstrap_ap",
    "population_stability_index",
    "regime_metrics",
    "safe_average_precision",
    "safe_roc_auc",
    "score_decile_summary",
    "threshold_sensitivity_table",
    "top_k_alert_metrics",
]
