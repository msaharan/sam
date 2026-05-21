"""Model runners for volatility-regime scoring."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sam.risk_metrics import classification_metrics, top_k_alert_metrics
from sam.volatility_config import VolatilityRegimeRunConfig

logger = logging.getLogger(__name__)

DOMAIN_RULES = {
    "Rule[SPY realized volatility 20d]": "spy_rv_20d",
    "Rule[SPY Parkinson realized volatility 20d]": "spy_parkinson_rv_20d",
    "Rule[SPY Garman-Klass realized volatility 20d]": "spy_garman_klass_rv_20d",
    "Rule[VIX close]": "vix_close",
    "Rule[VIX z-score 252d]": "vix_zscore_252d",
    "Rule[VIX/realized volatility gap 20d]": "vix_minus_spy_rv_20d",
}


@dataclass
class ModelRunResult:
    name: str
    family: str
    holdout_scores: pd.Series
    metrics: dict[str, float]
    timing: dict[str, float] = field(default_factory=dict)
    best_params: dict | None = None
    cv_summary: pd.DataFrame | None = None
    error: str | None = None


def run_domain_rules(
    frame: pd.DataFrame,
    *,
    holdout_mask: pd.Series,
) -> list[ModelRunResult]:
    holdout = frame.loc[holdout_mask]
    y = holdout["target_high_vol_20d"]
    results = []
    for name, column in DOMAIN_RULES.items():
        if column not in frame.columns:
            results.append(
                ModelRunResult(
                    name,
                    "domain_rule",
                    pd.Series(dtype=float),
                    {},
                    error=f"missing {column}",
                )
            )
            continue
        scores = holdout[column].astype(float)
        metrics = classification_metrics(y, scores, probabilities=False)
        metrics.update(top_k_alert_metrics(y, scores))
        results.append(
            ModelRunResult(
                name,
                "domain_rule",
                scores,
                metrics,
                timing={"workflow_seconds": 0.0},
            )
        )
    return results


def run_xgboost_incumbent(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    train_mask: pd.Series,
    holdout_mask: pd.Series,
    config: VolatilityRegimeRunConfig,
    show_progress: bool = True,
) -> ModelRunResult:
    name = "XGBoost[Raw all-history incumbent]"
    try:
        import xgboost as xgb
    except ImportError as exc:
        return ModelRunResult(
            name,
            "classical_incumbent",
            pd.Series(dtype=float),
            {},
            error=str(exc),
        )

    train = frame.loc[train_mask]
    holdout = frame.loc[holdout_mask]
    x_train = train[feature_columns]
    y_train = train["target_high_vol_20d"].astype(int)
    x_holdout = holdout[feature_columns]
    y_holdout = holdout["target_high_vol_20d"].astype(int)

    start = time.perf_counter()
    n_iter = (
        config.fast_xgboost_tuning_iterations
        if config.fast_mode
        else config.xgboost_tuning_iterations
    )
    params, cv_summary = _tune_xgboost(
        frame,
        feature_columns=feature_columns,
        tuning_mask=frame["split_tuning"] | frame["split_calibration"],
        n_iter=n_iter,
        cv_splits=config.xgboost_cv_splits,
        show_progress=show_progress,
    )
    from sam.progress import progress_task

    with progress_task("XGBoost final fit", enabled=show_progress):
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=7,
            n_jobs=1,
            **params,
        )
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    predict_start = time.perf_counter()
    scores = pd.Series(model.predict_proba(x_holdout)[:, 1], index=holdout.index)
    predict_seconds = time.perf_counter() - predict_start
    metrics = classification_metrics(y_holdout, scores)
    metrics.update(top_k_alert_metrics(y_holdout, scores))
    return ModelRunResult(
        name,
        "classical_incumbent",
        scores,
        metrics,
        timing={
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "workflow_seconds": fit_seconds + predict_seconds,
        },
        best_params=params,
        cv_summary=cv_summary,
    )


def run_tabpfn_direct(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    train_mask: pd.Series,
    holdout_mask: pd.Series,
) -> ModelRunResult:
    name = "TabPFN[Direct all-history]"
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        return ModelRunResult(name, "direct_tfm", pd.Series(dtype=float), {}, error=str(exc))

    train = frame.loc[train_mask]
    holdout = frame.loc[holdout_mask]
    start = time.perf_counter()
    model = TabPFNClassifier()
    model.fit(
        train[feature_columns].to_numpy(),
        train["target_high_vol_20d"].astype(int).to_numpy(),
    )
    fit_seconds = time.perf_counter() - start
    predict_start = time.perf_counter()
    probabilities = model.predict_proba(holdout[feature_columns].to_numpy())[:, 1]
    predict_seconds = time.perf_counter() - predict_start
    scores = pd.Series(probabilities, index=holdout.index)
    metrics = classification_metrics(holdout["target_high_vol_20d"], scores)
    metrics.update(top_k_alert_metrics(holdout["target_high_vol_20d"], scores))
    return ModelRunResult(
        name,
        "direct_tfm",
        scores,
        metrics,
        timing={
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "workflow_seconds": fit_seconds + predict_seconds,
        },
    )


def run_tabicl_direct(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    train_mask: pd.Series,
    holdout_mask: pd.Series,
) -> ModelRunResult:
    name = "TabICL[Direct all-history]"
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        try:
            from tabicl.sklearn.classifier import TabICLClassifier
        except ImportError as exc:
            return ModelRunResult(name, "direct_tfm", pd.Series(dtype=float), {}, error=str(exc))

    train = frame.loc[train_mask]
    holdout = frame.loc[holdout_mask]
    start = time.perf_counter()
    model = TabICLClassifier()
    model.fit(
        train[feature_columns].to_numpy(),
        train["target_high_vol_20d"].astype(int).to_numpy(),
    )
    fit_seconds = time.perf_counter() - start
    predict_start = time.perf_counter()
    probabilities = model.predict_proba(holdout[feature_columns].to_numpy())[:, 1]
    predict_seconds = time.perf_counter() - predict_start
    scores = pd.Series(probabilities, index=holdout.index)
    metrics = classification_metrics(holdout["target_high_vol_20d"], scores)
    metrics.update(top_k_alert_metrics(holdout["target_high_vol_20d"], scores))
    return ModelRunResult(
        name,
        "direct_tfm",
        scores,
        metrics,
        timing={
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "workflow_seconds": fit_seconds + predict_seconds,
        },
    )


def results_to_holdout_summary(
    results: list[ModelRunResult],
    *,
    holdout_rows: int,
    holdout_rate: float,
) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "Model": result.name,
            "Family": result.family,
            "Evaluation Window": "holdout",
            "Rows": holdout_rows,
            "High Vol Rate": holdout_rate,
            "Error": result.error,
            **result.metrics,
            **{f"{key}_seconds": value for key, value in result.timing.items()},
        }
        if result.best_params:
            row["Best Params"] = json.dumps(result.best_params)
        rows.append(row)
    return pd.DataFrame(rows)


def _tune_xgboost(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    tuning_mask: pd.Series,
    n_iter: int,
    cv_splits: int,
    show_progress: bool = True,
) -> tuple[dict, pd.DataFrame]:
    import xgboost as xgb
    from sklearn.model_selection import ParameterSampler

    tuning = frame.loc[tuning_mask].sort_values("date")
    dates = pd.to_datetime(tuning["date"]).unique()
    if len(dates) < cv_splits + 1:
        return {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 100}, pd.DataFrame()
    fold_size = len(dates) // (cv_splits + 1)
    param_space = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [100, 200, 300, 400],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 5, 10, 20],
    }
    from sam.progress import progress_iter

    sampler = list(ParameterSampler(param_space, n_iter=min(n_iter, 16), random_state=7))
    best_params = sampler[0]
    best_score = -1.0
    best_cv_rows: list[dict] = []
    for params in progress_iter(
        sampler,
        desc="XGBoost CV tuning",
        total=len(sampler),
        enabled=show_progress,
        unit="config",
    ):
        fold_scores = []
        fold_rows: list[dict] = []
        for fold in range(cv_splits):
            train_end = (fold + 1) * fold_size
            test_end = train_end + fold_size
            train_dates = set(dates[:train_end])
            test_dates = set(dates[train_end:test_end])
            train_idx = tuning["date"].isin(train_dates)
            test_idx = tuning["date"].isin(test_dates)
            if train_idx.sum() < 50 or test_idx.sum() < 10:
                continue
            if tuning.loc[train_idx, "target_high_vol_20d"].nunique() < 2:
                continue
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=7,
                n_jobs=1,
                **params,
            )
            model.fit(
                tuning.loc[train_idx, feature_columns],
                tuning.loc[train_idx, "target_high_vol_20d"].astype(int),
            )
            preds = model.predict_proba(tuning.loc[test_idx, feature_columns])[:, 1]
            y = tuning.loc[test_idx, "target_high_vol_20d"].astype(int)
            from sklearn.metrics import average_precision_score

            if y.sum() > 0:
                ap = float(average_precision_score(y, preds))
                fold_scores.append(ap)
                fold_rows.append(
                    {
                        "fold": fold,
                        "average_precision": ap,
                        "train_rows": int(train_idx.sum()),
                        "test_rows": int(test_idx.sum()),
                    }
                )
        if fold_scores and np.mean(fold_scores) > best_score:
            best_score = float(np.mean(fold_scores))
            best_params = params
            best_cv_rows = [
                {**row, "mean_cv_ap": best_score, **params} for row in fold_rows
            ]
    cv_summary = pd.DataFrame(best_cv_rows)
    return best_params, cv_summary


__all__ = [
    "DOMAIN_RULES",
    "ModelRunResult",
    "results_to_holdout_summary",
    "run_domain_rules",
    "run_tabicl_direct",
    "run_tabpfn_direct",
    "run_xgboost_incumbent",
]
