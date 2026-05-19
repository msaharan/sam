"""Walk-forward ML baselines for portfolio signal research."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sam.config import MLConfig
from sam.features import build_feature_frame
from sam.io import ensure_dir

FEATURE_COLUMNS = [
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "vol_21d",
    "vol_63d",
    "volume_21d",
    "dollar_volume_21d",
    "drawdown_63d",
    "ma_ratio_20_100",
    "beta_63d",
    "ret_21d_rank",
    "ret_63d_rank",
    "ret_126d_rank",
    "vol_63d_rank",
    "dollar_volume_21d_rank",
    "regime_vol_63d",
]


@dataclass(frozen=True)
class MLResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    feature_columns: list[str]
    signals: pd.DataFrame
    model_path: Path | None = None


def train_model(
    prices: pd.DataFrame,
    *,
    config: MLConfig | None = None,
    out_dir: str | Path | None = None,
) -> MLResult:
    cfg = config or MLConfig()
    frame = build_feature_frame(prices, horizon_days=cfg.horizon_days)
    frame["target_up"] = (frame["target_forward_return"] > 0).astype(int)
    predictions = []
    metrics = []
    last_model = None
    fold_count = 0
    for train_idx, test_idx in walk_forward_splits(
        frame,
        train_window_days=cfg.train_window_days,
        test_window_days=cfg.test_window_days,
        step_days=cfg.step_days,
        embargo_days=cfg.embargo_days,
    ):
        train = frame.loc[train_idx]
        test = frame.loc[test_idx]
        y_train = (
            train["target_up"]
            if cfg.task == "classification"
            else train["target_forward_return"]
        )
        model = _make_fold_model(cfg, y_train)
        model.fit(train[FEATURE_COLUMNS], y_train)
        pred = _predict(model, test[FEATURE_COLUMNS], cfg)
        fold = test[["date", "symbol", "target_forward_return"]].copy()
        fold["prediction"] = pred
        fold["fold_start"] = test["date"].min()
        predictions.append(fold)
        metrics.append(_fold_metrics(fold, cfg))
        last_model = model
        fold_count += 1
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    metrics_frame = pd.DataFrame(metrics)
    signals = signal_outputs(prediction_frame)
    model_path = None
    if out_dir is not None:
        target_dir = ensure_dir(out_dir)
        prediction_frame.to_parquet(target_dir / "predictions.parquet", index=False)
        metrics_frame.to_csv(target_dir / "metrics.csv", index=False)
        signals.to_parquet(target_dir / "signals.parquet", index=False)
        metadata = {
            "name": cfg.name,
            "model": cfg.model,
            "task": cfg.task,
            "horizon_days": cfg.horizon_days,
            "train_window_days": cfg.train_window_days,
            "test_window_days": cfg.test_window_days,
            "step_days": cfg.step_days,
            "embargo_days": cfg.embargo_days,
            "fold_count": fold_count,
            "feature_columns": FEATURE_COLUMNS,
            "validation_metrics": metrics_frame.to_dict(orient="records"),
        }
        (target_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if last_model is not None:
            model_path = target_dir / "model.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(
                    {"model": last_model, "config": cfg.model_dump(), "features": FEATURE_COLUMNS},
                    handle,
                )
    return MLResult(prediction_frame, metrics_frame, FEATURE_COLUMNS, signals, model_path)


def predict(prices: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    with Path(model_path).open("rb") as handle:
        payload = pickle.load(handle)
    config = MLConfig.model_validate(payload.get("config", {}))
    features = build_feature_frame(
        prices,
        horizon_days=config.horizon_days,
        include_target=False,
    )
    features["prediction"] = _predict(payload["model"], features[payload["features"]], config)
    return features[["date", "symbol", "prediction"]]


def walk_forward_splits(
    frame: pd.DataFrame,
    *,
    train_window_days: int,
    test_window_days: int,
    step_days: int,
    embargo_days: int = 0,
) -> list[tuple[pd.Index, pd.Index]]:
    if min(train_window_days, test_window_days, step_days) < 1:
        raise ValueError("train, test, and step windows must be positive")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    dates = pd.Series(pd.to_datetime(frame["date"]).sort_values().unique())
    splits = []
    start = 0
    while start + train_window_days + embargo_days + test_window_days <= len(dates):
        train_dates = set(dates.iloc[start : start + train_window_days])
        test_start = start + train_window_days + embargo_days
        test_dates = set(
            dates.iloc[test_start : test_start + test_window_days]
        )
        train_idx = frame.index[pd.to_datetime(frame["date"]).isin(train_dates)]
        test_idx = frame.index[pd.to_datetime(frame["date"]).isin(test_dates)]
        if len(train_idx) and len(test_idx):
            splits.append((train_idx, test_idx))
        start += step_days
    return splits


def ranking_diagnostics(
    predictions: pd.DataFrame,
    *,
    date_column: str = "date",
    symbol_column: str = "symbol",
    score_column: str = "prediction",
    target_column: str = "target_forward_return",
    top_k: int = 5,
) -> pd.DataFrame:
    """Evaluate cross-sectional ranking quality by date.

    The target is continuous by default. Positive-class metrics treat values above zero as useful
    outcomes, which matches the ETF/ranking workflows without making a hard trading claim.
    """

    required = {date_column, symbol_column, score_column, target_column}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    columns = [
        "date",
        "rows",
        "top_k",
        "rank_ic",
        "average_precision",
        "precision_at_k",
        "top_k_mean_return",
        "overall_mean_return",
        "positive_rate",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    frame = predictions[[date_column, symbol_column, score_column, target_column]].dropna().copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame[date_column] = pd.to_datetime(frame[date_column])

    rows = []
    for date, group in frame.groupby(date_column, sort=True):
        group = group.sort_values([score_column, symbol_column], ascending=[False, True])
        k = min(top_k, len(group))
        selected = group.head(k)
        positives = (group[target_column] > 0).astype(int)
        selected_positives = (selected[target_column] > 0).astype(int)
        rows.append(
            {
                "date": date,
                "rows": int(len(group)),
                "top_k": int(k),
                "rank_ic": _spearman_or_nan(group[target_column], group[score_column]),
                "average_precision": _average_precision_or_nan(positives, group[score_column]),
                "precision_at_k": float(selected_positives.mean()) if k else float("nan"),
                "top_k_mean_return": float(selected[target_column].mean())
                if k
                else float("nan"),
                "overall_mean_return": float(group[target_column].mean()),
                "positive_rate": float(positives.mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def aggregate_ranking_metrics(
    predictions: pd.DataFrame,
    *,
    top_k: int = 5,
) -> dict[str, float]:
    diagnostics = ranking_diagnostics(predictions, top_k=top_k)
    if diagnostics.empty:
        return {
            "mean_cross_sectional_rank_ic": float("nan"),
            "average_precision": float("nan"),
            "precision_at_k": float("nan"),
            "top_k_mean_return": float("nan"),
            "top_k_excess_return": float("nan"),
        }
    return {
        "mean_cross_sectional_rank_ic": float(diagnostics["rank_ic"].mean()),
        "average_precision": float(diagnostics["average_precision"].mean()),
        "precision_at_k": float(diagnostics["precision_at_k"].mean()),
        "top_k_mean_return": float(diagnostics["top_k_mean_return"].mean()),
        "top_k_excess_return": float(
            (diagnostics["top_k_mean_return"] - diagnostics["overall_mean_return"]).mean()
        ),
    }


def signal_outputs(predictions: pd.DataFrame, *, top_quantile: float = 0.2) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["date", "symbol", "prediction", "signal_rank", "long_signal"])
    signals = predictions[["date", "symbol", "prediction"]].copy()
    signals["signal_rank"] = signals.groupby("date")["prediction"].rank(pct=True, ascending=False)
    signals["long_signal"] = signals["signal_rank"] <= top_quantile
    return signals.sort_values(["date", "signal_rank", "symbol"]).reset_index(drop=True)


def _make_model(config: MLConfig):
    if config.model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if config.model == "elastic_net":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.2))
    if config.model == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if config.model == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=7, min_samples_leaf=5)
    if config.model == "ensemble_rank":
        return _EnsembleRankRegressor(
            [
                Ridge(alpha=1.0),
                RandomForestRegressor(n_estimators=100, random_state=7, min_samples_leaf=5),
                HistGradientBoostingRegressor(max_iter=100, random_state=7),
            ]
        )
    return HistGradientBoostingRegressor(max_iter=100, random_state=7)


def _make_fold_model(config: MLConfig, y_train: pd.Series):
    if config.task == "classification" and y_train.nunique() < 2:
        return DummyClassifier(strategy="constant", constant=int(y_train.iloc[0]))
    return _make_model(config)


def _predict(model, features: pd.DataFrame, config: MLConfig) -> np.ndarray:
    if config.task == "classification" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return probabilities[:, classes.index(1)]
        return np.zeros(len(features), dtype=float)
    return model.predict(features)


def _fold_metrics(fold: pd.DataFrame, config: MLConfig) -> dict[str, float | str]:
    y = fold["target_forward_return"]
    pred = fold["prediction"]
    top_quantile = 0.2
    top_cutoff = pred.rank(pct=True, ascending=False) <= top_quantile
    ranking = aggregate_ranking_metrics(
        fold,
        top_k=min(5, int(fold["symbol"].nunique()) if "symbol" in fold else 5),
    )
    payload: dict[str, float | str] = {
        "fold_start": str(fold["fold_start"].iloc[0]),
        "fold_rows": int(len(fold)),
        "positive_return_rate": float((y > 0).mean()),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "rank_ic": _spearman_or_nan(y, pred),
        "mean_cross_sectional_rank_ic": ranking["mean_cross_sectional_rank_ic"],
        "average_precision": ranking["average_precision"],
        "precision_at_k": ranking["precision_at_k"],
        "top_k_mean_return": ranking["top_k_mean_return"],
        "top_k_excess_return": ranking["top_k_excess_return"],
        "top_quantile_mean_return": (
            float(y[top_cutoff].mean()) if top_cutoff.any() else float("nan")
        ),
        "overall_mean_return": float(y.mean()),
    }
    if config.task == "classification":
        up = (y > 0).astype(int)
        payload["accuracy"] = float(accuracy_score(up, pred > 0.5))
        payload["roc_auc"] = float(roc_auc_score(up, pred)) if up.nunique() > 1 else float("nan")
        payload["average_precision"] = _average_precision_or_nan(up, pred)
        payload["brier_score"] = float(brier_score_loss(up, pred.clip(0.0, 1.0)))
        payload["top_quantile_hit_rate"] = (
            float(up[top_cutoff].mean()) if top_cutoff.any() else float("nan")
        )
    return payload


def _spearman_or_nan(y: pd.Series, pred: pd.Series) -> float:
    if y.nunique(dropna=True) < 2 or pred.nunique(dropna=True) < 2:
        return float("nan")
    return float(y.corr(pred, method="spearman"))


def _average_precision_or_nan(y_binary: pd.Series, pred: pd.Series) -> float:
    y_binary = y_binary.astype(int)
    if y_binary.sum() == 0:
        return float("nan")
    if y_binary.nunique(dropna=True) == 1:
        return 1.0
    return float(average_precision_score(y_binary, pred))


class _EnsembleRankRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators: list):
        self.estimators = estimators

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.estimators_ = [clone(estimator).fit(x, y) for estimator in self.estimators]
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        predictions = np.column_stack([estimator.predict(x) for estimator in self.estimators_])
        ranks = np.apply_along_axis(
            lambda values: pd.Series(values).rank(pct=True).to_numpy(),
            axis=0,
            arr=predictions,
        )
        return ranks.mean(axis=1)
