import json
from pathlib import Path

import pandas as pd

from sam.config import MLConfig
from sam.features import build_feature_frame
from sam.ml import predict, ranking_diagnostics, train_model, walk_forward_splits


def test_walk_forward_splits_are_chronological() -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    frame = pd.DataFrame({"date": dates, "symbol": "AAA", "target_forward_return": 0.01})
    splits = walk_forward_splits(frame, train_window_days=10, test_window_days=5, step_days=5)
    assert len(splits) == 4
    first_train, first_test = splits[0]
    assert frame.loc[first_train, "date"].max() < frame.loc[first_test, "date"].min()


def test_walk_forward_splits_respect_embargo_days() -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    frame = pd.DataFrame({"date": dates, "symbol": "AAA", "target_forward_return": 0.01})
    splits = walk_forward_splits(
        frame,
        train_window_days=10,
        test_window_days=5,
        step_days=5,
        embargo_days=2,
    )
    first_train, first_test = splits[0]
    train_end = frame.loc[first_train, "date"].max()
    test_start = frame.loc[first_test, "date"].min()
    assert len(pd.bdate_range(train_end, test_start)) == 4


def test_ranking_diagnostics_are_cross_sectional_by_date() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 4 + ["2020-02-29"] * 4),
            "symbol": ["A", "B", "C", "D"] * 2,
            "prediction": [4, 3, 2, 1, 1, 4, 3, 2],
            "target_forward_return": [0.04, 0.02, -0.01, -0.02, -0.03, 0.05, 0.01, -0.01],
        }
    )
    result = ranking_diagnostics(predictions, top_k=2)
    assert len(result) == 2
    assert result["rank_ic"].min() > 0
    assert result["precision_at_k"].iloc[0] == 1.0


def test_train_model_writes_metadata_and_signals(
    tmp_path: Path, sample_prices: pd.DataFrame
) -> None:
    result = train_model(
        sample_prices,
        config=MLConfig(
            model="ridge",
            horizon_days=5,
            train_window_days=150,
            test_window_days=30,
            step_days=60,
        ),
        out_dir=tmp_path,
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["feature_columns"] == result.feature_columns
    assert (tmp_path / "signals.parquet").is_file()
    assert not result.signals.empty


def test_classification_metrics_include_ranking_and_calibration(
    tmp_path: Path, sample_prices: pd.DataFrame
) -> None:
    result = train_model(
        sample_prices,
        config=MLConfig(
            model="logistic_regression",
            task="classification",
            horizon_days=5,
            train_window_days=150,
            test_window_days=30,
            step_days=60,
        ),
        out_dir=tmp_path,
    )
    assert not result.metrics.empty
    assert {"average_precision", "brier_score", "top_quantile_hit_rate"}.issubset(
        result.metrics.columns
    )
    assert result.metrics["brier_score"].notna().all()


def test_train_model_handles_empty_walk_forward_folds(sample_prices: pd.DataFrame) -> None:
    result = train_model(
        sample_prices,
        config=MLConfig(
            model="ridge",
            horizon_days=5,
            train_window_days=10_000,
            test_window_days=30,
            step_days=60,
        ),
    )
    assert result.predictions.empty
    assert result.metrics.empty
    assert result.signals.empty


def test_feature_frame_can_be_built_for_inference_without_future_targets(
    sample_prices: pd.DataFrame,
) -> None:
    features = build_feature_frame(sample_prices, include_target=False)
    assert "target_forward_return" not in features.columns
    assert pd.to_datetime(features["date"]).max() == pd.to_datetime(sample_prices["date"]).max()


def test_saved_classifier_predict_outputs_probabilities_for_latest_features(
    tmp_path: Path,
    sample_prices: pd.DataFrame,
) -> None:
    result = train_model(
        sample_prices,
        config=MLConfig(
            model="logistic_regression",
            task="classification",
            horizon_days=5,
            train_window_days=150,
            test_window_days=30,
            step_days=60,
        ),
        out_dir=tmp_path,
    )
    predictions = predict(sample_prices, result.model_path)
    assert pd.to_datetime(predictions["date"]).max() == pd.to_datetime(sample_prices["date"]).max()
    assert predictions["prediction"].between(0, 1).all()


def test_classification_training_handles_single_class_folds() -> None:
    dates = pd.date_range("2020-01-01", periods=420, freq="B")
    rows = []
    for idx, symbol in enumerate(["AAA", "BBB", "CCC", "SPY"]):
        prices = 100 + idx + pd.Series(range(len(dates)), dtype=float) * 0.1
        for date, price in zip(dates, prices, strict=True):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
    result = train_model(
        pd.DataFrame(rows),
        config=MLConfig(
            model="logistic_regression",
            task="classification",
            horizon_days=5,
            train_window_days=150,
            test_window_days=30,
            step_days=60,
        ),
    )
    assert not result.predictions.empty
    assert result.predictions["prediction"].between(0, 1).all()
