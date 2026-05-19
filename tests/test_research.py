from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from sam.report import summarize_run
from sam.research import (
    ExperimentRegistry,
    ValidationContract,
    validate_experiment_specs,
    validate_research_frame,
    write_experiment_bundle,
)


def test_registered_experiments_have_unique_ids() -> None:
    registry = ExperimentRegistry("configs/experiments")
    specs = registry.specs()
    ids = [spec.id for spec in specs]
    assert {"volatility-regime-scoring", "tactical-etf-allocation"}.issubset(ids)
    assert len(ids) == len(set(ids))
    assert registry.validation_issues().empty


def test_registry_validation_flags_open_core_private_paths() -> None:
    spec = ExperimentRegistry("configs/experiments").get("volatility-regime-scoring")
    unsafe = spec.model_copy(update={"notebook": "private/live_signals.ipynb"})
    issues = validate_experiment_specs([unsafe])
    assert not issues.empty
    assert issues["severity"].iloc[0] == "error"


def test_validation_windows_must_be_chronological() -> None:
    with pytest.raises(ValidationError):
        ValidationContract.model_validate(
            {
                "split_policy": "fixed_chronological",
                "windows": [
                    {
                        "name": "holdout",
                        "start": "2020-01-01",
                        "end": "2020-12-31",
                        "purpose": "final test",
                    },
                    {
                        "name": "train",
                        "start": "2019-01-01",
                        "end": "2019-12-31",
                        "purpose": "training",
                    },
                ],
            }
        )


def test_experiment_bundle_writes_standard_files(tmp_path: Path) -> None:
    spec = ExperimentRegistry("configs/experiments").get("tactical-etf-allocation")
    bundle = write_experiment_bundle(spec, tmp_path, artifact_paths=["missing_plot.png"])
    assert bundle.manifest.is_file()
    assert bundle.experiment_json.is_file()
    assert bundle.contracts_markdown.is_file()
    assert bundle.data_summary.is_file()
    assert bundle.split_summary.is_file()
    assert bundle.model_leaderboard.is_file()
    assert bundle.calibration_diagnostics.is_file()
    assert bundle.operating_points.is_file()
    assert bundle.decision_translation.is_file()
    assert bundle.cost_turnover_sensitivity.is_file()
    assert bundle.artifact_manifest.is_file()
    assert bundle.figure_manifest.is_file()
    assert bundle.limitations_markdown.is_file()
    assert bundle.publication_checklist.is_file()

    payload = json.loads(bundle.experiment_json.read_text(encoding="utf-8"))
    manifest = pd.read_csv(bundle.artifact_manifest)
    data_summary = pd.read_csv(bundle.data_summary)
    leaderboard = pd.read_csv(bundle.model_leaderboard)
    summary = summarize_run(tmp_path)
    assert payload["id"] == "tactical-etf-allocation"
    assert "missing_plot.png" in set(manifest["path"])
    assert data_summary["universe"].iloc[0] == "25 ETF cross-asset allocation universe"
    assert {"baseline", "candidate"}.issubset(set(leaderboard["model_type"]))
    assert summary["experiment"]["id"] == "tactical-etf-allocation"
    assert summary["manifest"]["experiment_id"] == "tactical-etf-allocation"
    assert summary["data_summary"]
    assert summary["model_leaderboard"]
    assert summary["split_summary"]


def test_experiment_bundle_artifact_manifest_is_reproducible(tmp_path: Path) -> None:
    spec = ExperimentRegistry("configs/experiments").get("volatility-regime-scoring")
    first = write_experiment_bundle(spec, tmp_path / "first")
    second = write_experiment_bundle(spec, tmp_path / "second")
    first_manifest = pd.read_csv(first.artifact_manifest)
    second_manifest = pd.read_csv(second.artifact_manifest)
    pd.testing.assert_frame_equal(first_manifest, second_manifest)


def test_experiment_bundle_summarizes_observed_artifacts(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    turnover_path = tmp_path / "turnover.csv"
    pd.DataFrame(
        {
            "model": ["ridge"],
            "rank_ic": [0.12],
            "average_precision": [0.62],
            "brier_score": [0.18],
        }
    ).to_csv(metrics_path, index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "turnover": [1.0, 0.4],
            "cost": [0.001, 0.0004],
        }
    ).to_csv(turnover_path, index=False)
    spec = ExperimentRegistry("configs/experiments").get("tactical-etf-allocation")
    bundle = write_experiment_bundle(
        spec,
        tmp_path / "bundle",
        artifact_paths=[metrics_path, turnover_path],
    )

    leaderboard = pd.read_csv(bundle.model_leaderboard)
    calibration = pd.read_csv(bundle.calibration_diagnostics)
    costs = pd.read_csv(bundle.cost_turnover_sensitivity)
    decision = pd.read_csv(bundle.decision_translation)
    assert leaderboard["status"].iloc[0] == "observed_results"
    assert set(leaderboard["model"]) == {"ridge"}
    assert calibration["brier_score"].iloc[0] == 0.18
    assert costs["turnover"].iloc[0] == 0.7
    assert decision["observed_turnover_rows"].iloc[0] == 2


def test_validate_research_frame_checks_leakage_dates() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "target_date": pd.to_datetime(["2020-01-03", "2020-01-06"]),
            "feature_available_date": pd.to_datetime(["2019-12-31", "2020-01-02"]),
        }
    )
    result = validate_research_frame(
        frame,
        target_date_column="target_date",
        feature_available_column="feature_available_date",
    )
    assert result["missing_dates"] is False
    assert result["monotonic_dates"] is True
    assert result["target_after_signal"] is True
    assert result["features_available_by_signal"] is True
    assert result["duplicate_keys"] == 0


def test_validate_research_frame_flags_future_features() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "feature_available_date": pd.to_datetime(["2020-01-02"]),
        }
    )
    result = validate_research_frame(frame, feature_available_column="feature_available_date")
    assert result["features_available_by_signal"] is False
    assert result["future_feature_rows"] == 1


def test_validate_research_frame_flags_panel_duplicates_and_missing_dates() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-03"]),
            "symbol": ["AAA", "AAA", "AAA"],
        }
    )
    result = validate_research_frame(frame, expected_frequency="D")
    assert result["duplicate_keys"] == 1
    assert result["missing_observations"] == 1


def test_validate_research_frame_checks_symbol_level_chronology() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-01"]),
            "symbol": ["AAA", "AAA", "BBB"],
        }
    )
    result = validate_research_frame(frame)
    assert result["monotonic_dates"] is False
