from __future__ import annotations

from pathlib import Path

import pandas as pd

from sam.cli import main
from sam.risk_metrics import classification_metrics, month_block_bootstrap_ap
from sam.volatility_experiment import run_volatility_regime_experiment


def test_classification_metrics_on_perfect_scores() -> None:
    y = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(y, scores)
    assert metrics["average_precision"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_run_volatility_regime_experiment_synthetic(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    result = run_volatility_regime_experiment(
        out_dir=out_dir,
        use_synthetic_data=True,
        skip_tfm=True,
        skip_xgboost=True,
        fast_mode=True,
        write_bundle=False,
        write_figures=False,
        write_report=True,
        show_progress=False,
    )
    assert (out_dir / "report.html").is_file()
    assert (out_dir / "report.md").is_file()
    summary = pd.read_csv(out_dir / "publication_holdout_summary.csv")
    operating = pd.read_csv(out_dir / "publication_operating_summary.csv")
    leakage = pd.read_csv(out_dir / "leakage_checks.csv")
    drift = pd.read_csv(out_dir / "feature_drift_summary.csv")
    assert not summary.empty
    assert not operating.empty
    assert not leakage.empty
    assert not drift.empty
    assert result.holdout_summary.shape[0] >= len(summary)
    rule_rows = summary[summary["Family"] == "domain_rule"]
    assert rule_rows["Error"].isna().all()
    split_summary = pd.read_csv(out_dir / "split_summary.csv")
    assert {"start_date", "end_date", "purpose"}.issubset(split_summary.columns)
    leakage_checks = pd.read_csv(out_dir / "leakage_checks.csv")
    assert len(leakage_checks) >= 10


def test_cli_experiment_run_synthetic_fast(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli-run"
    assert (
        main(
            [
                "experiment",
                "run",
                "--experiment",
                "volatility-regime-scoring",
                "--synthetic",
                "--fast",
                "--skip-tfm",
                "--skip-xgboost",
                "--no-figures",
                "--out",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (out_dir / "publication_holdout_summary.csv").is_file()
    assert (out_dir / "feature_drift_summary.csv").is_file()


def test_month_block_bootstrap_returns_intervals() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=120, freq="B"),
            "target_high_vol_20d": [0, 1] * 60,
            "target_forward_rv_20d": [0.1, 0.3] * 60,
            "score": [0.1, 0.9] * 60,
        }
    )
    boot = month_block_bootstrap_ap(frame, score_column="score", n_iterations=20)
    assert "ap_point" in boot
    assert "ap_low" in boot
