"""Functional tests for ML4T-backed backtest handoff and promotion gates."""

from pathlib import Path

import pytest

from sam.config.loader import BacktestRunConfig, SamSettings
from sam.pipeline.backtest_run import run_backtest

ROOT = Path(__file__).resolve().parents[2]


def test_backtest_promotion_gate_failure(tmp_path):
    cfg = BacktestRunConfig.model_validate(
        {
            "strategy_config": "configs/strategies/ma_crossover.yaml",
            "prices_path": "tests/fixtures/ohlcv_ma_baseline.parquet",
            "output_dir": str(tmp_path / "backtest_fail"),
            "min_trades": 10_000,
            "enforce_promotion_gates": True,
            "promotion_gates": {"min_trades": 10_000, "enforce": True},
        }
    )
    settings = SamSettings(sam_artifacts_dir=tmp_path / "artifacts")
    with pytest.raises(RuntimeError, match="promotion gates failed"):
        run_backtest(cfg, settings, root=ROOT)


def test_backtest_manifest_records_diagnostic_fields(tmp_path):
    cfg = BacktestRunConfig.model_validate(
        {
            "strategy_config": "configs/strategies/ma_crossover.yaml",
            "prices_path": "tests/fixtures/ohlcv_ma_baseline.parquet",
            "output_dir": str(tmp_path / "backtest_ok"),
            "enforce_promotion_gates": False,
        }
    )
    settings = SamSettings(sam_artifacts_dir=tmp_path / "artifacts")
    run_backtest(cfg, settings, root=ROOT)
    manifest = (tmp_path / "backtest_ok" / "run_manifest.json").read_text()
    assert "promotion_state" in manifest
    assert "dependency_versions" in manifest


def test_backtest_dsr_gate_recorded(tmp_path):
    cfg = BacktestRunConfig.model_validate(
        {
            "strategy_config": "configs/strategies/ma_crossover.yaml",
            "prices_path": "tests/fixtures/ohlcv_ma_baseline.parquet",
            "output_dir": str(tmp_path / "backtest_dsr"),
            "enforce_promotion_gates": False,
            "promotion_gates": {"min_dsr_probability": 0.0, "n_trials": 1, "enforce": False},
        }
    )
    settings = SamSettings(sam_artifacts_dir=tmp_path / "artifacts")
    run_backtest(cfg, settings, root=ROOT)
    import json

    manifest = json.loads((tmp_path / "backtest_dsr" / "run_manifest.json").read_text())
    checks = manifest["checks"]
    assert "dsr_probability" in checks
