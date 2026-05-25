from pathlib import Path

import pytest

from sam.config.loader import BacktestRunConfig, SamSettings
from sam.pipeline.backtest_run import run_backtest


@pytest.fixture
def ensure_fixtures():
    path = Path("tests/fixtures/ohlcv_ma_baseline.parquet")
    if not path.exists():
        from scripts.generate_fixtures import main

        main()


def test_ma_backtest_runs(ensure_fixtures, tmp_path):
    settings = SamSettings(sam_artifacts_dir=tmp_path / "artifacts")
    cfg = BacktestRunConfig.model_validate(
        {
            "strategy_config": "configs/strategies/ma_crossover.yaml",
            "prices_path": "tests/fixtures/ohlcv_ma_baseline.parquet",
            "preset": "realistic",
            "initial_cash": 100_000,
            "output_dir": str(tmp_path / "bt"),
        }
    )
    result = run_backtest(cfg, settings, root=Path.cwd())
    assert "total_return" in result["metrics"] or result["metrics"]
    assert (tmp_path / "bt" / "metrics.json").exists()
    manifest = (tmp_path / "bt" / "run_manifest.json").read_text()
    assert "dependency_versions" in manifest
    assert "config_hashes" in manifest
    assert "provider_adjusted" in manifest
