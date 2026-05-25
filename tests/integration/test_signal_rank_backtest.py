from pathlib import Path

import pytest

from sam.config.loader import BacktestRunConfig, SamSettings
from sam.pipeline.backtest_run import run_backtest


@pytest.fixture
def ensure_fixtures():
    for name in ("ohlcv_ma_baseline.parquet", "signals_etf_rank.parquet"):
        if not (Path("tests/fixtures") / name).exists():
            from scripts.generate_fixtures import main

            main()


def test_signal_rank_backtest(ensure_fixtures, tmp_path):
    settings = SamSettings(sam_artifacts_dir=tmp_path / "artifacts")
    cfg = BacktestRunConfig.model_validate(
        {
            "strategy_config": "configs/strategies/signal_rank.yaml",
            "prices_path": "tests/fixtures/ohlcv_ma_baseline.parquet",
            "signals_path": "tests/fixtures/signals_etf_rank.parquet",
            "preset": "realistic",
            "output_dir": str(tmp_path / "bt_etf"),
        }
    )
    result = run_backtest(cfg, settings, root=Path.cwd())
    assert (tmp_path / "bt_etf" / "metrics.json").exists()
    assert result["metrics"] is not None
