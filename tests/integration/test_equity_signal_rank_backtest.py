from pathlib import Path

from sam.config.loader import BacktestRunConfig, SamSettings
from sam.pipeline.backtest_run import run_backtest

ROOT = Path(__file__).resolve().parents[2]


def test_equity_signal_rank_backtest(tmp_path):
    settings = SamSettings(sam_artifacts_dir=tmp_path / "artifacts")
    cfg = BacktestRunConfig.model_validate(
        {
            "strategy_config": "configs/strategies/equity_signal_rank.yaml",
            "prices_path": "tests/fixtures/ohlcv_equity_sample.parquet",
            "signals_path": "tests/fixtures/signals_equity_rank.parquet",
            "preset": "realistic",
            "output_dir": str(tmp_path / "bt"),
        }
    )
    result = run_backtest(cfg, settings, root=ROOT)
    assert (tmp_path / "bt" / "run_manifest.json").exists()
    assert result["metrics"] is not None
