from pathlib import Path

from sam.cli.main import cmd_report_backtest
from sam.config.loader import BacktestRunConfig, SamSettings
from sam.pipeline.backtest_run import run_backtest

ROOT = Path(__file__).resolve().parents[2]


def test_report_backtest_from_artifacts(tmp_path):
    settings = SamSettings(sam_artifacts_dir=tmp_path / "artifacts")
    out_dir = tmp_path / "bt"
    cfg = BacktestRunConfig.model_validate(
        {
            "strategy_config": "configs/strategies/ma_crossover.yaml",
            "prices_path": "tests/fixtures/ohlcv_ma_baseline.parquet",
            "preset": "realistic",
            "output_dir": str(out_dir),
        }
    )
    run_backtest(cfg, settings, root=ROOT)

    class Args:
        output_dir = str(out_dir)

    assert cmd_report_backtest(Args()) == 0
    assert (out_dir / "report.html").exists()
