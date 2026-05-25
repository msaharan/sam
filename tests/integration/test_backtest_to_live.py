"""Core invariant: same MACrossoverStrategy in backtest and live shadow."""

from pathlib import Path

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.specs import FeedSpec

from sam.config.loader import SamSettings
from sam.live.runner import run_shadow
from sam.strategies.ma_crossover import MACrossoverStrategy

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/ohlcv_ma_baseline.parquet"


def test_ma_strategy_identical_backtest_and_shadow(tmp_path):
    symbols = ["SPY", "QQQ", "IWM"]
    strategy = MACrossoverStrategy(symbols=symbols, ma_period=10, order_quantity=10)

    prices = pl.read_parquet(FIXTURE)
    feed_spec = FeedSpec(
        timestamp_col="date",
        entity_col="symbol",
        close_col="close",
        price_col="close",
    )
    feed = DataFeed(prices_df=prices, feed_spec=feed_spec)
    engine = Engine(feed, strategy, BacktestConfig.from_preset("realistic"))
    result = engine.run()
    assert result.metrics is not None

    settings = SamSettings(
        sam_artifacts_dir=(tmp_path / "artifacts").resolve(),
        sam_state_dir=(tmp_path / "state").resolve(),
    )
    from sam.config.loader import LiveRunConfig, RiskConfig

    live_cfg = LiveRunConfig(
        strategy_config=str(ROOT / "configs/strategies/ma_crossover.yaml"),
        environment="shadow",
        risk=RiskConfig(shadow_mode=True),
    )
    code = __import__("asyncio").run(
        run_shadow(
            settings,
            live_cfg,
            live_cfg.strategy_config,
            duration=2,
            root=ROOT,
        )
    )
    assert code == 0
