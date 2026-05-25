from pathlib import Path

from sam.config.loader import (
    BacktestRunConfig,
    DataSyncConfig,
    PromotionGatesConfig,
    StrategyConfig,
    config_hash,
    load_yaml,
)


def test_strategy_config_loads():
    raw = load_yaml("configs/strategies/ma_crossover.yaml")
    cfg = StrategyConfig.model_validate(raw)
    assert cfg.strategy == "ma_crossover"
    assert "SPY" in cfg.symbols


def test_config_hash_stable():
    h1 = config_hash("configs/strategies/ma_crossover.yaml")
    h2 = config_hash("configs/strategies/ma_crossover.yaml")
    assert h1 == h2


def test_data_config_includes_feed_spec():
    raw = load_yaml("configs/data/default.yaml")
    assert "feed_spec" in raw
    assert raw["feed_spec"]["entity_col"] == "symbol"


def test_backtest_config_loads():
    raw = load_yaml("configs/backtest/ma_baseline.yaml")
    cfg = BacktestRunConfig.model_validate(raw)
    assert Path(cfg.prices_path).name.endswith(".parquet")


def test_data_sync_config_storage_fields():
    raw = load_yaml("configs/data/default.yaml")
    cfg = DataSyncConfig.model_validate(raw)
    assert cfg.storage_mode == "flat"
    assert cfg.incremental is False
    assert cfg.quality_gates.enforce is True


def test_promotion_gates_dsr_fields():
    raw = load_yaml("configs/backtest/equity_signal_rank.yaml")
    cfg = BacktestRunConfig.model_validate(raw)
    assert cfg.promotion_gates.min_dsr_probability == 0.0
    assert cfg.promotion_gates.n_trials == 1
    assert PromotionGatesConfig(min_dsr_probability=0.5, n_trials=3).n_trials == 3
