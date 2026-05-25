from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SamSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    sam_data_dir: Path = Field(default=Path("./data"))
    sam_artifacts_dir: Path = Field(default=Path("./artifacts"))
    sam_state_dir: Path = Field(default=Path("./state"))
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None


class RiskConfig(BaseModel):
    shadow_mode: bool = True
    max_position_value: float = 5_000.0
    max_order_value: float = 1_000.0
    max_positions: int = 3
    max_position_shares: int = 1_000
    max_total_exposure: float = 25_000.0
    max_order_shares: int = 500
    max_orders_per_minute: int = 10
    max_daily_loss: float = 1_000.0
    max_drawdown_pct: float = 0.05
    max_price_deviation_pct: float = 0.05
    max_data_staleness_seconds: float = 120.0
    dedup_window_seconds: float = 1.0
    kill_switch_enabled: bool = True
    fail_on_reconciliation_mismatch: bool = False


class StrategyConfig(BaseModel):
    strategy: str
    symbols: list[str] = Field(default_factory=list)
    universe_file: str | None = None
    ma_period: int = 10
    order_quantity: float = 10
    signal_column: str = "score"
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    top_k: int = 3


class QualityGatesConfig(BaseModel):
    min_completeness: float = 0.95
    max_errors: int = 5
    max_critical_anomalies: int = 0
    max_staleness_minutes: float = 24 * 60.0
    enforce: bool = True


class ModelConfig(BaseModel):
    model_type: str = "linear_feature_portfolio"
    batch_type: Literal["cross_section", "persistent_panel"] = "persistent_panel"
    train_fraction: float = 0.7
    ridge_alpha: float = 1e-4
    fit_intercept: bool = True
    gross_exposure: float = 1.0
    net_exposure: float | None = None
    max_abs_weight: float | None = None


class PromotionGatesConfig(BaseModel):
    min_trades: int = 0
    max_drawdown: float | None = None
    min_total_return: float | None = None
    min_dsr_probability: float | None = None
    n_trials: int = 1
    enforce: bool = True


class BacktestRunConfig(BaseModel):
    strategy_config: str
    prices_path: str
    signals_path: str | None = None
    preset: str = "realistic"
    initial_cash: float = 100_000.0
    output_dir: str = "artifacts/backtest/run"
    calendar: str = "NYSE"
    timezone: str = "America/New_York"
    data_frequency: str = "daily"
    commission_rate: float = 0.002
    slippage_rate: float = 0.002
    start_date: str | None = None
    end_date: str | None = None
    corporate_actions: str = "provider_adjusted"
    signal_lag_bars: int = 1
    min_trades: int = 0
    max_drawdown: float | None = None
    min_total_return: float | None = None
    enforce_promotion_gates: bool = True
    promotion_gates: PromotionGatesConfig = Field(default_factory=PromotionGatesConfig)


class LiveRunConfig(BaseModel):
    strategy_config: str
    environment: str = "paper"
    risk: RiskConfig = Field(default_factory=RiskConfig)
    state_file: str = "state/sam_risk.json"
    duration_seconds: int | None = None
    ib_port: int = 7497
    broker: str = "alpaca"
    live_config_path: str | None = None
    environment_config_path: str | None = None
    require_shadow_before_paper: bool = True
    require_paper_before_live: bool = True


class FeedSpecConfig(BaseModel):
    timestamp_col: str = "date"
    entity_col: str = "symbol"
    close_col: str = "close"
    price_col: str = "close"
    calendar: str | None = "NYSE"
    timezone: str | None = "America/New_York"
    data_frequency: str | None = "daily"


class DataSyncConfig(BaseModel):
    provider: str = "yahoo"
    start_date: str = "2020-01-01"
    end_date: str | None = None
    universe_file: str = "configs/universes/ma_baseline.txt"
    output_subdir: str = "raw/default"
    storage_mode: Literal["flat", "hive"] = "flat"
    incremental: bool = False
    feed_spec: FeedSpecConfig = Field(default_factory=FeedSpecConfig)
    quality_gates: QualityGatesConfig = Field(default_factory=QualityGatesConfig)


class DataValidateConfig(BaseModel):
    data_dir: str | None = None
    provider: str = "local"
    universe_file: str | None = None
    feed_spec: FeedSpecConfig = Field(default_factory=FeedSpecConfig)
    quality_gates: QualityGatesConfig = Field(default_factory=QualityGatesConfig)


class ResearchTrainConfig(BaseModel):
    prices_path: str
    features_path: str | None = None
    output_dir: str = "artifacts/research/signals"
    strategy_id: str = "signal_rank"
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {p}")
    return data


def config_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_universe(path: str | Path) -> list[str]:
    symbols: list[str] = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            symbols.append(s)
    return symbols


def resolve_symbols(strategy: StrategyConfig, root: Path | None = None) -> list[str]:
    if strategy.symbols:
        return list(strategy.symbols)
    if strategy.universe_file:
        uni = Path(strategy.universe_file)
        if root and not uni.is_absolute():
            uni = root / uni
        return load_universe(uni)
    return []


def ensure_dirs(settings: SamSettings) -> None:
    settings.sam_data_dir.mkdir(parents=True, exist_ok=True)
    settings.sam_artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.sam_state_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
