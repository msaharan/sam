"""Configuration models and repository defaults."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Market = Literal["us"]
PortfolioMethod = Literal[
    "equal_weight",
    "inverse_volatility",
    "minimum_variance",
    "risk_parity",
    "momentum_tilt",
    "etf_baseline",
]


DEFAULT_ETFS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "GLD",
    "SLV",
    "DBC",
    "VNQ",
    "IYR",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
]

DEFAULT_US_EQUITIES = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "BRK-B",
    "LLY",
    "JPM",
    "V",
]

class MarketProfile(BaseModel):
    id: Market
    label: str
    currency: str
    benchmark: str
    default_universe: list[str]


class UniverseConfig(BaseModel):
    name: str
    market: Market = "us"
    symbols: list[str] = Field(default_factory=list)
    include_default_etfs: bool = False

    @field_validator("symbols")
    @classmethod
    def require_symbols(cls, symbols: list[str]) -> list[str]:
        if not symbols:
            raise ValueError("universe must include at least one symbol")
        return symbols

    @property
    def normalized_symbols(self) -> list[str]:
        symbols = [normalize_ticker(symbol, self.market) for symbol in self.symbols]
        if self.include_default_etfs:
            symbols.extend(DEFAULT_ETFS)
        return sorted(dict.fromkeys(symbols))


class PortfolioConfig(BaseModel):
    name: str = "sam-default"
    method: PortfolioMethod = "inverse_volatility"
    market: Market = "us"
    max_weight: float = Field(default=0.2, gt=0, le=1)
    min_history_days: int = Field(default=126, ge=20)
    lookback_days: int = Field(default=252, ge=20)
    rebalance: Literal["M", "W", "Q"] = "M"
    transaction_cost_bps: float = Field(default=5.0, ge=0)
    risk_free_rate: float = Field(default=0.0, ge=0)
    benchmark: str | None = None

    @model_validator(mode="after")
    def default_market_benchmark(self) -> PortfolioConfig:
        if self.benchmark is None:
            self.benchmark = market_profile(self.market).benchmark
        return self


class BacktestConfig(BaseModel):
    name: str = "sam-backtest"
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    start: str | None = None
    end: str | None = None
    slippage_bps: float = Field(default=2.0, ge=0)


class MLConfig(BaseModel):
    name: str = "sam-ml-baseline"
    model: Literal[
        "ridge",
        "elastic_net",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "ensemble_rank",
    ] = "hist_gradient_boosting"
    task: Literal["regression", "classification"] = "regression"
    horizon_days: int = Field(default=21, ge=1)
    train_window_days: int = Field(default=504, ge=126)
    test_window_days: int = Field(default=63, ge=20)
    step_days: int = Field(default=63, ge=1)
    embargo_days: int = Field(default=0, ge=0)


class StressScenarioConfig(BaseModel):
    name: str
    kind: Literal["historical", "parametric"]
    start: str | None = None
    end: str | None = None
    equity_shock: float = 0.0
    cost_shock: float = 0.0
    vol_multiplier: float = Field(default=1.0, ge=0)


def market_profile(market: Market) -> MarketProfile:
    return MarketProfile(
        id="us",
        label="United States",
        currency="USD",
        benchmark="SPY",
        default_universe=DEFAULT_US_EQUITIES + ["SPY"],
    )


def normalize_ticker(symbol: str, market: Market = "us") -> str:
    return symbol.strip().upper()


def load_toml(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_universe(path: str | Path) -> UniverseConfig:
    return UniverseConfig.model_validate(load_toml(path))


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    return PortfolioConfig.model_validate(load_toml(path))


def load_backtest_config(path: str | Path) -> BacktestConfig:
    return BacktestConfig.model_validate(load_toml(path))


def load_ml_config(path: str | Path) -> MLConfig:
    return MLConfig.model_validate(load_toml(path))


def load_stress_scenarios(path: str | Path) -> list[dict]:
    payload = load_toml(path)
    scenarios = payload.get("scenario", [])
    return [StressScenarioConfig.model_validate(item).model_dump() for item in scenarios]
