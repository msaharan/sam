from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sam.volatility_config import load_volatility_run_config
from sam.volatility_data import (
    DEFAULT_VOLATILITY_FIXTURE_DIR,
    FRED_SETUP_HINT,
    build_market_panel,
    fetch_vix_history,
    ingest_volatility_research_data,
    load_fred_from_csv,
    load_volatility_regime_fixtures,
    synthetic_volatility_fixtures,
)


def test_build_market_panel_aligns_vix_and_fred_on_spy_calendar() -> None:
    data = synthetic_volatility_fixtures(n_days=120, start="2020-01-01")
    panel = build_market_panel(
        prices=data["prices"],
        vix=data["vix"],
        fred=data["fred"],
    )
    assert len(panel) == data["prices"]["date"].nunique()
    assert panel["vix_close"].notna().all()
    assert panel["treasury_10y_yield"].notna().all()
    assert panel["date"].is_monotonic_increasing


def test_load_fred_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "fred.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="B"),
            "treasury_10y_yield": [4.0, 4.1, 4.2, 4.1, 4.0],
        }
    ).to_csv(path, index=False)
    frame = load_fred_from_csv(path)
    assert "treasury_10y_yield" in frame.columns
    assert len(frame) == 5


def test_volatility_run_config_loads() -> None:
    cfg = load_volatility_run_config("configs/risk/volatility_regime_run.toml")
    assert cfg.target_horizon_days == 20
    assert "covid_shock_2020" in cfg.holdout_regimes


def test_build_market_panel_has_no_future_dated_macro_rows() -> None:
    data = synthetic_volatility_fixtures(n_days=120, start="2020-01-01")
    panel = build_market_panel(
        prices=data["prices"],
        vix=data["vix"],
        fred=data["fred"],
    )
    assert (panel["date"] <= panel["date"].max()).all()
    assert panel["date"].is_monotonic_increasing


def test_load_volatility_regime_fixtures_from_disk() -> None:
    data = load_volatility_regime_fixtures(DEFAULT_VOLATILITY_FIXTURE_DIR)
    assert (DEFAULT_VOLATILITY_FIXTURE_DIR / "vix.csv").is_file()
    unique_dates = pd.to_datetime(data["prices"]["date"]).drop_duplicates()
    assert unique_dates.is_monotonic_increasing
    assert data["vix"]["date"].is_monotonic_increasing
    assert len(data["panel"]) == data["prices"]["date"].nunique()


def test_ingest_uses_cached_fred_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    data = synthetic_volatility_fixtures(n_days=60, start="2020-01-01")
    data["prices"].to_parquet(cache / "prices.parquet", index=False)
    data["vix"].to_parquet(cache / "vix.parquet", index=False)
    data["fred"].to_parquet(cache / "fred.parquet", index=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("fetch_fred_series should not be called when cache exists")

    monkeypatch.setattr("sam.volatility_data.fetch_fred_series", fail_fetch)
    monkeypatch.setattr("sam.volatility_data.fetch_prices", fail_fetch)
    monkeypatch.setattr("sam.volatility_data.fetch_vix_history", fail_fetch)

    result = ingest_volatility_research_data(cache_dir=cache)
    assert not result["fred"].empty
    assert "treasury_10y_yield" in result["fred"].columns


def test_ingest_without_fred_key_or_cache_raises_helpful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    data = synthetic_volatility_fixtures(n_days=60, start="2020-01-01")
    data["prices"].to_parquet(cache / "prices.parquet", index=False)
    data["vix"].to_parquet(cache / "vix.parquet", index=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(ValueError, match="FRED_API_KEY") as exc_info:
        ingest_volatility_research_data(cache_dir=cache)
    assert "--synthetic" in exc_info.value.args[0]
    assert FRED_SETUP_HINT in exc_info.value.args[0]


@pytest.mark.integration
def test_fetch_vix_history_live() -> None:
    frame = fetch_vix_history(start="2024-01-01", end="2024-03-01")
    assert not frame.empty
    assert frame["date"].is_monotonic_increasing
