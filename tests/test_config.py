from sam.config import (
    DEFAULT_ETFS,
    StressScenarioConfig,
    UniverseConfig,
    load_stress_scenarios,
    normalize_ticker,
)


def test_normalize_ticker_preserves_us_research_symbols() -> None:
    assert normalize_ticker("spy") == "SPY"
    assert normalize_ticker("brk-b") == "BRK-B"


def test_universe_default_etfs_are_deduplicated() -> None:
    universe = UniverseConfig(name="test", market="us", symbols=["SPY"], include_default_etfs=True)
    assert universe.normalized_symbols.count("SPY") == 1


def test_default_etfs_are_us_listed_research_assets() -> None:
    assert "SPY" in DEFAULT_ETFS
    assert len(DEFAULT_ETFS) >= 20
    assert not any(symbol.endswith(".NS") for symbol in DEFAULT_ETFS)


def test_stress_scenarios_do_not_include_legacy_currency_or_india_fields() -> None:
    scenarios = load_stress_scenarios("configs/stress/default_scenarios.toml")
    assert scenarios
    assert all("india_shock" not in scenario for scenario in scenarios)
    assert all("currency_shock" not in scenario for scenario in scenarios)
    assert "india_shock" not in StressScenarioConfig.model_fields
    assert "currency_shock" not in StressScenarioConfig.model_fields
