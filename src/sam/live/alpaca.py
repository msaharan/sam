from __future__ import annotations

import os

from ml4t.live import AlpacaBroker, AlpacaDataFeed, LiveRiskConfig, SafeBroker
from ml4t.live.engine import LiveEngine

from sam.config.loader import LiveRunConfig, RiskConfig, SamSettings


def risk_from_config(risk: RiskConfig, state_file: str) -> LiveRiskConfig:
    return LiveRiskConfig(
        shadow_mode=risk.shadow_mode,
        max_position_value=risk.max_position_value,
        max_position_shares=risk.max_position_shares,
        max_total_exposure=risk.max_total_exposure,
        max_order_value=risk.max_order_value,
        max_order_shares=risk.max_order_shares,
        max_orders_per_minute=risk.max_orders_per_minute,
        max_daily_loss=risk.max_daily_loss,
        max_drawdown_pct=risk.max_drawdown_pct,
        max_price_deviation_pct=risk.max_price_deviation_pct,
        max_data_staleness_seconds=risk.max_data_staleness_seconds,
        dedup_window_seconds=risk.dedup_window_seconds,
        max_positions=risk.max_positions,
        kill_switch_enabled=risk.kill_switch_enabled,
        fail_on_reconciliation_mismatch=risk.fail_on_reconciliation_mismatch,
        state_file=state_file,
        journal_file=state_file.replace(".json", "-journal.jsonl"),
    )


def build_alpaca_session(
    settings: SamSettings,
    live_config: LiveRunConfig,
    strategy,
    symbols: list[str],
) -> tuple[LiveEngine, SafeBroker]:
    api_key = settings.alpaca_api_key or os.environ.get("ALPACA_API_KEY")
    secret_key = settings.alpaca_secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY for paper/live trading.")

    paper = live_config.environment != "live"
    broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=paper)
    feed = AlpacaDataFeed(
        api_key=api_key,
        secret_key=secret_key,
        symbols=symbols,
        data_type="bars",
        feed="iex",
    )
    safe = SafeBroker(broker, risk_from_config(live_config.risk, live_config.state_file))
    engine = LiveEngine(strategy, safe, feed)
    return engine, safe
