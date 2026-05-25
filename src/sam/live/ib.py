from __future__ import annotations

from ml4t.live import IBBroker, IBDataFeed, LiveEngine, SafeBroker

from sam.config.loader import LiveRunConfig, SamSettings
from sam.live.alpaca import risk_from_config


def build_ib_session(
    settings: SamSettings,
    live_config: LiveRunConfig,
    strategy,
    symbols: list[str],
) -> tuple[LiveEngine, SafeBroker]:
    """Wire IB paper (7497) or live (7496) via ml4t-live."""
    ib_port = live_config.ib_port
    if live_config.environment == "live" and ib_port == 7497:
        ib_port = 7496

    broker = IBBroker(host="127.0.0.1", port=ib_port, client_id=77)
    feed = IBDataFeed(broker.ib, symbols=symbols, tick_throttle_ms=1_000)
    safe = SafeBroker(broker, risk_from_config(live_config.risk, live_config.state_file))
    engine = LiveEngine(strategy, safe, feed)
    return engine, safe
