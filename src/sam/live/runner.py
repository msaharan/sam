from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ml4t.live import LiveEngine, SafeBroker

from sam.config.loader import LiveRunConfig, SamSettings, config_hash, ensure_dirs
from sam.live.alpaca import build_alpaca_session, risk_from_config
from sam.live.offline import OfflineBrokerStub, ParquetReplayFeed, default_fixture_path
from sam.logging import get_logger
from sam.manifests.run_manifest import (
    PromotionState,
    RunManifest,
    installed_dependency_versions,
)
from sam.strategies.registry import build_strategy

log = get_logger(__name__)


async def _stop_after(duration: int, engine: LiveEngine) -> None:
    await asyncio.sleep(duration)
    await engine.stop()


def _has_alpaca_credentials(settings: SamSettings) -> bool:
    return bool(
        (settings.alpaca_api_key or os.environ.get("ALPACA_API_KEY"))
        and (settings.alpaca_secret_key or os.environ.get("ALPACA_SECRET_KEY"))
    )


def _build_fixture_replay_session(
    live_config: LiveRunConfig,
    strategy,
    symbols: list[str],
    *,
    state_path: Path,
    max_bars: int,
    fixture_path: Path,
) -> tuple[LiveEngine, SafeBroker]:
    safe = SafeBroker(
        OfflineBrokerStub(),
        risk_from_config(live_config.risk, str(state_path)),
    )
    feed = ParquetReplayFeed(fixture_path, symbols, max_bars=max_bars)
    engine = LiveEngine(strategy, safe, feed)
    return engine, safe


async def run_shadow(
    settings: SamSettings,
    live_config: LiveRunConfig,
    strategy_config: str,
    duration: int,
    root: Path,
) -> int:
    ensure_dirs(settings)
    strategy = build_strategy(strategy_config, root=root)
    symbols = strategy.symbols
    live_config.risk.shadow_mode = True
    state_path = settings.sam_state_dir / "shadow_risk.json"
    live_config.state_file = str(state_path)

    broker_label = "fixture_replay"
    if _has_alpaca_credentials(settings):
        engine, safe = build_alpaca_session(settings, live_config, strategy, symbols)
        broker_label = "alpaca_shadow"
        log.info("live.shadow.start", broker=broker_label, symbols=symbols)
    else:
        engine, safe = _build_fixture_replay_session(
            live_config,
            strategy,
            symbols,
            state_path=state_path,
            max_bars=max(duration, 1),
            fixture_path=default_fixture_path(root),
        )
        log.info(
            "live.shadow.start",
            broker=broker_label,
            symbols=symbols,
            fixture=str(default_fixture_path(root)),
        )

    await engine.connect()
    stop_task = asyncio.create_task(_stop_after(duration, engine))
    try:
        await engine.run()
    finally:
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)
        await engine.stop()

    virtual_positions = {asset: position.quantity for asset, position in safe.positions.items()}

    manifest = RunManifest(
        strategy_id=strategy.__class__.__name__,
        config_path=strategy_config,
        config_hash=_hash_if_exists(strategy_config),
        promotion_state=PromotionState.SHADOW,
        environment="shadow",
        broker=broker_label,
        inputs={"strategy_config": strategy_config},
        config_hashes=_live_config_hashes(live_config, strategy_config),
        dependency_versions=installed_dependency_versions(),
        risk_config=live_config.risk.model_dump(mode="json"),
        checks={
            "shadow_mode": True,
            "duration_seconds": duration,
            "symbols": symbols,
            "virtual_positions": virtual_positions,
        },
        artifact_paths={"state": str(state_path)},
        ml4t_artifacts={"risk_state": str(state_path)},
    )
    out = settings.sam_artifacts_dir / "live" / "shadow"
    manifest.save(out / "run_manifest.json")
    log.info("live.shadow.complete", broker=broker_label, positions=virtual_positions)
    return 0


async def run_preview(
    settings: SamSettings,
    strategy_config: str,
    bars: int,
    root: Path,
) -> dict:
    """Dry-run preview through LiveEngine shadow mode and fixture replay."""
    ensure_dirs(settings)
    strategy = build_strategy(strategy_config, root=root)
    state_path = settings.sam_state_dir / "preview_risk.json"
    live_config = LiveRunConfig(
        strategy_config=strategy_config,
        environment="shadow",
        state_file=str(state_path),
    )
    live_config.risk.shadow_mode = True
    engine, safe = _build_fixture_replay_session(
        live_config,
        strategy,
        strategy.symbols,
        state_path=state_path,
        max_bars=bars,
        fixture_path=default_fixture_path(root),
    )
    await engine.connect()
    try:
        await engine.run()
    finally:
        await engine.stop()

    return {
        "bars": bars,
        "symbols": strategy.symbols,
        "virtual_positions": {
            asset: position.quantity for asset, position in safe.positions.items()
        },
        "broker": "fixture_replay",
    }


async def run_broker_session(
    settings: SamSettings,
    live_config: LiveRunConfig,
    strategy_config: str,
    duration: int | None,
    root: Path,
    *,
    broker: str = "alpaca",
) -> int:
    """Run paper or live session on Alpaca or Interactive Brokers."""
    ensure_dirs(settings)
    _validate_live_promotion(settings, live_config)
    strategy = build_strategy(strategy_config, root=root)
    if broker == "ib":
        from sam.live.ib import build_ib_session

        engine, _safe = build_ib_session(settings, live_config, strategy, strategy.symbols)
    else:
        engine, _safe = build_alpaca_session(settings, live_config, strategy, strategy.symbols)

    await engine.connect()
    tasks = []
    if duration:
        tasks.append(asyncio.create_task(_stop_after(duration, engine)))

    try:
        await engine.run()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await engine.stop()

    state = PromotionState.LIVE if live_config.environment == "live" else PromotionState.PAPER
    manifest = RunManifest(
        strategy_id=strategy.__class__.__name__,
        config_path=strategy_config,
        config_hash=_hash_if_exists(strategy_config),
        promotion_state=state,
        environment=live_config.environment,
        broker=broker,
        inputs={"strategy_config": strategy_config},
        config_hashes=_live_config_hashes(live_config, strategy_config),
        dependency_versions=installed_dependency_versions(),
        risk_config=live_config.risk.model_dump(mode="json"),
        checks={"duration_seconds": duration, "symbols": strategy.symbols},
        artifact_paths={"state": live_config.state_file, "broker": broker},
        ml4t_artifacts={"risk_state": live_config.state_file},
    )
    out = settings.sam_artifacts_dir / "live" / live_config.environment
    manifest.save(out / "run_manifest.json")
    log.info("live.session.complete", environment=live_config.environment, broker=broker)
    return 0


async def run_paper(
    settings: SamSettings,
    live_config: LiveRunConfig,
    strategy_config: str,
    duration: int | None,
    root: Path,
) -> int:
    return await run_broker_session(
        settings, live_config, strategy_config, duration, root, broker="alpaca"
    )


async def run_live(
    settings: SamSettings,
    live_config: LiveRunConfig,
    strategy_config: str,
    duration: int | None,
    root: Path,
) -> int:
    live_config.environment = "live"
    live_config.risk.shadow_mode = False
    return await run_broker_session(
        settings, live_config, strategy_config, duration, root, broker="alpaca"
    )


async def run_ib_paper(
    settings: SamSettings,
    live_config: LiveRunConfig,
    strategy_config: str,
    duration: int | None,
    root: Path,
) -> int:
    return await run_broker_session(
        settings, live_config, strategy_config, duration, root, broker="ib"
    )


async def run_preflight(settings: SamSettings, live_config: LiveRunConfig, root: Path) -> dict:
    from sam.live.alpaca import build_alpaca_session
    from sam.strategies.registry import build_strategy

    strategy = build_strategy(live_config.strategy_config, root=root)
    engine, safe = build_alpaca_session(settings, live_config, strategy, strategy.symbols)
    await engine.connect()
    try:
        return await safe.preflight_async()
    finally:
        await engine.stop()


def _hash_if_exists(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    return config_hash(p) if p.exists() else ""


def _live_config_hashes(live_config: LiveRunConfig, strategy_config: str) -> dict[str, str]:
    hashes = {"strategy": _hash_if_exists(strategy_config)}
    if live_config.live_config_path:
        hashes["live"] = _hash_if_exists(live_config.live_config_path)
    if live_config.environment_config_path:
        hashes["environment"] = _hash_if_exists(live_config.environment_config_path)
    return {k: v for k, v in hashes.items() if v}


def _validate_live_promotion(settings: SamSettings, live_config: LiveRunConfig) -> None:
    if live_config.environment == "paper" and live_config.require_shadow_before_paper:
        shadow_manifest = settings.sam_artifacts_dir / "live" / "shadow" / "run_manifest.json"
        if not shadow_manifest.exists():
            raise RuntimeError("Run `sam live shadow` before `sam live paper`.")
    if live_config.environment == "live" and live_config.require_paper_before_live:
        paper_manifest = settings.sam_artifacts_dir / "live" / "paper" / "run_manifest.json"
        if not paper_manifest.exists():
            raise RuntimeError("Run `sam live paper` before `sam live live`.")
