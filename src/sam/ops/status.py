from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from sam.config.loader import LiveRunConfig, SamSettings


def _read_risk_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def collect_ops_status(
    settings: SamSettings,
    *,
    include_broker: bool = False,
    live_config: LiveRunConfig | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Aggregate risk state, kill-switch flags, and run manifests."""
    status: dict[str, Any] = {
        "state_dir": str(settings.sam_state_dir),
        "data_dir": str(settings.sam_data_dir),
        "artifacts_dir": str(settings.sam_artifacts_dir),
        "risk_files": [],
        "run_manifests": [],
        "latest_manifest": None,
        "data_freshness": collect_data_freshness(settings),
        "kill_switch_active": False,
        "broker_snapshot": {"enabled": include_broker, "available": False},
    }

    for path in sorted(settings.sam_state_dir.glob("*_risk.json")):
        payload = _read_risk_state(path)
        entry = {
            "file": str(path),
            "kill_switch_activated": payload.get("kill_switch_activated", False),
            "kill_switch_reason": payload.get("kill_switch_reason", ""),
            "daily_loss": payload.get("daily_loss"),
            "orders_placed": payload.get("orders_placed"),
            "date": payload.get("date"),
        }
        status["risk_files"].append(entry)
        if entry["kill_switch_activated"]:
            status["kill_switch_active"] = True

    manifests: list[dict[str, Any]] = []
    for manifest in sorted(settings.sam_artifacts_dir.rglob("run_manifest.json")):
        try:
            data = json.loads(manifest.read_text())
            entry = {
                "path": str(manifest),
                "strategy_id": data.get("strategy_id"),
                "promotion_state": data.get("promotion_state"),
                "environment": data.get("environment"),
                "broker": data.get("broker"),
                "config_hash": data.get("config_hash"),
                "created_at": data.get("created_at"),
                "checks": data.get("checks", {}),
            }
            manifests.append(entry)
        except (json.JSONDecodeError, OSError):
            continue
    status["run_manifests"] = manifests
    if manifests:
        status["latest_manifest"] = max(manifests, key=lambda item: item.get("created_at") or "")

    journal_files = list(settings.sam_state_dir.glob("*-journal.jsonl"))
    status["journal_files"] = [str(p) for p in journal_files]
    if include_broker and live_config is not None:
        try:
            status["broker_snapshot"] = asyncio.run(
                collect_broker_snapshot(settings, live_config, root or Path.cwd())
            )
        except RuntimeError as exc:
            status["broker_snapshot"] = {
                "enabled": True,
                "available": False,
                "error": str(exc),
            }
    return status


def collect_data_freshness(settings: SamSettings) -> dict[str, Any]:
    manifests = sorted(settings.sam_data_dir.rglob("sync_manifest.json"))
    payload: dict[str, Any] = {
        "sync_manifests": [str(p) for p in manifests],
        "latest_sync_manifest": str(manifests[-1]) if manifests else None,
        "latest_bar_timestamp": None,
        "latest_file_mtime": None,
        "staleness_seconds": None,
    }
    parquet_files = sorted(settings.sam_data_dir.rglob("*.parquet"))
    if not parquet_files:
        return payload
    latest_mtime = max(p.stat().st_mtime for p in parquet_files)
    payload["latest_file_mtime"] = datetime.fromtimestamp(latest_mtime, UTC).isoformat()
    payload["staleness_seconds"] = max(0.0, datetime.now(UTC).timestamp() - latest_mtime)

    latest_bar: Any = None
    for path in parquet_files:
        try:
            frame = pl.read_parquet(path, columns=["date"])
            value = frame["date"].max()
            if value is not None and (latest_bar is None or value > latest_bar):
                latest_bar = value
        except Exception:
            continue
    if latest_bar is not None:
        payload["latest_bar_timestamp"] = str(latest_bar)
    return payload


async def collect_broker_snapshot(
    settings: SamSettings,
    live_config: LiveRunConfig,
    root: Path,
) -> dict[str, Any]:
    from sam.live.alpaca import build_alpaca_session
    from sam.strategies.registry import build_strategy

    strategy = build_strategy(live_config.strategy_config, root=root)
    engine, safe = build_alpaca_session(settings, live_config, strategy, strategy.symbols)
    await engine.connect()
    try:
        positions = await safe.get_positions_async()
        pending_orders = await safe.get_pending_orders_async()
        return {
            "enabled": True,
            "available": True,
            "environment": live_config.environment,
            "broker": live_config.broker,
            "account_value": await safe.get_account_value_async(),
            "cash": await safe.get_cash_async(),
            "positions": {symbol: _jsonable(value) for symbol, value in positions.items()},
            "pending_orders": [_jsonable(order) for order in pending_orders],
        }
    finally:
        await engine.stop()


def set_kill_switch_state(
    state_file: Path,
    active: bool,
    reason: str = "operator",
) -> dict[str, Any]:
    """Activate or clear the kill switch via ML4T RiskState persistence."""
    from ml4t.live.safety import RiskState

    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = RiskState.load(str(state_file)) or RiskState.create_for_today()
    if active:
        state.kill_switch_activated = True
        state.kill_switch_reason = reason
    else:
        state.kill_switch_activated = False
        state.kill_switch_reason = ""
    RiskState.save_atomic(state, str(state_file))
    payload = state.to_dict()
    payload["updated_at"] = datetime.now(UTC).isoformat()
    return payload


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value
