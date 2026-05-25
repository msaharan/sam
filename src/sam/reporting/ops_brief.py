from __future__ import annotations

import json
from pathlib import Path

from sam.config.loader import SamSettings, write_json
from sam.ops.status import collect_ops_status


def build_ops_brief(settings: SamSettings) -> dict:
    brief: dict = {
        "data_dir": str(settings.sam_data_dir),
        "artifacts_dir": str(settings.sam_artifacts_dir),
        "state_dir": str(settings.sam_state_dir),
        "ops_status": collect_ops_status(settings),
    }
    sync_manifest = settings.sam_data_dir / "raw/ma_baseline/sync_manifest.json"
    if sync_manifest.exists():
        brief["last_data_sync"] = json.loads(sync_manifest.read_text())
    return brief


def write_ops_brief(settings: SamSettings, fmt: str = "md") -> Path:
    payload = build_ops_brief(settings)
    out_dir = settings.sam_artifacts_dir / "ops"
    if fmt == "json":
        path = out_dir / "brief.json"
        write_json(path, payload)
        return path
    status = payload["ops_status"]
    freshness = status.get("data_freshness", {})
    latest = status.get("latest_manifest") or {}
    lines = [
        "# SAM Operator Brief",
        "",
        f"- **kill_switch_active**: `{status.get('kill_switch_active')}`",
        f"- **latest_manifest**: `{latest.get('path')}`",
        f"- **latest_state**: `{latest.get('promotion_state')}`",
        f"- **latest_strategy**: `{latest.get('strategy_id')}`",
        f"- **latest_bar_timestamp**: `{freshness.get('latest_bar_timestamp')}`",
        f"- **data_staleness_seconds**: `{freshness.get('staleness_seconds')}`",
        f"- **risk_files**: `{len(status.get('risk_files', []))}`",
        f"- **run_manifests**: `{len(status.get('run_manifests', []))}`",
    ]
    path = out_dir / "brief.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
