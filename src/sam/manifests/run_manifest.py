from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sam.config.loader import write_json


class PromotionState(StrEnum):
    RESEARCH = "research"
    BACKTEST_PASSED = "backtest_passed"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class RunManifest(BaseModel):
    run_id: str = Field(default_factory=lambda: datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    strategy_id: str
    config_path: str
    config_hash: str
    promotion_state: PromotionState = PromotionState.RESEARCH
    environment: str | None = None
    broker: str | None = None
    data_window_start: str | None = None
    data_window_end: str | None = None
    inputs: dict[str, str] = Field(default_factory=dict)
    config_hashes: dict[str, str] = Field(default_factory=dict)
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    risk_config: dict[str, Any] = Field(default_factory=dict)
    checks: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    ml4t_artifacts: dict[str, str] = Field(default_factory=dict)
    quality_reports: dict[str, str] = Field(default_factory=dict)
    model_artifacts: dict[str, str] = Field(default_factory=dict)
    diagnostic_reports: dict[str, str] = Field(default_factory=dict)
    promotion_decision: dict[str, Any] | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str | None = None

    def save(self, path: Path) -> Path:
        write_json(path, self.model_dump(mode="json"))
        return path

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        import json

        data: dict[str, Any] = json.loads(path.read_text())
        return cls.model_validate(data)


def installed_dependency_versions() -> dict[str, str]:
    packages = [
        "sam",
        "ml4t-data",
        "ml4t-engineer",
        "ml4t-diagnostic",
        "ml4t-models",
        "ml4t-backtest",
        "ml4t-live",
        "ml4t-specs",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions
