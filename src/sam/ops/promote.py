from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sam.manifests.run_manifest import PromotionState, RunManifest

PROMOTION_ORDER: tuple[PromotionState, ...] = (
    PromotionState.RESEARCH,
    PromotionState.BACKTEST_PASSED,
    PromotionState.SHADOW,
    PromotionState.PAPER,
    PromotionState.LIVE,
)


def validate_transition(from_state: PromotionState, to_state: PromotionState) -> None:
    try:
        from_idx = PROMOTION_ORDER.index(from_state)
        to_idx = PROMOTION_ORDER.index(to_state)
    except ValueError as exc:
        raise ValueError(f"Unknown promotion state: {from_state} -> {to_state}") from exc
    if to_idx != from_idx + 1:
        next_stage = (
            PROMOTION_ORDER[from_idx + 1].value if from_idx + 1 < len(PROMOTION_ORDER) else "none"
        )
        raise ValueError(
            f"Invalid promotion transition {from_state.value} -> {to_state.value}; "
            f"expected next stage {next_stage}"
        )


def promotion_checks_passed(manifest: RunManifest) -> bool:
    checks = manifest.checks or {}
    if isinstance(checks.get("passed"), bool):
        return checks["passed"]
    gates = manifest.promotion_decision or {}
    if isinstance(gates.get("passed"), bool):
        return gates["passed"]
    return manifest.promotion_state != PromotionState.RESEARCH


def promote_manifest(
    manifest_path: Path,
    *,
    from_state: PromotionState,
    to_state: PromotionState,
    reason: str = "operator",
    force: bool = False,
) -> dict[str, Any]:
    manifest = RunManifest.load(manifest_path)
    if manifest.promotion_state != from_state:
        raise RuntimeError(
            f"Manifest state is {manifest.promotion_state.value}, expected {from_state.value}"
        )
    validate_transition(from_state, to_state)
    if not force and not promotion_checks_passed(manifest):
        raise RuntimeError(
            f"Promotion gates failed for {manifest_path}: {manifest.checks.get('failures', [])}"
        )

    decision = {
        "from_state": from_state.value,
        "to_state": to_state.value,
        "reason": reason,
        "approved_at": datetime.now(tz=UTC).isoformat(),
        "passed": True,
        "manifest_path": str(manifest_path),
        "dependency_versions": manifest.dependency_versions,
        "quality_reports": manifest.quality_reports,
        "model_artifacts": manifest.model_artifacts,
        "diagnostic_reports": manifest.diagnostic_reports,
        "ml4t_artifacts": manifest.ml4t_artifacts,
    }
    manifest.promotion_state = to_state
    manifest.promotion_decision = decision
    manifest.save(manifest_path)
    return decision
