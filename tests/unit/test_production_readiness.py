import json
from pathlib import Path

import pytest

from sam.config.loader import DataValidateConfig, QualityGatesConfig, SamSettings
from sam.manifests.run_manifest import PromotionState, RunManifest
from sam.ops.promote import promote_manifest, validate_transition


def test_quality_gates_config_defaults():
    cfg = QualityGatesConfig()
    assert cfg.min_completeness == 0.95
    assert cfg.max_errors == 5


def test_run_manifest_extended_fields():
    manifest = RunManifest(
        strategy_id="ma_crossover",
        config_path="configs/strategies/ma_crossover.yaml",
        config_hash="abc123",
        ml4t_artifacts={"risk_state": "state/shadow_risk.json"},
        quality_reports={"SPY": "data/quality/SPY.json"},
        model_artifacts={"model": "artifacts/research/model.json"},
        diagnostic_reports={"backtest": "artifacts/backtest/profile.json"},
    )
    payload = manifest.model_dump(mode="json")
    assert payload["quality_reports"]["SPY"] == "data/quality/SPY.json"
    assert payload["ml4t_artifacts"]["risk_state"] == "state/shadow_risk.json"


def test_validate_transition_rejects_skips():
    with pytest.raises(ValueError, match="Invalid promotion transition"):
        validate_transition(PromotionState.RESEARCH, PromotionState.SHADOW)


def test_promote_manifest_updates_state(tmp_path: Path):
    manifest_path = tmp_path / "run_manifest.json"
    manifest = RunManifest(
        strategy_id="ma_crossover",
        config_path="configs/strategies/ma_crossover.yaml",
        config_hash="abc123",
        promotion_state=PromotionState.RESEARCH,
        checks={"passed": True},
    )
    manifest.save(manifest_path)

    decision = promote_manifest(
        manifest_path,
        from_state=PromotionState.RESEARCH,
        to_state=PromotionState.BACKTEST_PASSED,
    )
    assert decision["to_state"] == "backtest_passed"
    updated = json.loads(manifest_path.read_text())
    assert updated["promotion_state"] == "backtest_passed"
    assert updated["promotion_decision"]["from_state"] == "research"


def test_data_validate_fixture_parquet(tmp_path: Path):
    pytest.importorskip("ml4t.diagnostic.integration")

    from sam.pipeline.data_sync import run_data_validate

    settings = SamSettings(sam_data_dir=tmp_path / "data")
    cfg = DataValidateConfig.model_validate(
        {
            "data_dir": "tests/fixtures",
            "provider": "fixture",
            "quality_gates": {"enforce": False},
        }
    )
    summary = run_data_validate(cfg, settings)
    assert summary["symbols"]
    assert summary["quality_reports"]
