from pathlib import Path

import pytest

from sam.config.loader import LiveRunConfig, RiskConfig, SamSettings
from sam.live.runner import run_shadow

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_shadow_session_completes(tmp_path):
    artifacts = (tmp_path / "artifacts").resolve()
    state_dir = (tmp_path / "state").resolve()
    settings = SamSettings(sam_artifacts_dir=artifacts, sam_state_dir=state_dir)
    strategy_config = str(ROOT / "configs/strategies/ma_crossover.yaml")
    live_cfg = LiveRunConfig(
        strategy_config=strategy_config,
        environment="shadow",
        risk=RiskConfig(shadow_mode=True),
    )
    code = await run_shadow(settings, live_cfg, strategy_config, duration=2, root=ROOT)
    assert code == 0
    manifest_path = artifacts / "live" / "shadow" / "run_manifest.json"
    assert manifest_path.exists(), f"expected manifest at {manifest_path}"
    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["broker"] == "fixture_replay"
    assert "virtual_positions" in manifest["checks"]
