from pathlib import Path

import pytest

from sam.config.loader import LiveRunConfig, RiskConfig, SamSettings
from sam.live.runner import run_shadow

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_signal_rank_shadow_completes(tmp_path):
    artifacts = (tmp_path / "artifacts").resolve()
    state_dir = (tmp_path / "state").resolve()
    settings = SamSettings(sam_artifacts_dir=artifacts, sam_state_dir=state_dir)
    strategy_config = str(ROOT / "configs/strategies/signal_rank.yaml")
    live_cfg = LiveRunConfig(
        strategy_config=strategy_config,
        environment="shadow",
        risk=RiskConfig(shadow_mode=True),
    )
    code = await run_shadow(settings, live_cfg, strategy_config, duration=2, root=ROOT)
    assert code == 0
    assert (artifacts / "live" / "shadow" / "run_manifest.json").exists()
