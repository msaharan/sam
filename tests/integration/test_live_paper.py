import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_alpaca_paper_requires_credentials():
    """Manual integration: set ALPACA_API_KEY and ALPACA_SECRET_KEY to run."""
    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_SECRET_KEY"):
        pytest.skip("Alpaca credentials not configured")

    from pathlib import Path

    from sam.config.loader import LiveRunConfig, SamSettings
    from sam.live.runner import run_paper

    root = Path(__file__).resolve().parents[2]
    settings = SamSettings()
    live_cfg = LiveRunConfig.model_validate(
        {
            "strategy_config": "configs/strategies/ma_crossover.yaml",
            "environment": "paper",
            "state_file": "state/paper_integration_risk.json",
            "require_shadow_before_paper": False,
        }
    )
    code = await run_paper(
        settings,
        live_cfg,
        live_cfg.strategy_config,
        duration=10,
        root=root,
    )
    assert code == 0
