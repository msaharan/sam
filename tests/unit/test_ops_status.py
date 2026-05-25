import json

from sam.config.loader import SamSettings
from sam.ops.status import collect_ops_status, set_kill_switch_state


def test_collect_ops_status_reads_kill_switch(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "paper_risk.json").write_text(
        json.dumps({"kill_switch_activated": True, "kill_switch_reason": "daily_loss"})
    )
    settings = SamSettings(sam_state_dir=state_dir, sam_artifacts_dir=tmp_path / "artifacts")
    status = collect_ops_status(settings)
    assert status["kill_switch_active"] is True
    assert status["risk_files"][0]["kill_switch_reason"] == "daily_loss"
    assert "data_freshness" in status


def test_set_kill_switch_state(tmp_path):
    state_file = tmp_path / "state" / "paper_risk.json"
    payload = set_kill_switch_state(state_file, True, "manual_test")
    assert payload["kill_switch_activated"] is True
    assert payload["kill_switch_reason"] == "manual_test"

    cleared = set_kill_switch_state(state_file, False)
    assert cleared["kill_switch_activated"] is False
