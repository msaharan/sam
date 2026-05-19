from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from sam.cli import main
from sam.config import MLConfig
from sam.daily import DailyBriefConfig, load_daily_brief_config, run_daily_brief
from sam.io import write_table
from sam.ml import train_model


def _trained_model(tmp_path: Path, sample_prices: pd.DataFrame) -> Path:
    result = train_model(
        sample_prices,
        config=MLConfig(
            model="ridge",
            horizon_days=5,
            train_window_days=150,
            test_window_days=30,
            step_days=60,
        ),
        out_dir=tmp_path / "model",
    )
    assert result.model_path is not None
    return result.model_path


def _daily_config(
    tmp_path: Path,
    sample_prices: pd.DataFrame,
    *,
    model_path: Path,
    price_path: Path | None = None,
) -> DailyBriefConfig:
    prices = price_path or tmp_path / "prices.parquet"
    if price_path is None:
        write_table(sample_prices, prices)
    return DailyBriefConfig(
        price_output=prices,
        refresh_data=False,
        as_of=str(pd.to_datetime(sample_prices["date"]).max().date()),
        model_path=model_path,
        model_dir=model_path.parent,
        train_if_missing=False,
        risk_threshold_end="2020-12-31",
        allocation_top_k=2,
        allocation_max_weight=0.6,
        report_root=tmp_path / "reports",
    )


def test_daily_config_loads_default_and_rejects_invalid_settings() -> None:
    config = load_daily_brief_config("configs/daily/default.toml")
    assert config.allocation_top_k == 5
    assert config.allocation_max_weight == 0.2

    with pytest.raises(ValidationError):
        DailyBriefConfig(allocation_top_k=3, allocation_max_weight=0.2)
    with pytest.raises(ValidationError):
        DailyBriefConfig(transaction_cost_bps=-1)


def test_daily_brief_writes_expected_files(tmp_path: Path, sample_prices: pd.DataFrame) -> None:
    model_path = _trained_model(tmp_path, sample_prices)
    result = run_daily_brief(_daily_config(tmp_path, sample_prices, model_path=model_path))
    expected = {
        "brief.md",
        "summary.json",
        "data_validation.csv",
        "risk_snapshot.csv",
        "allocation_scores.csv",
        "target_weights.csv",
        "turnover.csv",
        "manifest.json",
    }

    assert expected.issubset({path.name for path in result.out_dir.iterdir()})
    assert expected.issubset({path.name for path in result.latest_dir.iterdir()})
    assert result.summary["model_status"] == "reused"
    assert result.summary["top_selected_symbols"]
    assert result.summary["output_paths"]["brief"].endswith("brief.md")
    assert result.summary["output_paths"]["summary"].endswith("summary.json")
    assert "Research only. Not investment advice." in result.brief_markdown.read_text()


def test_daily_brief_reports_data_validation_failures(
    tmp_path: Path,
    sample_prices: pd.DataFrame,
) -> None:
    model_path = _trained_model(tmp_path, sample_prices)
    broken = sample_prices.copy()
    broken.loc[0, "adj_close"] = -1
    price_path = tmp_path / "broken.parquet"
    write_table(broken, price_path)

    result = run_daily_brief(
        _daily_config(tmp_path, broken, model_path=model_path, price_path=price_path)
    )
    brief = result.brief_markdown.read_text(encoding="utf-8")
    assert result.summary["validation_failures"] == 1
    assert "Validation failures: 1" in brief


def test_daily_brief_reuses_saved_model_by_default(
    tmp_path: Path,
    sample_prices: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _trained_model(tmp_path, sample_prices)

    def fail_train(*args, **kwargs):
        raise AssertionError("daily brief should reuse the saved model")

    monkeypatch.setattr("sam.daily.train_model", fail_train)
    result = run_daily_brief(_daily_config(tmp_path, sample_prices, model_path=model_path))
    assert result.summary["model_status"] == "reused"


def test_daily_brief_requires_model_when_training_disabled(
    tmp_path: Path,
    sample_prices: pd.DataFrame,
) -> None:
    config = _daily_config(tmp_path, sample_prices, model_path=tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError):
        run_daily_brief(config)


def test_daily_brief_retrains_when_requested(tmp_path: Path, sample_prices: pd.DataFrame) -> None:
    model_path = _trained_model(tmp_path, sample_prices)
    ml_config = tmp_path / "small-ml.toml"
    ml_config.write_text(
        "\n".join(
            [
                'name = "small-test"',
                'model = "ridge"',
                'task = "regression"',
                "horizon_days = 5",
                "train_window_days = 150",
                "test_window_days = 30",
                "step_days = 60",
            ]
        ),
        encoding="utf-8",
    )
    config = _daily_config(tmp_path, sample_prices, model_path=model_path).model_copy(
        update={"retrain": True, "model_dir": tmp_path / "retrained", "ml_config": ml_config}
    )
    result = run_daily_brief(config)
    assert result.summary["model_status"] == "retrained"
    assert (tmp_path / "retrained" / "model.pkl").is_file()


def test_daily_brief_turnover_uses_prior_latest_weights(
    tmp_path: Path,
    sample_prices: pd.DataFrame,
) -> None:
    model_path = _trained_model(tmp_path, sample_prices)
    config = _daily_config(tmp_path, sample_prices, model_path=model_path)
    first = run_daily_brief(config)

    scores = pd.read_csv(first.latest_dir / "allocation_scores.csv")
    previous = scores.tail(2).copy()
    previous["date"] = pd.to_datetime(previous["date"]).max() - pd.Timedelta(days=1)
    previous["weight"] = 0.5
    previous["score"] = previous["prediction"]
    previous[["date", "symbol", "weight", "score", "score_rank"]].to_csv(
        first.latest_dir / "target_weights.csv",
        index=False,
    )

    second = run_daily_brief(config)
    assert second.summary["turnover"] > 0


def test_daily_brief_cli_uses_mocked_fetch(
    tmp_path: Path,
    sample_prices: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _trained_model(tmp_path, sample_prices)
    universe_path = tmp_path / "universe.toml"
    universe_path.write_text(
        'name = "fixture"\nmarket = "us"\nsymbols = ["AAA", "BBB", "CCC", "SPY"]\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "daily.toml"
    config_path.write_text(
        f"""
universe = "{universe_path}"
start = "2020-01-01"
price_output = "{tmp_path / "fetched.parquet"}"
ml_config = "configs/ml/baseline.toml"
model_dir = "{model_path.parent}"
model_path = "{model_path}"
train_if_missing = false
risk_threshold_end = "2020-12-31"
allocation_top_k = 2
allocation_max_weight = 0.6
report_root = "{tmp_path / "reports"}"
""",
        encoding="utf-8",
    )

    def fake_fetch(symbols, *, start, end=None, market="us", out=None):
        frame = sample_prices[sample_prices["symbol"].isin(symbols)].copy()
        if out is not None:
            write_table(frame, out)
        return frame

    monkeypatch.setattr("sam.daily.fetch_prices", fake_fetch)
    assert (
        main(["daily", "brief", "--config", str(config_path), "--out", str(tmp_path / "out")])
        == 0
    )
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_status"] == "reused"


def test_daily_brief_markdown_avoids_automation_language(
    tmp_path: Path,
    sample_prices: pd.DataFrame,
) -> None:
    model_path = _trained_model(tmp_path, sample_prices)
    result = run_daily_brief(_daily_config(tmp_path, sample_prices, model_path=model_path))
    brief = result.brief_markdown.read_text(encoding="utf-8").lower()
    assert "research only. not investment advice." in brief
    assert "live signal" not in brief
    assert "broker" not in brief
    assert "order" not in brief
    assert "trade" not in brief
