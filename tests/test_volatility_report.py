from __future__ import annotations

from pathlib import Path

import pandas as pd

from sam.cli import main
from sam.volatility_experiment import run_volatility_regime_experiment
from sam.volatility_report import (
    build_volatility_market_snapshot,
    render_volatility_report_html,
    render_volatility_report_markdown,
    write_volatility_regime_report,
)


def test_write_volatility_regime_report_from_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_volatility_regime_experiment(
        out_dir=run_dir,
        use_synthetic_data=True,
        skip_tfm=True,
        skip_xgboost=True,
        fast_mode=True,
        write_bundle=False,
        write_figures=False,
        write_report=False,
        show_progress=False,
    )
    snapshot = {"as_of": "2020-01-01", "risk_level": "watch", "vol_to_threshold": 0.8}
    result = write_volatility_regime_report(run_dir, snapshot=snapshot, open_browser_hint=False)
    assert result.markdown_path.is_file()
    assert result.html_path.is_file()
    assert (run_dir / "report_manifest.json").is_file()
    assert (run_dir.parent / "latest" / "report.html").is_file()
    md = result.markdown_path.read_text(encoding="utf-8")
    assert "Holdout model leaderboard" in md
    assert "How to use this day to day" in md


def test_html_report_styled_structure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    holdout = pd.DataFrame(
        {
            "Model": ["Rule[VIX close]", "XGBoost[Raw all-history incumbent]"],
            "Family": ["domain_rule", "classical_incumbent"],
            "Rows": [100, 100],
            "High Vol Rate": [0.2, 0.2],
            "average_precision": [0.631, 0.601],
            "roc_auc": [0.83, 0.78],
            "top_10pct_precision": [0.7, 0.75],
            "top_10pct_recall": [0.3, 0.33],
            "Error": [None, None],
        }
    )
    holdout.to_csv(run_dir / "publication_holdout_summary.csv", index=False)
    snapshot = {
        "as_of": "2026-05-21",
        "risk_level": "normal",
        "realized_vol_20d": 0.11,
        "threshold": 0.2,
        "vol_to_threshold": 0.53,
        "vix_close": 24.0,
        "vix_zscore_252d": 1.2,
        "interpretation": "SPY 20d realized vol is in **normal** regime.",
    }
    doc = render_volatility_report_html(run_dir, bundle_dir=tmp_path, snapshot=snapshot)
    assert "risk-badge" in doc
    assert "row-best" in doc
    assert "figure-grid" in doc
    assert "**" not in doc
    assert "<ul class=\"usage-list\">" in doc


def test_render_markdown_includes_holdout_table(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    holdout = pd.DataFrame(
        {
            "Model": ["VIX close"],
            "Family": ["domain_rule"],
            "Rows": [100],
            "High Vol Rate": [0.2],
            "average_precision": [0.63],
            "roc_auc": [0.7],
            "top_10pct_precision": [0.5],
            "top_10pct_recall": [0.4],
            "Error": [None],
        }
    )
    holdout.to_csv(run_dir / "publication_holdout_summary.csv", index=False)
    md = render_volatility_report_markdown(run_dir, bundle_dir=tmp_path, snapshot={})
    assert "VIX close" in md
    assert "0.630" in md or "0.63" in md


def test_build_market_snapshot_from_fixture_cache(tmp_path: Path) -> None:
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "volatility_regime"
    prices = pd.read_csv(fixture_root / "prices.csv", parse_dates=["date"])
    vix = pd.read_csv(fixture_root / "vix.csv", parse_dates=["date"])
    cache = tmp_path / "cache"
    cache.mkdir()
    prices.to_parquet(cache / "prices.parquet", index=False)
    vix.to_parquet(cache / "vix.parquet", index=False)
    snapshot = build_volatility_market_snapshot(cache_dir=cache)
    assert snapshot["risk_level"] in {"normal", "watch", "elevated", "unknown"}
    assert "interpretation" in snapshot
    assert snapshot["symbol"] == "SPY"


def test_cli_risk_snapshot_and_experiment_report(tmp_path: Path) -> None:
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "volatility_regime"
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.read_csv(fixture_root / "prices.csv", parse_dates=["date"]).to_parquet(
        cache / "prices.parquet",
        index=False,
    )
    pd.read_csv(fixture_root / "vix.csv", parse_dates=["date"]).to_parquet(
        cache / "vix.parquet",
        index=False,
    )
    snap_out = tmp_path / "snapshot.json"
    assert (
        main(
            [
                "risk",
                "snapshot",
                "--cache-dir",
                str(cache),
                "--out",
                str(snap_out),
            ]
        )
        == 0
    )
    assert snap_out.is_file()

    run_dir = tmp_path / "run"
    run_volatility_regime_experiment(
        out_dir=run_dir,
        use_synthetic_data=True,
        skip_tfm=True,
        skip_xgboost=True,
        fast_mode=True,
        write_bundle=False,
        write_figures=False,
        write_report=False,
        show_progress=False,
    )
    assert (
        main(
            [
                "experiment",
                "report",
                "volatility-regime-scoring",
                "--run-dir",
                str(run_dir),
                "--cache-dir",
                str(cache),
            ]
        )
        == 0
    )
    assert (run_dir / "report.html").is_file()
