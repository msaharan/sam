"""Report summaries for SAM run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def summarize_run(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    summary: dict = {"run_dir": str(root), "files": sorted(path.name for path in root.glob("*"))}
    manifest = root / "manifest.json"
    if manifest.is_file():
        summary["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    metrics = root / "metrics.json"
    if metrics.is_file():
        summary["metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
    elif (root / "metrics.csv").is_file():
        summary["metrics"] = pd.read_csv(root / "metrics.csv").to_dict(orient="records")
    for name in ["config", "experiment", "metadata"]:
        path = root / f"{name}.json"
        if path.is_file():
            summary[name] = json.loads(path.read_text(encoding="utf-8"))
    for name in [
        "artifact_manifest",
        "calibration_diagnostics",
        "cost_turnover_sensitivity",
        "data_summary",
        "decision_translation",
        "figure_manifest",
        "model_leaderboard",
        "operating_points",
        "split_summary",
    ]:
        path = root / f"{name}.csv"
        if path.is_file():
            summary[name] = pd.read_csv(path).to_dict(orient="records")
    return summary
