"""Human-readable reports for volatility-regime research and daily monitoring."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sam.data import fetch_prices
from sam.risk import build_volatility_regime_frame
from sam.volatility_config import load_volatility_run_config


@dataclass(frozen=True)
class VolatilityReportResult:
    run_dir: Path
    markdown_path: Path
    html_path: Path
    snapshot: dict[str, Any]


def write_volatility_regime_report(
    run_dir: str | Path,
    *,
    bundle_dir: str | Path | None = None,
    snapshot: dict[str, Any] | None = None,
    open_browser_hint: bool = True,
) -> VolatilityReportResult:
    """Build consolidated Markdown and HTML reports from an experiment run directory."""

    root = Path(run_dir)
    bundle = Path(bundle_dir) if bundle_dir is not None else root.parent
    if snapshot is None:
        snapshot = {}

    markdown = render_volatility_report_markdown(root, bundle_dir=bundle, snapshot=snapshot)
    html_doc = render_volatility_report_html(root, bundle_dir=bundle, snapshot=snapshot)

    md_path = root / "report.md"
    html_path = root / "report.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_doc, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_dir": str(root.resolve()),
        "bundle_dir": str(bundle.resolve()) if bundle.is_dir() else None,
        "report_markdown": str(md_path.resolve()),
        "report_html": str(html_path.resolve()),
        "snapshot": snapshot,
    }
    (root / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    latest = root.parent / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "report.html").write_text(html_doc, encoding="utf-8")
    (latest / "report.md").write_text(markdown, encoding="utf-8")

    if open_browser_hint:
        print(f"Open report: {html_path.resolve()}")

    return VolatilityReportResult(
        run_dir=root,
        markdown_path=md_path,
        html_path=html_path,
        snapshot=snapshot,
    )


def build_volatility_market_snapshot(
    *,
    cache_dir: str | Path = "data/research/volatility_regime",
    run_config_path: str | Path = "configs/risk/volatility_regime_run.toml",
    symbol: str = "SPY",
) -> dict[str, Any]:
    """Lightweight today view: current RV vs historical threshold and VIX context."""

    run_cfg = load_volatility_run_config(run_config_path)
    cache = Path(cache_dir)
    prices_path = cache / "prices.parquet"

    if prices_path.is_file():
        prices = pd.read_parquet(prices_path)
    else:
        prices = fetch_prices([symbol, "^VIX"], start=run_cfg.data_start, market="us")

    threshold_end = run_cfg.threshold_source_end
    if threshold_end is not None and "date" in prices.columns:
        price_dates = pd.to_datetime(prices["date"], utc=False)
        if pd.Timestamp(threshold_end) < price_dates.min():
            threshold_end = None

    vol_frame = build_volatility_regime_frame(
        prices,
        symbol=symbol,
        horizon_days=run_cfg.target_horizon_days,
        threshold_quantile=run_cfg.high_vol_quantile,
        threshold_end=threshold_end,
        include_unlabeled=True,
    )
    col = f"realized_vol_{run_cfg.target_horizon_days}d"
    usable = vol_frame.frame.dropna(subset=[col])
    if usable.empty:
        raise ValueError("no volatility rows available for market snapshot")
    latest = usable.iloc[-1]
    threshold = float(latest["target_threshold"])
    current_rv = float(latest[col])
    ratio = current_rv / threshold if threshold > 0 else float("nan")

    vix_close = float("nan")
    vix_z = float("nan")
    if cache.joinpath("vix.parquet").is_file():
        vix = pd.read_parquet(cache / "vix.parquet")
        if not vix.empty:
            vix_close = float(vix["vix_close"].iloc[-1])
            mean = vix["vix_close"].rolling(252, min_periods=60).mean().iloc[-1]
            std = vix["vix_close"].rolling(252, min_periods=60).std(ddof=1).iloc[-1]
            if std and std > 0:
                vix_z = float((vix_close - mean) / std)

    return {
        "as_of": str(pd.to_datetime(latest["date"]).date()),
        "symbol": symbol,
        "risk_level": _risk_level(ratio),
        "realized_vol_20d": current_rv,
        "threshold": threshold,
        "vol_to_threshold": ratio,
        "vix_close": vix_close,
        "vix_zscore_252d": vix_z,
        "interpretation": _interpret_snapshot(ratio, vix_z),
    }


def render_volatility_report_markdown(
    run_dir: Path,
    *,
    bundle_dir: Path,
    snapshot: dict[str, Any],
) -> str:
    """Assemble a single Markdown document from run artifacts."""

    holdout = _read_csv(run_dir / "publication_holdout_summary.csv")
    operating = _read_csv(run_dir / "publication_operating_summary.csv")
    regimes = _read_csv(run_dir / "regime_holdout_summary.csv")
    bootstrap = _read_csv(run_dir / "publication_month_block_bootstrap_uncertainty.csv")
    splits = _read_csv(run_dir / "split_summary.csv")
    leakage = _read_csv(run_dir / "leakage_checks.csv")
    limitations_path = bundle_dir / "limitations.md"

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        "# Volatility Regime Scoring Report",
        "",
        f"_Generated {generated}. Research only — not investment advice._",
        "",
        "## How to use this day to day",
        "",
        _daily_use_section(snapshot),
        "",
        "## Executive summary",
        "",
        _executive_summary(holdout, snapshot),
        "",
        "## Holdout model leaderboard",
        "",
        _leaderboard_section(holdout),
        "",
        "## Alert operating points (holdout)",
        "",
        _df_to_markdown(operating.head(24)) if operating is not None else "_No operating summary._",
        "",
        "## Regime diagnostics (2020+)",
        "",
        _regime_section(regimes),
        "",
        "## Bootstrap uncertainty (AP)",
        "",
        _df_to_markdown(bootstrap) if bootstrap is not None else "_No bootstrap table._",
        "",
        "## Chronological splits",
        "",
        _df_to_markdown(splits) if splits is not None else "_No split summary._",
        "",
        "## Leakage and methodology checks",
        "",
        _df_to_markdown(leakage[["Check", "Status"]]) if leakage is not None else "_No leakage checks._",
        "",
        "## Figures",
        "",
        _figures_section(run_dir),
        "",
        "## Limitations",
        "",
        limitations_path.read_text(encoding="utf-8")
        if limitations_path.is_file()
        else "_See experiment contract limitations._",
        "",
        "## Artifact index",
        "",
        _artifact_index(run_dir),
    ]
    return "\n".join(sections)


def render_volatility_report_html(
    run_dir: Path,
    *,
    bundle_dir: Path,
    snapshot: dict[str, Any],
) -> str:
    """Build a styled HTML report from run artifacts (not markdown conversion)."""

    data = _load_report_artifacts(run_dir, bundle_dir)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    holdout = data["holdout"]
    best_model = None
    if holdout is not None and not holdout.empty and "Model" in holdout.columns:
        scored = holdout[holdout["Error"].isna() & holdout["average_precision"].notna()]
        if not scored.empty:
            best_model = str(scored.sort_values("average_precision", ascending=False).iloc[0]["Model"])

    sections: list[str] = []
    if snapshot:
        sections.append(_html_snapshot_hero(snapshot))

    sections.extend(
        [
            _html_section(
                "daily",
                "How to use this day to day",
                _html_daily_use(snapshot),
            ),
            _html_section(
                "summary",
                "Executive summary",
                _html_executive_summary(data["holdout"], snapshot),
            ),
            _html_section(
                "leaderboard",
                "Holdout model leaderboard",
                _html_dataframe_table(
                    _leaderboard_frame(data["holdout"]),
                    highlight_column="Model",
                    highlight_value=best_model,
                    compact=False,
                ),
            ),
            _html_section(
                "operating",
                "Alert operating points (holdout)",
                _html_dataframe_table(
                    data["operating"].head(24) if data["operating"] is not None else None,
                    compact=True,
                ),
            ),
            _html_section(
                "regimes",
                "Regime diagnostics (2020+)",
                _html_dataframe_table(_regime_highlights(data["regimes"]), compact=True),
            ),
            _html_section(
                "bootstrap",
                "Bootstrap uncertainty (AP)",
                _html_dataframe_table(data["bootstrap"], compact=True),
            ),
            _html_section(
                "splits",
                "Chronological splits",
                _html_dataframe_table(data["splits"], compact=False),
            ),
            _html_section(
                "leakage",
                "Leakage and methodology checks",
                _html_leakage_table(data["leakage"]),
            ),
            _html_figures_section(run_dir),
            _html_section(
                "limitations",
                "Limitations",
                _html_limitations(data["limitations_text"]),
            ),
            _html_section(
                "artifacts",
                "Artifact index",
                _html_artifact_index(run_dir),
            ),
        ]
    )

    toc = _html_toc(
        [
            ("daily", "Daily use"),
            ("summary", "Summary"),
            ("leaderboard", "Leaderboard"),
            ("operating", "Operating points"),
            ("regimes", "Regimes"),
            ("bootstrap", "Bootstrap"),
            ("splits", "Splits"),
            ("leakage", "Leakage checks"),
            ("figures", "Figures"),
            ("limitations", "Limitations"),
            ("artifacts", "Artifacts"),
        ]
    )

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>SAM · Volatility Regime Report</title>
  <style>{_REPORT_CSS}</style>
</head>
<body>
  <header class="site-header">
    <div class="site-header-inner">
      <span class="brand">SAM</span>
      <div>
        <h1 class="report-title">Volatility Regime Scoring</h1>
        <p class="report-meta">Generated {html.escape(generated)} · Research only — not investment advice</p>
      </div>
    </div>
  </header>
  <div class="layout">
    <nav class="toc" aria-label="Report sections">{toc}</nav>
    <main class="report-main">{body}</main>
  </div>
  <footer class="site-footer">
    <p>SAM experiment <code>volatility-regime-scoring</code> · Educational research software</p>
  </footer>
</body>
</html>
"""


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _risk_level(ratio: float) -> str:
    if pd.isna(ratio):
        return "unknown"
    if ratio >= 1.0:
        return "elevated"
    if ratio >= 0.75:
        return "watch"
    return "normal"


def _interpret_snapshot(ratio: float, vix_z: float) -> str:
    level = _risk_level(ratio)
    parts = [f"SPY 20d realized vol is in **{level}** regime vs the pre-2018 threshold."]
    if pd.notna(vix_z):
        if vix_z >= 1.5:
            parts.append("VIX is elevated vs its 252d history.")
        elif vix_z <= -1.0:
            parts.append("VIX is calm vs its 252d history.")
    parts.append(
        "Use this as a risk dial for sizing and hedging — not as a standalone trade signal."
    )
    return " ".join(parts)


def _daily_use_section(snapshot: dict[str, Any]) -> str:
    snap = ""
    if snapshot:
        snap = (
            f"\n**Latest snapshot ({snapshot.get('as_of', 'n/a')})**: "
            f"{snapshot.get('risk_level', 'n/a')} — "
            f"vol/threshold = {_format_cell(snapshot.get('vol_to_threshold'))}, "
            f"VIX z = {_format_cell(snapshot.get('vix_zscore_252d'))}.\n"
        )
    return (
        "- **Each morning**: `sam risk snapshot` (or `sam daily brief`) for a quick SPY/VIX risk dial.\n"
        "- **Weekly / after major moves**: re-run `sam experiment run volatility-regime-scoring` "
        "to refresh holdout metrics, figures, and this report.\n"
        "- **Portfolio work**: combine risk level with ETF allocation from `sam daily brief` — "
        "reduce risk when level is `elevated`, stay selective when `watch`.\n"
        "- **Full research pack**: open `report.html` in this run folder (figures + tables in one place)."
        f"{snap}"
    )


def _executive_summary(holdout: pd.DataFrame | None, snapshot: dict[str, Any]) -> str:
    if holdout is None or holdout.empty:
        return "_No holdout summary available._"
    scored = holdout[holdout["Error"].isna() & holdout["average_precision"].notna()].copy()
    if scored.empty:
        return "_No successful models in holdout summary._"
    best = scored.sort_values("average_precision", ascending=False).iloc[0]
    xgb = scored[scored["Model"].str.contains("XGBoost", na=False)]
    xgb_ap = _format_cell(xgb["average_precision"].iloc[0]) if not xgb.empty else "n/a"
    lines = [
        f"- Holdout rows: **{int(best['Rows'])}**; high-vol rate: **{float(best['High Vol Rate']):.1%}**.",
        f"- Best holdout AP: **{best['Model']}** ({float(best['average_precision']):.3f}).",
        f"- XGBoost incumbent AP: **{xgb_ap}**.",
        "- Simple VIX / realized-vol rules remain the benchmark — learned models must beat them on "
        "alert quality, not just AUC.",
    ]
    if snapshot:
        lines.append(
            f"- Today: **{snapshot.get('risk_level')}** "
            f"(vol/threshold { _format_cell(snapshot.get('vol_to_threshold')) })."
        )
    return "\n".join(lines)


def _leaderboard_frame(holdout: pd.DataFrame | None) -> pd.DataFrame | None:
    if holdout is None or holdout.empty:
        return None
    cols = [
        "Model",
        "Family",
        "average_precision",
        "roc_auc",
        "top_10pct_precision",
        "top_10pct_recall",
        "Error",
    ]
    present = [column for column in cols if column in holdout.columns]
    return holdout[present].copy()


def _leaderboard_section(holdout: pd.DataFrame | None) -> str:
    table = _leaderboard_frame(holdout)
    if table is None:
        return "_No holdout leaderboard._"
    for column in ["average_precision", "roc_auc", "top_10pct_precision", "top_10pct_recall"]:
        if column in table.columns:
            table[column] = table[column].map(lambda value: _format_cell(value, precision=3))
    return _df_to_markdown(table)


def _regime_highlights(regimes: pd.DataFrame | None) -> pd.DataFrame | None:
    if regimes is None or regimes.empty:
        return None
    if "average_precision" not in regimes.columns:
        return regimes.head(20)
    return regimes.sort_values("average_precision", ascending=False).head(8)


def _regime_section(regimes: pd.DataFrame | None) -> str:
    highlights = _regime_highlights(regimes)
    if highlights is None:
        return "_No regime table._"
    return _df_to_markdown(highlights)


def _figures_section(run_dir: Path) -> str:
    figures = sorted(run_dir.glob("*.png"))
    if not figures:
        return "_No PNG figures in run directory. Install `.[volatility]` and re-run without `--no-figures`._"
    return "\n".join(f"![{path.stem}]({path.name})" for path in figures)


def _artifact_index(run_dir: Path) -> str:
    rows = sorted(path.name for path in run_dir.iterdir() if path.is_file())
    return "\n".join(f"- `{name}`" for name in rows)


def _df_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.astype(object).where(pd.notna(frame), "")
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    body = [
        "| " + " | ".join(_format_cell(value) for value in row)
        for row in text.to_numpy()
    ]
    return "\n".join([header, sep, *body])


def _format_cell(value: Any, *, precision: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _load_report_artifacts(run_dir: Path, bundle_dir: Path) -> dict[str, Any]:
    limitations_path = bundle_dir / "limitations.md"
    return {
        "holdout": _read_csv(run_dir / "publication_holdout_summary.csv"),
        "operating": _read_csv(run_dir / "publication_operating_summary.csv"),
        "regimes": _read_csv(run_dir / "regime_holdout_summary.csv"),
        "bootstrap": _read_csv(run_dir / "publication_month_block_bootstrap_uncertainty.csv"),
        "splits": _read_csv(run_dir / "split_summary.csv"),
        "leakage": _read_csv(run_dir / "leakage_checks.csv"),
        "limitations_text": limitations_path.read_text(encoding="utf-8")
        if limitations_path.is_file()
        else "",
    }


_REPORT_CSS = """
:root {
  --bg: #f4f6f9;
  --surface: #ffffff;
  --text: #1a2332;
  --muted: #5c6b7a;
  --accent: #0d5c7a;
  --accent-soft: #e8f2f7;
  --border: #d8e0e8;
  --shadow: 0 2px 12px rgba(15, 35, 55, 0.08);
  --radius: 12px;
  --risk-normal: #1a7f4b;
  --risk-watch: #b8860b;
  --risk-elevated: #c0392b;
  --pass: #1a7f4b;
  --review: #b8860b;
  --fail: #c0392b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 15px;
  line-height: 1.55;
  color: var(--text);
  background: var(--bg);
}
.site-header {
  background: linear-gradient(135deg, #0a3d52 0%, #0d5c7a 55%, #14708f 100%);
  color: #fff;
  padding: 1.75rem 1.5rem;
  box-shadow: var(--shadow);
}
.site-header-inner {
  max-width: 1180px;
  margin: 0 auto;
  display: flex;
  align-items: flex-start;
  gap: 1.25rem;
}
.brand {
  font-weight: 700;
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  background: rgba(255,255,255,0.15);
  padding: 0.35rem 0.6rem;
  border-radius: 6px;
}
.report-title { margin: 0; font-size: 1.65rem; font-weight: 600; }
.report-meta { margin: 0.35rem 0 0; opacity: 0.88; font-size: 0.9rem; }
.layout {
  max-width: 1180px;
  margin: 0 auto;
  padding: 1.5rem;
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 1.75rem;
  align-items: start;
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .toc { position: static !important; }
}
.toc {
  position: sticky;
  top: 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
  font-size: 0.82rem;
  box-shadow: var(--shadow);
}
.toc-title {
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin: 0 0 0.6rem;
  font-size: 0.7rem;
}
.toc a {
  display: block;
  color: var(--accent);
  text-decoration: none;
  padding: 0.25rem 0;
}
.toc a:hover { text-decoration: underline; }
.report-main { min-width: 0; }
.section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  margin-bottom: 1.25rem;
  box-shadow: var(--shadow);
}
.section h2 {
  margin: 0 0 1rem;
  font-size: 1.15rem;
  color: var(--accent);
  border-bottom: 2px solid var(--accent-soft);
  padding-bottom: 0.5rem;
}
.hero {
  border-left: 4px solid var(--accent);
  padding: 1.5rem;
}
.hero[data-risk="normal"] { border-left-color: var(--risk-normal); }
.hero[data-risk="watch"] { border-left-color: var(--risk-watch); }
.hero[data-risk="elevated"] { border-left-color: var(--risk-elevated); }
.hero-top {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
}
.hero h2 { margin: 0; border: none; padding: 0; font-size: 1.25rem; }
.risk-badge {
  font-weight: 700;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 0.4rem 0.85rem;
  border-radius: 999px;
  color: #fff;
}
.risk-badge.normal { background: var(--risk-normal); }
.risk-badge.watch { background: var(--risk-watch); }
.risk-badge.elevated { background: var(--risk-elevated); }
.risk-badge.unknown { background: var(--muted); }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.75rem;
  margin: 1rem 0;
}
.metric-card {
  background: var(--accent-soft);
  border-radius: 8px;
  padding: 0.75rem 1rem;
}
.metric-card .label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--muted);
}
.metric-card .value {
  font-size: 1.2rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  margin-top: 0.15rem;
}
.hero-note { color: var(--muted); margin: 0; font-size: 0.95rem; }
.summary-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.75rem;
}
.summary-card {
  background: var(--bg);
  border-radius: 8px;
  padding: 0.85rem 1rem;
  border: 1px solid var(--border);
}
.summary-card strong { display: block; color: var(--accent); font-size: 1.05rem; margin-top: 0.2rem; }
.summary-card span { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
.usage-list { margin: 0; padding-left: 1.2rem; }
.usage-list li { margin: 0.4rem 0; }
.usage-list code {
  background: #eef2f6;
  padding: 0.12rem 0.35rem;
  border-radius: 4px;
  font-size: 0.88em;
}
.table-wrap { overflow-x: auto; margin: 0.5rem 0; }
table.data {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
}
table.data th, table.data td {
  border-bottom: 1px solid var(--border);
  padding: 0.5rem 0.65rem;
  text-align: left;
}
table.data th {
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
  white-space: nowrap;
}
table.data tr:hover td { background: #fafbfc; }
table.data tr.row-best td { background: #f0f9f4; }
table.data td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.data.compact { font-size: 0.8rem; }
.status {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: lowercase;
}
.status.pass, .status.passed { background: #e6f4ec; color: var(--pass); }
.status.review { background: #fff8e6; color: var(--review); }
.status.known_limitation, .status.not_used_in_default_model_path,
.status.educational_only { background: #eef2f6; color: var(--muted); }
.status.fail { background: #fdecea; color: var(--fail); }
.figure-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.25rem;
}
.figure-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  background: #fafbfc;
}
.figure-card figcaption {
  padding: 0.5rem 0.75rem;
  font-size: 0.82rem;
  color: var(--muted);
  background: var(--surface);
  border-top: 1px solid var(--border);
}
.figure-card img { width: 100%; display: block; }
.artifact-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.artifact-pills code {
  background: var(--bg);
  border: 1px solid var(--border);
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-size: 0.78rem;
}
.limitations-body h3 { font-size: 1rem; margin: 1rem 0 0.5rem; color: var(--accent); }
.limitations-body ul { margin: 0.5rem 0; padding-left: 1.2rem; }
.empty { color: var(--muted); font-style: italic; }
.site-footer {
  text-align: center;
  padding: 1.5rem;
  color: var(--muted);
  font-size: 0.82rem;
}
.site-footer code { background: #eef2f6; padding: 0.1rem 0.3rem; border-radius: 4px; }
"""


def _html_section(section_id: str, title: str, inner: str) -> str:
    return f'<section class="section" id="{section_id}"><h2>{html.escape(title)}</h2>{inner}</section>'


def _html_toc(entries: list[tuple[str, str]]) -> str:
    links = "".join(
        f'<a href="#{sid}">{html.escape(label)}</a>' for sid, label in entries
    )
    return f'<p class="toc-title">Contents</p>{links}'


def _html_snapshot_hero(snapshot: dict[str, Any]) -> str:
    level = str(snapshot.get("risk_level", "unknown"))
    as_of = html.escape(str(snapshot.get("as_of", "")))
    interpretation = _plain_text(snapshot.get("interpretation", ""))

    def metric(label: str, key: str, *, pct: bool = False) -> str:
        raw = snapshot.get(key)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            value = "—"
        elif pct and isinstance(raw, (int, float)):
            value = f"{float(raw) * 100:.1f}%"
        elif isinstance(raw, float):
            value = _format_cell(raw, precision=4 if key != "vix_close" else 2)
        else:
            value = html.escape(str(raw))
        return (
            f'<div class="metric-card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{value}</div></div>'
        )

    ratio = snapshot.get("vol_to_threshold")
    ratio_pct = ""
    if isinstance(ratio, (int, float)) and not pd.isna(ratio):
        ratio_pct = f"{float(ratio) * 100:.0f}% of threshold"

    return f"""
<section class="section hero" id="snapshot" data-risk="{html.escape(level)}">
  <div class="hero-top">
    <h2>Today&apos;s market snapshot · {as_of}</h2>
    <span class="risk-badge {html.escape(level)}">{html.escape(level)}</span>
  </div>
  <div class="metric-grid">
    {metric("SPY 20d realized vol", "realized_vol_20d")}
    {metric("High-vol threshold", "threshold")}
    {metric("Vol / threshold", "vol_to_threshold", pct=False)}
    {metric("VIX close", "vix_close")}
    {metric("VIX z-score (252d)", "vix_zscore_252d")}
  </div>
  <p class="hero-note">{html.escape(interpretation)}{f" · {html.escape(ratio_pct)}" if ratio_pct else ""}</p>
</section>
"""


def _html_daily_use(snapshot: dict[str, Any]) -> str:
    snap_line = ""
    if snapshot:
        snap_line = (
            f"<p><strong>Latest ({html.escape(str(snapshot.get('as_of', '')))})</strong>: "
            f"<span class='risk-badge {html.escape(str(snapshot.get('risk_level', '')))}'>"
            f"{html.escape(str(snapshot.get('risk_level', '')))}</span> — "
            f"vol/threshold {_format_cell(snapshot.get('vol_to_threshold'))}, "
            f"VIX z {_format_cell(snapshot.get('vix_zscore_252d'))}.</p>"
        )
    return f"""
<ul class="usage-list">
  <li><strong>Each morning</strong> — <code>sam risk snapshot</code> or <code>sam daily brief</code> for a quick SPY/VIX risk dial.</li>
  <li><strong>Weekly / after stress</strong> — <code>sam experiment run volatility-regime-scoring</code> to refresh metrics and this report.</li>
  <li><strong>Portfolio</strong> — combine risk level with ETF allocation; reduce exposure when <em>elevated</em>, stay selective when <em>watch</em>.</li>
</ul>
{snap_line}
"""


def _html_executive_summary(holdout: pd.DataFrame | None, snapshot: dict[str, Any]) -> str:
    if holdout is None or holdout.empty:
        return '<p class="empty">No holdout summary available.</p>'
    scored = holdout[holdout["Error"].isna() & holdout["average_precision"].notna()].copy()
    if scored.empty:
        return '<p class="empty">No successful models in holdout summary.</p>'
    best = scored.sort_values("average_precision", ascending=False).iloc[0]
    xgb = scored[scored["Model"].str.contains("XGBoost", na=False)]
    xgb_ap = _format_cell(xgb["average_precision"].iloc[0], precision=3) if not xgb.empty else "n/a"
    cards = [
        ("Holdout rows", str(int(best["Rows"]))),
        ("High-vol rate", f"{float(best['High Vol Rate']):.1%}"),
        ("Best holdout AP", f"{best['Model']} ({float(best['average_precision']):.3f})"),
        ("XGBoost AP", xgb_ap),
    ]
    if snapshot:
        cards.append(
            ("Today", f"{snapshot.get('risk_level')} · vol/threshold {_format_cell(snapshot.get('vol_to_threshold'))}")
        )
    card_html = "".join(
        f'<div class="summary-card"><span>{html.escape(label)}</span><strong>{html.escape(val)}</strong></div>'
        for label, val in cards
    )
    note = (
        "<p class='hero-note'>Simple VIX / realized-vol rules are the benchmark — "
        "learned models must beat them on alert quality, not AUC alone.</p>"
    )
    return f'<div class="summary-cards">{card_html}</div>{note}'


def _html_dataframe_table(
    frame: pd.DataFrame | None,
    *,
    highlight_column: str | None = None,
    highlight_value: str | None = None,
    compact: bool = False,
    numeric_columns: set[str] | None = None,
) -> str:
    if frame is None or frame.empty:
        return '<p class="empty">No data available.</p>'
    if numeric_columns is None:
        numeric_columns = {
            column
            for column in frame.columns
            if frame[column].dtype.kind in "fiu"
            or column
            in {
                "average_precision",
                "roc_auc",
                "top_10pct_precision",
                "top_10pct_recall",
                "precision",
                "realized_recall",
                "high_vol_rate",
                "ap_point",
                "ap_median",
                "ap_low",
                "ap_high",
            }
        }

    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    rows: list[str] = []
    for _, row in frame.iterrows():
        cells: list[str] = []
        is_best = (
            highlight_column is not None
            and highlight_value is not None
            and str(row.get(highlight_column, "")) == highlight_value
        )
        for column in frame.columns:
            value = row[column]
            if pd.isna(value) or value == "":
                text = ""
            elif column in numeric_columns and isinstance(value, (int, float)):
                text = _format_cell(float(value), precision=3)
            else:
                text = html.escape(str(value))
            css = "num" if column in numeric_columns else ""
            cells.append(f'<td class="{css}">{text}</td>')
        tr_class = ' class="row-best"' if is_best else ""
        rows.append(f"<tr{tr_class}>{''.join(cells)}</tr>")
    compact_class = " compact" if compact else ""
    return (
        f'<div class="table-wrap"><table class="data{compact_class}">'
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _html_leakage_table(leakage: pd.DataFrame | None) -> str:
    if leakage is None or leakage.empty:
        return '<p class="empty">No leakage checks.</p>'
    rows = []
    for _, row in leakage.iterrows():
        check = html.escape(str(row["Check"]))
        status = str(row["Status"]).lower().replace(" ", "_")
        badge = f'<span class="status {html.escape(status)}">{html.escape(str(row["Status"]))}</span>'
        rows.append(f"<tr><td>{check}</td><td>{badge}</td></tr>")
    return (
        '<div class="table-wrap"><table class="data">'
        "<thead><tr><th>Check</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _html_figures_section(run_dir: Path) -> str:
    figures = sorted(run_dir.glob("*.png"))
    if not figures:
        return _html_section(
            "figures",
            "Figures",
            '<p class="empty">No PNG figures. Install <code>.[volatility]</code> and re-run without <code>--no-figures</code>.</p>',
        )
    cards = []
    for path in figures:
        title = path.stem.replace("_", " ").title()
        cards.append(
            f'<figure class="figure-card">'
            f'<img src="{html.escape(path.name)}" alt="{html.escape(path.name)}" loading="lazy"/>'
            f"<figcaption>{html.escape(title)}</figcaption></figure>"
        )
    inner = f'<div class="figure-grid">{"".join(cards)}</div>'
    return _html_section("figures", "Figures", inner)


def _html_limitations(text: str) -> str:
    if not text.strip():
        return '<p class="empty">See experiment contract limitations.</p>'
    parts: list[str] = ['<div class="limitations-body">']
    in_list = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<h3>{html.escape(stripped[2:])}</h3>")
        elif stripped.startswith("## "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<h3>{html.escape(stripped[3:])}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{html.escape(stripped[2:])}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<p>{html.escape(stripped)}</p>")
    if in_list:
        parts.append("</ul>")
    parts.append("</div>")
    return "".join(parts)


def _html_artifact_index(run_dir: Path) -> str:
    names = sorted(path.name for path in run_dir.iterdir() if path.is_file())
    pills = "".join(f"<code>{html.escape(name)}</code>" for name in names)
    return f'<div class="artifact-pills">{pills}</div>'


def _plain_text(value: Any) -> str:
    text = str(value)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


__all__ = [
    "VolatilityReportResult",
    "build_volatility_market_snapshot",
    "render_volatility_report_html",
    "render_volatility_report_markdown",
    "write_volatility_regime_report",
]
