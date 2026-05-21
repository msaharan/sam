# SAM

SAM is an open-core personal quant research and engineering platform and is a companion to the [DSAIEngineering Newsletter](https://newsletter.dsaiengineering.com). The workflows and primitives described in the newsletter are implemented in SAM. Currently, it focuses on production-style US-listed ETF allocation, volatility/risk scoring, and future US cross-sectional
equity ranking workflows. More functionality will be integrated from the newsletter into SAM to make it more capable over time.

SAM is research software for educational purposes. It does not place trades, connect to brokers, track tax lots, reconcile fills, or provide investment, legal, tax, or regulatory advice.

## Install

```bash
cd sam
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,notebooks]"

cp .env.example .env
# Edit .env and set FRED_API_KEY (required for live volatility-regime runs with macro data)
```

## Quickstart

Fetch data for the core ETF allocation universe:

```bash
sam data fetch \
  --universe configs/universes/etf_core.toml \
  --market us \
  --start 2018-01-01 \
  --out data/raw/etf_core.parquet \
  --dataset-dir data/research
```

Validate the data, build the experiment contract bundle, create volatility-risk diagnostics, train walk-forward ML baselines, and translate scores into allocation diagnostics:

```bash
sam data validate --prices data/raw/etf_core.parquet --out reports/data_validation.csv
sam experiment bundle --experiment tactical-etf-allocation --out reports/experiments/tactical-etf-allocation
sam risk volatility-frame --prices data/raw/etf_core.parquet --symbol SPY --out reports/risk/volatility.csv
sam risk alerts --scores reports/risk/volatility.csv --score-columns realized_vol_20d --out reports/risk/alerts.csv
sam ml train --prices data/raw/etf_core.parquet --config configs/ml/baseline.toml --out models/latest
sam allocation weights --scores models/latest/predictions.parquet --top-k 5 --max-weight 0.2 --out reports/allocation/weights.csv
sam allocation turnover --weights reports/allocation/weights.csv --transaction-cost-bps 5 --slippage-bps 1 --turnover-limit 2 --out reports/allocation/turnover.csv
```

Build the daily research brief:

```bash
sam daily brief --config configs/daily/default.toml
open reports/daily/latest/brief.md
```

### Volatility regime: everyday vs full research

| Cadence | Command | What you get |
|--------|---------|----------------|
| **Each morning** | `sam risk snapshot` | SPY 20d realized vol vs pre-2018 threshold, VIX context, `normal` / `watch` / `elevated` dial |
| **Daily brief** | `sam daily brief` | ETF allocation + risk snapshot in one markdown brief |
| **Weekly / after stress** | `sam experiment run volatility-regime-scoring` | Refreshed holdout metrics, leakage checks, figures, and a single HTML report |
| **Re-report only** | `sam experiment report volatility-regime-scoring` | Rebuild `report.html` from an existing run folder without retraining |

```bash
# Morning risk dial (uses cached prices/vix under data/research/volatility_regime)
sam risk snapshot --cache-dir data/research/volatility_regime

# Full pipeline + consolidated report (opens path at end of run)
pip install -e ".[volatility]"   # matplotlib for publication PNGs
sam experiment run volatility-regime-scoring \
  --out reports/experiments/volatility-regime-scoring/run \
  --cache-dir data/research/volatility_regime
open reports/experiments/volatility-regime-scoring/run/report.html
# Stable symlink copy: reports/experiments/volatility-regime-scoring/latest/report.html
```

## Experiment Registry

Registered experiments live under `configs/experiments`. They define data, feature, target,
validation, decision, model, metric, artifact, limitation, and blog-post contracts.
Experiment bundles include data summaries, split summaries, contracts, model leaderboards,
calibration diagnostics, operating points, decision translation, cost/turnover sensitivity,
artifact and figure manifests, limitations, and publication checklists. When run artifacts are
provided with `--artifact`, the bundle summarizes observed metrics, calibration, allocation, and
turnover outputs instead of leaving those sections as templates.

```bash
sam experiment list
sam experiment validate
sam experiment bundle --experiment volatility-regime-scoring --out reports/experiments/volatility-regime-scoring
```

Run the P19 volatility-regime scoring workflow (domain rules + **XGBoost on CPU** included in the base install; optional `tfm` extra for TabPFN / TabICL; optional `volatility` extra for publication figures):

```bash
sam data fetch \
  --universe configs/universes/volatility_regime.toml \
  --start 2006-01-01 \
  --out data/research/volatility_regime/prices.parquet

# Live run also needs VIX (auto-downloaded) and FRED macro series.
# Set FRED_API_KEY in sam/.env (loaded automatically), or use --fred-csv / cache-dir/fred.csv

sam experiment run volatility-regime-scoring \
  --config configs/risk/volatility_regime_run.toml \
  --out reports/experiments/volatility-regime-scoring/run \
  --cache-dir data/research/volatility_regime
# Writes report.md + report.html (tables, executive summary, embedded figures). Use --no-report to skip.
# Progress bars (tqdm) show data load, features, XGBoost tuning, bootstrap, etc. Use --no-progress to disable.

sam experiment report volatility-regime-scoring \
  --run-dir reports/experiments/volatility-regime-scoring/run

# Optional: TabPFN / TabICL (large; GPU helps but not required)
# pip install -e ".[tfm]"

# Offline smoke test (no network, no FRED_API_KEY):
sam experiment run volatility-regime-scoring --synthetic --fast --skip-tfm --no-figures
```

Lower-level risk utilities:

```bash
sam risk snapshot --cache-dir data/research/volatility_regime
sam risk volatility-frame --prices data/raw/etf_core.parquet --symbol SPY --out reports/risk/volatility.csv
sam allocation weights --scores models/latest/predictions.parquet --top-k 5 --max-weight 0.2 --out reports/allocation/weights.csv
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

Default tests do not require network access. `yfinance` behavior is covered with mocks and static
fixtures.
