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
