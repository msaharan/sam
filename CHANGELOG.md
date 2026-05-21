# Changelog

All notable changes to SAM are documented in this file.

## [0.2.0] - 2026-05-21

### Added

- Volatility-regime scoring experiment (`volatility-regime-scoring`): data ingest (yfinance, VIX, FRED), feature frame, XGBoost and optional TabPFN/TabICL baselines, leakage checks, risk-policy diagnostics, markdown/HTML reports, and `sam risk snapshot` for daily monitoring.
- Progress reporting (`tqdm`) for long experiment steps; `--no-progress` to disable.
- `sam experiment report` to rebuild HTML reports from an existing run directory.

### Changed

- Experiment artifacts default under `reports/experiments/volatility-regime-scoring/` (aligned CLI paths and README).
- FRED macro data fetched via the public REST API (`FRED_API_KEY` or cached `fred.csv`); removed unused `fredapi` optional extra.

### Fixed

- Ruff lint compliance for CI (import order, line length in CLI and feature modules).

## [0.1.0] - 2026-05-20

### Added

- Initial open-core release: ETF data fetch/validate, tactical allocation experiment contracts, daily brief, ML baselines, allocation/turnover utilities, and core research CLI.
