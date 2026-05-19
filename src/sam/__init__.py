"""SAM quantitative research platform."""

from sam.allocation import build_score_allocation, scores_to_weights
from sam.backtest import run_backtest
from sam.daily import DailyBriefConfig, DailyBriefResult, run_daily_brief
from sam.data import fetch_prices
from sam.metrics import calculate_metrics
from sam.ml import ranking_diagnostics, train_model
from sam.portfolio import build_portfolio
from sam.research import ExperimentRegistry, ExperimentSpec, load_experiment
from sam.risk import build_volatility_regime_frame
from sam.stress import run_stress_tests

__all__ = [
    "ExperimentRegistry",
    "ExperimentSpec",
    "DailyBriefConfig",
    "DailyBriefResult",
    "build_score_allocation",
    "build_portfolio",
    "build_volatility_regime_frame",
    "calculate_metrics",
    "fetch_prices",
    "load_experiment",
    "ranking_diagnostics",
    "run_backtest",
    "run_daily_brief",
    "run_stress_tests",
    "scores_to_weights",
    "train_model",
]
