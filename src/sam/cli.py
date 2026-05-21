"""Command-line interface for SAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sam.allocation import allocation_turnover, scores_to_weights
from sam.backtest import run_backtest
from sam.config import (
    BacktestConfig,
    MLConfig,
    PortfolioConfig,
    load_backtest_config,
    load_ml_config,
    load_portfolio_config,
    load_stress_scenarios,
    load_universe,
    market_profile,
)
from sam.daily import load_daily_brief_config, run_daily_brief
from sam.data import (
    fetch_prices,
    long_returns_from_prices,
    validate_prices,
    write_price_dataset,
    write_return_dataset,
)
from sam.io import ensure_dir, read_table, write_manifest, write_table
from sam.ml import predict, train_model
from sam.portfolio import build_portfolio, portfolio_diagnostics
from sam.report import summarize_run
from sam.research import ExperimentRegistry, load_experiment, write_experiment_bundle
from sam.risk import alert_operating_points, build_volatility_regime_frame
from sam.stress import run_stress_tests

_SAM_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_project_dotenv() -> None:
    """Load sam/.env when present; existing shell exports take precedence."""

    from dotenv import load_dotenv

    for root in (Path.cwd(), _SAM_PROJECT_ROOT):
        env_file = root / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)


def main(argv: list[str] | None = None) -> int:
    _load_project_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sam", description="SAM quant research platform")
    subparsers = parser.add_subparsers(required=True)

    data = subparsers.add_parser("data", help="Market data ingestion and validation")
    data_sub = data.add_subparsers(required=True)
    fetch = data_sub.add_parser("fetch", help="Fetch yfinance OHLCV data")
    fetch.add_argument("--symbols", nargs="*", default=[])
    fetch.add_argument("--universe")
    fetch.add_argument("--market", choices=["us"], default="us")
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end")
    fetch.add_argument("--out", default="data/raw/prices.parquet")
    fetch.add_argument(
        "--dataset-dir",
        help="Also write partitioned Parquet datasets under this root",
    )
    fetch.set_defaults(func=cmd_data_fetch)
    validate = data_sub.add_parser("validate", help="Validate canonical price data")
    validate.add_argument("--prices", required=True)
    validate.add_argument("--out", default="reports/data_validation.csv")
    validate.set_defaults(func=cmd_data_validate)

    portfolio = subparsers.add_parser("portfolio", help="Portfolio construction")
    portfolio_sub = portfolio.add_subparsers(required=True)
    build = portfolio_sub.add_parser("build", help="Build target weights")
    build.add_argument("--prices", required=True)
    build.add_argument("--config")
    build.add_argument("--method", default="inverse_volatility")
    build.add_argument("--out", default="reports/portfolio/latest")
    build.set_defaults(func=cmd_portfolio_build)

    backtest = subparsers.add_parser("backtest", help="Backtest workflows")
    backtest_sub = backtest.add_subparsers(required=True)
    run = backtest_sub.add_parser("run", help="Run a vectorized backtest")
    run.add_argument("--prices", required=True)
    run.add_argument("--config")
    run.add_argument("--method", default="inverse_volatility")
    run.add_argument("--benchmark-symbol")
    run.add_argument("--out", default="reports/backtests/latest")
    run.set_defaults(func=cmd_backtest_run)
    compare = backtest_sub.add_parser("compare", help="Compare backtest run directories")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--out", default="reports/backtests/comparison.csv")
    compare.set_defaults(func=cmd_backtest_compare)

    stress = subparsers.add_parser("stress", help="Stress testing")
    stress_sub = stress.add_subparsers(required=True)
    stress_run = stress_sub.add_parser("run", help="Run default stress scenarios")
    stress_run.add_argument("--prices", required=True)
    stress_run.add_argument("--weights", required=True)
    stress_run.add_argument("--config")
    stress_run.add_argument("--benchmark-symbol")
    stress_run.add_argument("--out", default="reports/stress/latest")
    stress_run.set_defaults(func=cmd_stress_run)

    ml = subparsers.add_parser("ml", help="Walk-forward ML research")
    ml_sub = ml.add_subparsers(required=True)
    train = ml_sub.add_parser("train", help="Train baseline walk-forward models")
    train.add_argument("--prices", required=True)
    train.add_argument("--config")
    train.add_argument("--out", default="models/latest")
    train.set_defaults(func=cmd_ml_train)
    pred = ml_sub.add_parser("predict", help="Generate predictions with a saved model")
    pred.add_argument("--prices", required=True)
    pred.add_argument("--model", required=True)
    pred.add_argument("--out", default="reports/ml/predictions.parquet")
    pred.set_defaults(func=cmd_ml_predict)

    risk = subparsers.add_parser("risk", help="Risk-scoring research utilities")
    risk_sub = risk.add_subparsers(required=True)
    vol = risk_sub.add_parser("volatility-frame", help="Build volatility-regime research table")
    vol.add_argument("--prices", required=True)
    vol.add_argument("--symbol", required=True)
    vol.add_argument("--horizon-days", type=int, default=20)
    vol.add_argument("--threshold-quantile", type=float, default=0.8)
    vol.add_argument("--threshold-end")
    vol.add_argument("--out", default="reports/risk/volatility_regime.csv")
    vol.set_defaults(func=cmd_risk_volatility_frame)
    alerts = risk_sub.add_parser("alerts", help="Build alert-queue operating points")
    alerts.add_argument("--scores", required=True)
    alerts.add_argument("--score-columns", nargs="+", required=True)
    alerts.add_argument("--target-column", default="target_high_vol")
    alerts.add_argument("--recalls", nargs="+", type=float, default=[0.5, 0.8, 0.9])
    alerts.add_argument("--out", default="reports/risk/alert_operating_points.csv")
    alerts.set_defaults(func=cmd_risk_alerts)
    snapshot = risk_sub.add_parser(
        "snapshot",
        help="Today's SPY/VIX volatility snapshot for daily monitoring",
    )
    snapshot.add_argument("--cache-dir", default="data/research/volatility_regime")
    snapshot.add_argument("--config", default="configs/risk/volatility_regime_run.toml")
    snapshot.add_argument("--symbol", default="SPY")
    snapshot.add_argument(
        "--out",
        default="reports/experiments/volatility-regime-scoring/snapshot.json",
    )
    snapshot.set_defaults(func=cmd_risk_snapshot)

    allocation = subparsers.add_parser("allocation", help="Score-driven allocation utilities")
    allocation_sub = allocation.add_subparsers(required=True)
    weights_cmd = allocation_sub.add_parser("weights", help="Convert scores into target weights")
    weights_cmd.add_argument("--scores", required=True)
    weights_cmd.add_argument("--score-column", default="prediction")
    weights_cmd.add_argument("--top-k", type=int)
    weights_cmd.add_argument("--top-quantile", type=float)
    weights_cmd.add_argument("--max-weight", type=float, default=1.0)
    weights_cmd.add_argument("--weighting", choices=["equal", "rank"], default="equal")
    weights_cmd.add_argument("--out", default="reports/allocation/weights.csv")
    weights_cmd.set_defaults(func=cmd_allocation_weights)
    turnover_cmd = allocation_sub.add_parser(
        "turnover",
        help="Compute turnover and cost diagnostics",
    )
    turnover_cmd.add_argument("--weights", required=True)
    turnover_cmd.add_argument("--transaction-cost-bps", type=float, default=0.0)
    turnover_cmd.add_argument("--slippage-bps", type=float, default=0.0)
    turnover_cmd.add_argument("--turnover-limit", type=float)
    turnover_cmd.add_argument("--out", default="reports/allocation/turnover.csv")
    turnover_cmd.set_defaults(func=cmd_allocation_turnover)

    report = subparsers.add_parser("report", help="Summarize run artifacts")
    report_sub = report.add_subparsers(required=True)
    summarize = report_sub.add_parser("summarize", help="Summarize a run directory")
    summarize.add_argument("--run-dir", required=True)
    summarize.add_argument("--out")
    summarize.set_defaults(func=cmd_report_summarize)

    experiment = subparsers.add_parser("experiment", help="Research experiment registry")
    experiment_sub = experiment.add_subparsers(required=True)
    list_cmd = experiment_sub.add_parser("list", help="List registered experiments")
    list_cmd.add_argument("--registry", default="configs/experiments")
    list_cmd.add_argument("--out")
    list_cmd.set_defaults(func=cmd_experiment_list)
    validate_cmd = experiment_sub.add_parser("validate", help="Validate experiment contracts")
    validate_cmd.add_argument("--registry", default="configs/experiments")
    validate_cmd.set_defaults(func=cmd_experiment_validate)
    bundle_cmd = experiment_sub.add_parser("bundle", help="Write a standard experiment bundle")
    bundle_cmd.add_argument("--experiment", required=True, help="Experiment id or TOML path")
    bundle_cmd.add_argument("--registry", default="configs/experiments")
    bundle_cmd.add_argument("--artifact", action="append", default=[])
    bundle_cmd.add_argument("--out", default="reports/experiments/latest")
    bundle_cmd.set_defaults(func=cmd_experiment_bundle)
    run_cmd = experiment_sub.add_parser(
        "run",
        help="Execute a registered experiment workflow",
    )
    run_cmd.add_argument(
        "experiment",
        nargs="?",
        metavar="EXPERIMENT",
        help="Registered experiment id (e.g. volatility-regime-scoring)",
    )
    run_cmd.add_argument(
        "--experiment",
        dest="experiment_flag",
        metavar="EXPERIMENT",
        help="Same as positional EXPERIMENT (for scripts that prefer flags)",
    )
    run_cmd.add_argument("--config", default="configs/risk/volatility_regime_run.toml")
    run_cmd.add_argument("--universe", default="configs/universes/volatility_regime.toml")
    run_cmd.add_argument("--registry", default="configs/experiments")
    run_cmd.add_argument("--out", default="reports/experiments/volatility-regime-scoring/run")
    run_cmd.add_argument("--cache-dir", default="data/research/volatility_regime")
    run_cmd.add_argument("--vix-csv")
    run_cmd.add_argument("--fred-csv")
    run_cmd.add_argument("--synthetic", action="store_true")
    run_cmd.add_argument("--refresh-data", action="store_true")
    run_cmd.add_argument("--skip-tfm", action="store_true")
    run_cmd.add_argument("--skip-xgboost", action="store_true")
    run_cmd.add_argument(
        "--fast",
        action="store_true",
        help="Reduced bootstrap/tuning iterations for offline smoke runs",
    )
    run_cmd.add_argument("--no-figures", action="store_true")
    run_cmd.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    run_cmd.add_argument(
        "--no-report",
        action="store_true",
        help="Skip consolidated report.md / report.html",
    )
    run_cmd.set_defaults(func=cmd_experiment_run)
    report_cmd = experiment_sub.add_parser(
        "report",
        help="Build consolidated HTML/Markdown report from a prior run",
    )
    report_cmd.add_argument(
        "experiment",
        nargs="?",
        metavar="EXPERIMENT",
        help="Experiment id (volatility-regime-scoring)",
    )
    report_cmd.add_argument(
        "--run-dir",
        default="reports/experiments/volatility-regime-scoring/run",
    )
    report_cmd.add_argument("--bundle-dir", default="reports/experiments/volatility-regime-scoring")
    report_cmd.add_argument("--cache-dir", default="data/research/volatility_regime")
    report_cmd.set_defaults(func=cmd_experiment_report)

    daily = subparsers.add_parser("daily", help="Daily research brief workflows")
    daily_sub = daily.add_subparsers(required=True)
    brief = daily_sub.add_parser("brief", help="Build the daily SAM research brief")
    brief.add_argument("--config", default="configs/daily/default.toml")
    brief.add_argument("--out")
    brief.add_argument("--as-of")
    brief.add_argument("--no-refresh-data", action="store_true")
    brief.add_argument("--retrain", action="store_true")
    brief.add_argument("--train-if-missing", action="store_true")
    brief.set_defaults(func=cmd_daily_brief)
    return parser


def cmd_data_fetch(args: argparse.Namespace) -> int:
    symbols = list(args.symbols)
    if args.universe:
        universe = load_universe(args.universe)
        symbols.extend(universe.normalized_symbols)
    if not symbols:
        symbols = market_profile(args.market).default_universe
    frame = fetch_prices(symbols, start=args.start, end=args.end, market=args.market, out=args.out)
    if args.dataset_dir:
        write_price_dataset(frame, args.dataset_dir, table="raw_ohlcv")
        returns = long_returns_from_prices(frame)
        write_return_dataset(returns, args.dataset_dir, table="returns")
    write_manifest(Path(args.out).parent, command="data fetch", config=vars(args))
    print(f"Wrote {len(frame)} price rows to {args.out}")
    return 0


def cmd_data_validate(args: argparse.Namespace) -> int:
    prices = read_table(args.prices)
    diagnostics = validate_prices(prices)
    write_table(diagnostics, args.out)
    failures = int((diagnostics["status"] == "fail").sum())
    print(f"Validated {len(diagnostics)} symbols ({failures} failures)")
    return 1 if failures else 0


def cmd_portfolio_build(args: argparse.Namespace) -> int:
    prices = read_table(args.prices)
    config = (
        load_portfolio_config(args.config)
        if args.config
        else PortfolioConfig(method=args.method)
    )
    weights = build_portfolio(prices, method=config.method, config=config)
    diagnostics = portfolio_diagnostics(prices, weights, config=config)
    out_dir = ensure_dir(args.out)
    weights_path = write_table(weights, out_dir / "weights.csv")
    write_table(diagnostics, out_dir / "diagnostics.csv")
    (out_dir / "config.json").write_text(
        json.dumps(config.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_manifest(out_dir, command="portfolio build", config=config.model_dump())
    print(f"Wrote target weights to {weights_path}")
    return 0


def cmd_backtest_run(args: argparse.Namespace) -> int:
    prices = read_table(args.prices)
    if args.config:
        config = load_backtest_config(args.config)
    else:
        config = BacktestConfig(portfolio=PortfolioConfig(method=args.method))
    benchmark_prices = _filter_symbol(prices, args.benchmark_symbol or config.portfolio.benchmark)
    result = run_backtest(prices, config=config, benchmark_prices=benchmark_prices)
    out_dir = ensure_dir(args.out)
    write_table(result.returns, out_dir / "returns.csv")
    write_table(result.weights, out_dir / "weights.csv")
    (out_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_manifest(out_dir, command="backtest run", config=config.model_dump())
    print(f"Wrote backtest artifacts to {out_dir}")
    return 0


def cmd_backtest_compare(args: argparse.Namespace) -> int:
    rows = []
    for run_dir in args.runs:
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        rows.append(
            {
                "name": Path(run_dir).name,
                **json.loads(metrics_path.read_text(encoding="utf-8")),
            }
        )
    frame = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    write_table(frame, args.out)
    print(f"Wrote comparison to {args.out}")
    return 0


def cmd_stress_run(args: argparse.Namespace) -> int:
    prices = read_table(args.prices)
    weights = read_table(args.weights)
    scenarios = load_stress_scenarios(args.config) if args.config else None
    benchmark_prices = (
        _filter_symbol(prices, args.benchmark_symbol) if args.benchmark_symbol else None
    )
    stresses = run_stress_tests(
        prices,
        weights,
        scenarios=scenarios,
        benchmark_prices=benchmark_prices,
    )
    out_dir = ensure_dir(args.out)
    write_table(stresses, out_dir / "stress.csv")
    write_manifest(out_dir, command="stress run", config=vars(args))
    print(f"Wrote stress results to {out_dir}")
    return 0


def cmd_ml_train(args: argparse.Namespace) -> int:
    prices = read_table(args.prices)
    config = load_ml_config(args.config) if args.config else MLConfig()
    result = train_model(prices, config=config, out_dir=args.out)
    write_manifest(args.out, command="ml train", config=config.model_dump())
    print(f"Wrote ML predictions and metrics to {args.out}")
    if result.predictions.empty:
        print("No walk-forward folds were available; use a longer history or shorter windows.")
        return 1
    return 0


def cmd_ml_predict(args: argparse.Namespace) -> int:
    prices = read_table(args.prices)
    predictions = predict(prices, args.model)
    write_table(predictions, args.out)
    print(f"Wrote predictions to {args.out}")
    return 0


def cmd_risk_volatility_frame(args: argparse.Namespace) -> int:
    prices = read_table(args.prices)
    result = build_volatility_regime_frame(
        prices,
        symbol=args.symbol,
        horizon_days=args.horizon_days,
        threshold_quantile=args.threshold_quantile,
        threshold_end=args.threshold_end,
    )
    write_table(result.frame, args.out)
    write_manifest(
        Path(args.out).parent,
        command="risk volatility-frame",
        config={
            "symbol": args.symbol,
            "horizon_days": args.horizon_days,
            "threshold": result.threshold,
            "threshold_quantile": result.threshold_quantile,
            "threshold_end": result.threshold_end,
        },
    )
    print(f"Wrote volatility-regime frame to {args.out}")
    return 0


def cmd_risk_alerts(args: argparse.Namespace) -> int:
    scores = read_table(args.scores)
    operating_points = alert_operating_points(
        scores,
        score_columns=args.score_columns,
        target_column=args.target_column,
        recalls=args.recalls,
    )
    write_table(operating_points, args.out)
    print(f"Wrote alert operating points to {args.out}")
    return 0


def cmd_allocation_weights(args: argparse.Namespace) -> int:
    scores = read_table(args.scores)
    weights = scores_to_weights(
        scores,
        score_column=args.score_column,
        top_k=args.top_k,
        top_quantile=args.top_quantile,
        max_weight=args.max_weight,
        weighting=args.weighting,
    )
    write_table(weights, args.out)
    print(f"Wrote score-driven weights to {args.out}")
    return 0


def cmd_allocation_turnover(args: argparse.Namespace) -> int:
    weights = read_table(args.weights)
    diagnostics = allocation_turnover(
        weights,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        turnover_limit=args.turnover_limit,
    )
    write_table(diagnostics, args.out)
    print(f"Wrote allocation turnover diagnostics to {args.out}")
    return 0


def cmd_report_summarize(args: argparse.Namespace) -> int:
    summary = summarize_run(args.run_dir)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        target = Path(args.out)
        ensure_dir(target.parent)
        target.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def cmd_experiment_list(args: argparse.Namespace) -> int:
    frame = ExperimentRegistry(args.registry).index_frame()
    if args.out:
        write_table(frame, args.out)
    if frame.empty:
        print("No experiments registered")
    else:
        print(frame.to_string(index=False))
    return 0


def cmd_experiment_validate(args: argparse.Namespace) -> int:
    registry = ExperimentRegistry(args.registry)
    specs = registry.specs()
    issues = registry.validation_issues()
    if not issues.empty:
        print(issues.to_string(index=False))
        return 1
    print(f"Validated {len(specs)} experiment contracts")
    return 0


def cmd_experiment_run(args: argparse.Namespace) -> int:
    from sam.volatility_experiment import run_volatility_regime_experiment

    experiment_id = args.experiment_flag or args.experiment
    if not experiment_id:
        print(
            "Experiment id required. Example:\n"
            "  sam experiment run volatility-regime-scoring --synthetic --fast\n"
            "  sam experiment run --experiment volatility-regime-scoring --synthetic --fast"
        )
        return 2
    if experiment_id != "volatility-regime-scoring":
        print(f"Unsupported experiment run id: {experiment_id}")
        print("Supported ids: volatility-regime-scoring")
        return 1
    result = run_volatility_regime_experiment(
        experiment_id=experiment_id,
        run_config_path=args.config,
        universe_path=args.universe,
        out_dir=args.out,
        cache_dir=args.cache_dir,
        registry_path=args.registry,
        use_synthetic_data=args.synthetic,
        vix_csv=args.vix_csv,
        fred_csv=args.fred_csv,
        skip_tfm=args.skip_tfm,
        skip_xgboost=args.skip_xgboost,
        fast_mode=args.fast,
        refresh_data=args.refresh_data,
        write_figures=not args.no_figures,
        show_progress=not args.no_progress,
        write_report=not args.no_report,
    )
    print(f"Wrote volatility-regime run artifacts to {result.out_dir}")
    if not args.no_report:
        report_html = Path(result.out_dir) / "report.html"
        if report_html.is_file():
            print(f"Open consolidated report: {report_html.resolve()}")
    if result.bundle_dir is not None:
        print(f"Updated experiment bundle at {result.bundle_dir}")
    return 0


def cmd_risk_snapshot(args: argparse.Namespace) -> int:
    from sam.volatility_report import build_volatility_market_snapshot

    snapshot = build_volatility_market_snapshot(
        cache_dir=args.cache_dir,
        run_config_path=args.config,
        symbol=args.symbol,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Risk level: {snapshot['risk_level']} (as of {snapshot['as_of']})")
    print(snapshot["interpretation"])
    print(f"Wrote snapshot to {out}")
    return 0


def cmd_experiment_report(args: argparse.Namespace) -> int:
    from sam.volatility_report import (
        build_volatility_market_snapshot,
        write_volatility_regime_report,
    )

    experiment_id = args.experiment or "volatility-regime-scoring"
    if experiment_id != "volatility-regime-scoring":
        print(f"Report generation not implemented for: {experiment_id}")
        return 1
    snapshot = {}
    try:
        snapshot = build_volatility_market_snapshot(cache_dir=args.cache_dir)
    except (ValueError, OSError) as exc:
        print(f"Market snapshot skipped: {exc}")
    result = write_volatility_regime_report(
        args.run_dir,
        bundle_dir=args.bundle_dir,
        snapshot=snapshot,
    )
    print(f"Wrote report to {result.html_path}")
    return 0


def cmd_experiment_bundle(args: argparse.Namespace) -> int:
    experiment_path = Path(args.experiment)
    spec = (
        load_experiment(experiment_path)
        if experiment_path.suffix == ".toml" or experiment_path.exists()
        else ExperimentRegistry(args.registry).get(args.experiment)
    )
    bundle = write_experiment_bundle(spec, args.out, artifact_paths=args.artifact)
    print(f"Wrote experiment bundle to {bundle.out_dir}")
    return 0


def cmd_daily_brief(args: argparse.Namespace) -> int:
    overrides = {
        "output_dir": args.out,
        "as_of": args.as_of,
        "refresh_data": False if args.no_refresh_data else None,
        "retrain": True if args.retrain else None,
        "train_if_missing": True if args.train_if_missing else None,
    }
    config = load_daily_brief_config(args.config, overrides=overrides)
    result = run_daily_brief(config)
    print(f"Wrote daily SAM brief to {result.out_dir}")
    return 0


def _filter_symbol(prices: pd.DataFrame, symbol: str | None) -> pd.DataFrame | None:
    if symbol is None or "symbol" not in prices.columns:
        return None
    filtered = prices[prices["symbol"] == symbol]
    return filtered if not filtered.empty else None


__all__ = ["build_parser", "main"]
