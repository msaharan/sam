from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sam import __version__
from sam.config.loader import (
    BacktestRunConfig,
    DataSyncConfig,
    LiveRunConfig,
    RiskConfig,
    SamSettings,
    ensure_dirs,
    load_yaml,
)
from sam.pipeline.backtest_run import run_backtest
from sam.pipeline.data_sync import run_data_sync, run_data_validate
from sam.pipeline.research_run import research_pipeline, run_diagnose, run_features, run_train
from sam.reporting.ops_brief import write_ops_brief


def _root() -> Path:
    return Path.cwd()


def cmd_data_sync(args: argparse.Namespace) -> int:
    settings = SamSettings()
    ensure_dirs(settings)
    cfg = DataSyncConfig.model_validate(load_yaml(args.config))
    manifest = run_data_sync(cfg, settings)
    print(f"Synced {len(manifest.get('symbols', []))} symbols to {settings.sam_data_dir}")
    return 0


def cmd_data_validate(args: argparse.Namespace) -> int:
    from sam.config.loader import DataValidateConfig

    settings = SamSettings()
    ensure_dirs(settings)
    cfg = DataValidateConfig.model_validate(load_yaml(args.config))
    summary = run_data_validate(cfg, settings)
    print(f"Validated {len(summary.get('symbols', []))} symbols; passed={summary['passed']}")
    return 0 if summary["passed"] else 1


def cmd_backtest_run(args: argparse.Namespace) -> int:
    settings = SamSettings()
    cfg = BacktestRunConfig.model_validate(load_yaml(args.config))
    result = run_backtest(cfg, settings, root=_root())
    print(f"Backtest complete: {result['output_dir']}")
    print(f"Metrics: {result['metrics']}")
    return 0


def cmd_report_backtest(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    report_path = out_dir / "report.html"
    try:
        from ml4t.backtest.result import BacktestResult
        from ml4t.diagnostic.integration import generate_tearsheet_from_result

        result = BacktestResult.from_parquet(out_dir)
        generate_tearsheet_from_result(
            result,
            template="quant_trader",
            output_path=report_path,
        )
    except ImportError:
        import json

        import polars as pl

        trades_path = out_dir / "trades.parquet"
        equity_path = out_dir / "equity.parquet"
        metrics_path = out_dir / "metrics.json"
        if not trades_path.exists() and not equity_path.exists():
            print(f"Missing trades/equity under {out_dir}")
            return 1
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        trades = pl.read_parquet(trades_path) if trades_path.exists() else pl.DataFrame()
        if equity_path.exists():
            equity = pl.read_parquet(equity_path).sort("timestamp")
            value_col = "portfolio_value" if "portfolio_value" in equity.columns else "equity"
            daily_returns = equity[value_col].pct_change().drop_nulls().to_list()
        else:
            daily_returns = []
        try:
            from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

            html = generate_backtest_tearsheet(
                trades=trades,
                returns=daily_returns,
                metrics=metrics,
                template="quant_trader",
            )
        except ImportError:
            html = (
                "<html><body><h1>SAM Backtest Report (fallback)</h1>"
                f"<p>Install <code>sam[research]</code> for full diagnostic tearsheets "
                "(requires CMake for numba/llvmlite on some platforms).</p>"
                f"<pre>{json.dumps(metrics, indent=2)}</pre></body></html>"
            )
        report_path.write_text(html)
    print(f"Wrote {report_path}")
    return 0


def cmd_research_features(args: argparse.Namespace) -> int:
    settings = SamSettings()
    out = settings.sam_artifacts_dir / "research" / args.run_id / "features"
    path = run_features(args.prices, out)
    print(f"Features: {path}")
    return 0


def cmd_research_diagnose(args: argparse.Namespace) -> int:
    settings = SamSettings()
    base = settings.sam_artifacts_dir / "research" / args.run_id
    feat = str(base / "features" / "features.parquet")
    summary = run_diagnose(args.prices, feat, base / "diagnostics")
    print(summary)
    return 0


def cmd_research_train(args: argparse.Namespace) -> int:
    settings = SamSettings()
    base = settings.sam_artifacts_dir / "research" / args.run_id
    feat = str(base / "features" / "features.parquet")
    path = run_train(args.prices, feat, base / "signals")
    print(f"Signals: {path}")
    return 0


def cmd_research_all(args: argparse.Namespace) -> int:
    settings = SamSettings()
    result = research_pipeline(settings, args.prices, args.run_id)
    print(result)
    return 0


async def _live_shadow(args: argparse.Namespace) -> int:
    from sam.live.runner import run_shadow

    settings = SamSettings()
    live_cfg = _load_live_config(args)
    return await run_shadow(
        settings,
        live_cfg,
        live_cfg.strategy_config,
        args.duration,
        _root(),
    )


async def _live_paper(args: argparse.Namespace) -> int:
    from sam.live.runner import run_paper

    settings = SamSettings()
    live_cfg = _load_live_config(args)
    return await run_paper(
        settings,
        live_cfg,
        live_cfg.strategy_config,
        args.duration,
        _root(),
    )


async def _live_live(args: argparse.Namespace) -> int:
    from sam.live.runner import run_live

    settings = SamSettings()
    live_cfg = _load_live_config(args)
    return await run_live(
        settings,
        live_cfg,
        live_cfg.strategy_config,
        args.duration,
        _root(),
    )


async def _live_ib(args: argparse.Namespace) -> int:
    from sam.live.runner import run_ib_paper

    settings = SamSettings()
    live_cfg = _load_live_config(args)
    return await run_ib_paper(
        settings,
        live_cfg,
        live_cfg.strategy_config,
        args.duration,
        _root(),
    )


def cmd_live_preview(args: argparse.Namespace) -> int:
    import json

    from sam.live.runner import run_preview

    settings = SamSettings()
    result = asyncio.run(
        run_preview(settings, args.config, args.bars, _root())
    )
    print(json.dumps(result, indent=2))
    return 0


async def _ops_preflight(args: argparse.Namespace) -> int:
    from sam.live.runner import run_preflight

    settings = SamSettings()
    live_cfg = _load_live_config(args)
    result = await run_preflight(settings, live_cfg, _root())
    print(result)
    return 0 if result.get("passed", True) else 1


def cmd_ops_status(args: argparse.Namespace) -> int:
    import json

    from sam.ops.status import collect_ops_status

    settings = SamSettings()
    live_cfg = _load_live_config(args) if args.broker_snapshot else None
    payload = collect_ops_status(
        settings,
        include_broker=args.broker_snapshot,
        live_config=live_cfg,
        root=_root(),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"kill_switch_active: {payload['kill_switch_active']}")
        freshness = payload.get("data_freshness", {})
        print(f"latest_bar_timestamp: {freshness.get('latest_bar_timestamp')}")
        print(f"data_staleness_seconds: {freshness.get('staleness_seconds')}")
        latest = payload.get("latest_manifest") or {}
        print(f"latest_manifest: {latest.get('path')}")
        for entry in payload["risk_files"]:
            print(
                f"  {entry['file']}: kill={entry['kill_switch_activated']} "
                f"reason={entry['kill_switch_reason']!r} date={entry.get('date')}"
            )
        for entry in payload["run_manifests"]:
            print(
                f"  manifest {entry['path']}: {entry['strategy_id']} "
                f"state={entry['promotion_state']}"
            )
        broker = payload.get("broker_snapshot", {})
        if broker.get("enabled"):
            print(f"broker_snapshot_available: {broker.get('available')}")
    return 0


def cmd_ops_brief(args: argparse.Namespace) -> int:
    settings = SamSettings()
    path = write_ops_brief(settings, fmt=args.format)
    print(path)
    return 0


def cmd_ops_kill_switch(args: argparse.Namespace) -> int:
    from sam.ops.status import set_kill_switch_state

    settings = SamSettings()
    state = Path(args.state_file)
    if not state.is_absolute():
        state = settings.sam_state_dir / state.name
    payload = set_kill_switch_state(state, args.activate, args.reason)
    print(f"{state}: kill_switch_activated={payload['kill_switch_activated']}")
    return 0


def cmd_ops_promote(args: argparse.Namespace) -> int:
    import json

    from sam.manifests.run_manifest import PromotionState
    from sam.ops.promote import promote_manifest

    manifest_path = Path(args.manifest)
    decision = promote_manifest(
        manifest_path,
        from_state=PromotionState(args.from_state),
        to_state=PromotionState(args.to_state),
        reason=args.reason,
        force=args.force,
    )
    print(json.dumps(decision, indent=2))
    return 0


def _load_live_config(args: argparse.Namespace) -> LiveRunConfig:
    raw = load_yaml(args.config)
    env_raw = load_yaml(args.environment) if args.environment else {}
    risk_data = {**env_raw.get("risk", {}), **raw.get("risk", {})}
    if args.environment and env_raw.get("environment"):
        raw["environment"] = env_raw["environment"]
    if getattr(args, "duration", None) is not None:
        raw["duration_seconds"] = args.duration
    raw["live_config_path"] = args.config
    raw["environment_config_path"] = args.environment
    raw["risk"] = risk_data
    cfg = LiveRunConfig.model_validate(raw)
    settings = SamSettings()
    state = Path(cfg.state_file)
    if not state.is_absolute():
        cfg.state_file = str(settings.sam_state_dir / state.name)
    cfg.risk = RiskConfig.model_validate(risk_data)
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sam", description="SAM production trading CLI")
    p.add_argument("--version", action="version", version=f"sam {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="data operations")
    data_sub = data.add_subparsers(dest="data_cmd", required=True)
    sync = data_sub.add_parser("sync", help="sync market data via ml4t-data")
    sync.add_argument("--config", default="configs/data/default.yaml")
    sync.set_defaults(func=cmd_data_sync)
    validate = data_sub.add_parser("validate", help="validate existing parquet data")
    validate.add_argument("--config", default="configs/data/default.yaml")
    validate.set_defaults(func=cmd_data_validate)

    bt = sub.add_parser("backtest", help="backtest operations")
    bt_sub = bt.add_subparsers(dest="bt_cmd", required=True)
    run = bt_sub.add_parser("run", help="run backtest")
    run.add_argument("--config", default="configs/backtest/ma_baseline.yaml")
    run.set_defaults(func=cmd_backtest_run)

    live = sub.add_parser("live", help="live trading")
    live_sub = live.add_subparsers(dest="live_cmd", required=True)
    shadow = live_sub.add_parser("shadow", help="shadow mode with synthetic feed")
    shadow.add_argument("--config", default="configs/live/ma_baseline.yaml")
    shadow.add_argument("--environment", default="configs/environments/shadow.yaml")
    shadow.add_argument("--duration", type=int, default=30)
    shadow.set_defaults(func=lambda a: asyncio.run(_live_shadow(a)))

    paper = live_sub.add_parser("paper", help="Alpaca paper trading")
    paper.add_argument("--config", default="configs/live/ma_baseline.yaml")
    paper.add_argument("--environment", default="configs/environments/paper.yaml")
    paper.add_argument("--duration", type=int, default=95)
    paper.set_defaults(func=lambda a: asyncio.run(_live_paper(a)))

    live_cmd = live_sub.add_parser("live", help="Alpaca live trading (requires keys)")
    live_cmd.add_argument("--config", default="configs/live/ma_baseline.yaml")
    live_cmd.add_argument("--environment", default="configs/environments/live.yaml")
    live_cmd.add_argument("--duration", type=int, default=None)
    live_cmd.set_defaults(func=lambda a: asyncio.run(_live_live(a)))

    ib_cmd = live_sub.add_parser("ib", help="Interactive Brokers paper (TWS 7497)")
    ib_cmd.add_argument("--config", default="configs/live/ib_ma_baseline.yaml")
    ib_cmd.add_argument("--environment", default="configs/environments/paper.yaml")
    ib_cmd.add_argument("--duration", type=int, default=75)
    ib_cmd.set_defaults(func=lambda a: asyncio.run(_live_ib(a)))

    preview = live_sub.add_parser("preview", help="dry-run synthetic order preview")
    preview.add_argument("--config", default="configs/strategies/ma_crossover.yaml")
    preview.add_argument("--bars", type=int, default=20)
    preview.set_defaults(func=cmd_live_preview)

    research = sub.add_parser("research", help="research pipeline")
    res_sub = research.add_subparsers(dest="res_cmd", required=True)
    for name, func in [
        ("features", cmd_research_features),
        ("diagnose", cmd_research_diagnose),
        ("train", cmd_research_train),
    ]:
        cmd = res_sub.add_parser(name)
        cmd.add_argument("--prices", default="tests/fixtures/ohlcv_ma_baseline.parquet")
        cmd.add_argument("--run-id", default="default")
        cmd.set_defaults(func=func)
    all_cmd = res_sub.add_parser("all", help="run full research pipeline")
    all_cmd.add_argument("--prices", default="tests/fixtures/ohlcv_ma_baseline.parquet")
    all_cmd.add_argument("--run-id", default="default")
    all_cmd.set_defaults(func=cmd_research_all)

    report = sub.add_parser("report", help="reports")
    rep_sub = report.add_subparsers(dest="rep_cmd", required=True)
    bt_rep = rep_sub.add_parser("backtest", help="diagnostic tearsheet from backtest artifacts")
    bt_rep.add_argument("--output-dir", default="artifacts/backtest/ma_baseline")
    bt_rep.set_defaults(func=cmd_report_backtest)

    ops = sub.add_parser("ops", help="operator tools")
    ops_sub = ops.add_subparsers(dest="ops_cmd", required=True)
    pre = ops_sub.add_parser("preflight", help="broker preflight")
    pre.add_argument("--config", default="configs/live/ma_baseline.yaml")
    pre.add_argument("--environment", default="configs/environments/paper.yaml")
    pre.set_defaults(func=lambda a: asyncio.run(_ops_preflight(a)))
    status = ops_sub.add_parser("status", help="risk state, kill-switch, run manifests")
    status.add_argument("--json", action="store_true", help="emit full JSON status")
    status.add_argument(
        "--broker-snapshot",
        action="store_true",
        help="include broker/account snapshot",
    )
    status.add_argument("--config", default="configs/live/ma_baseline.yaml")
    status.add_argument("--environment", default="configs/environments/paper.yaml")
    status.set_defaults(func=cmd_ops_status)
    brief = ops_sub.add_parser("brief", help="operator brief")
    brief.add_argument("--format", choices=("md", "json"), default="md")
    brief.set_defaults(func=cmd_ops_brief)
    kill = ops_sub.add_parser("kill-switch", help="activate or clear the local kill-switch")
    kill_mode = kill.add_mutually_exclusive_group(required=True)
    kill_mode.add_argument("--activate", action="store_true")
    kill_mode.add_argument("--clear", action="store_false", dest="activate")
    kill.add_argument("--state-file", default="state/ma_baseline_risk.json")
    kill.add_argument("--reason", default="operator")
    kill.set_defaults(func=cmd_ops_kill_switch)
    promote = ops_sub.add_parser("promote", help="explicit promotion stage transition")
    promote.add_argument("--from", dest="from_state", required=True)
    promote.add_argument("--to", dest="to_state", required=True)
    promote.add_argument("--manifest", required=True)
    promote.add_argument("--reason", default="operator")
    promote.add_argument("--force", action="store_true")
    promote.set_defaults(func=cmd_ops_promote)

    return p


def main(argv: list[str] | None = None) -> int:
    from sam.logging import configure_logging

    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
