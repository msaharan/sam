from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.config import DataFrequency

from sam.config.loader import BacktestRunConfig, SamSettings, config_hash, ensure_dirs, write_json
from sam.logging import get_logger
from sam.manifests.run_manifest import (
    PromotionState,
    RunManifest,
    installed_dependency_versions,
)
from sam.strategies.registry import build_strategy

log = get_logger(__name__)

def run_backtest(
    config: BacktestRunConfig,
    settings: SamSettings,
    root: Path | None = None,
) -> dict:
    ensure_dirs(settings)
    root = root or Path.cwd()
    strategy = build_strategy(config.strategy_config, root=root)

    prices = pl.read_parquet(config.prices_path)
    date_col = "date" if "date" in prices.columns else "timestamp"
    entity_col = "symbol" if "symbol" in prices.columns else "ticker"
    prices = _filter_window(prices, date_col, config.start_date, config.end_date)
    signals = pl.read_parquet(config.signals_path) if config.signals_path else None
    if signals is not None:
        signals = _filter_window(signals, date_col, config.start_date, config.end_date)
        signals = _lag_signals(signals, date_col, entity_col, config.signal_lag_bars)

    feed = _build_datafeed(prices, signals, config, date_col, entity_col)

    bt_config = replace(
        BacktestConfig.from_preset(config.preset),
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate,
        calendar=config.calendar,
        timezone=config.timezone,
        data_frequency=DataFrequency(config.data_frequency),
    )

    engine = Engine(feed, strategy, bt_config)
    result = engine.run()

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out)
    metrics_path = out / "metrics.json"
    write_json(metrics_path, dict(result.metrics))

    cfg_path = (
        root / config.strategy_config
        if not Path(config.strategy_config).is_absolute()
        else Path(config.strategy_config)
    )
    window_start = str(prices[date_col].min())
    window_end = str(prices[date_col].max())
    checks, diagnostic_reports = _promotion_checks(result, config)
    promotion_state = (
        PromotionState.BACKTEST_PASSED if checks["passed"] else PromotionState.RESEARCH
    )

    config_hashes = {"strategy": config_hash(cfg_path)}
    if Path(config.prices_path).exists():
        config_hashes["prices"] = config_hash(config.prices_path)
    if config.signals_path and Path(config.signals_path).exists():
        config_hashes["signals"] = config_hash(config.signals_path)

    manifest = RunManifest(
        strategy_id=load_strategy_id(config.strategy_config, root),
        config_path=str(cfg_path),
        config_hash=config_hash(cfg_path),
        promotion_state=promotion_state,
        environment="backtest",
        data_window_start=window_start,
        data_window_end=window_end,
        inputs={
            "prices": str(Path(config.prices_path)),
            **({"signals": str(Path(config.signals_path))} if config.signals_path else {}),
        },
        config_hashes=config_hashes,
        dependency_versions=installed_dependency_versions(),
        checks=checks,
        diagnostic_reports=diagnostic_reports,
        artifact_paths={
            "parquet_dir": str(out),
            "metrics": str(metrics_path),
            "trades": str(out / "trades.parquet"),
            "fills": str(out / "fills.parquet"),
            "equity": str(out / "equity.parquet"),
        },
        notes=(
            f"preset={config.preset}; calendar={config.calendar}; timezone={config.timezone}; "
            f"commission_rate={config.commission_rate}; slippage_rate={config.slippage_rate}; "
            f"corporate_actions={config.corporate_actions}; "
            f"signal_lag_bars={config.signal_lag_bars}"
        ),
    )
    manifest.save(out / "run_manifest.json")
    enforce = config.enforce_promotion_gates and config.promotion_gates.enforce
    if enforce and not checks["passed"]:
        raise RuntimeError(f"Backtest promotion gates failed: {checks['failures']}")
    return {"metrics": result.metrics, "output_dir": str(out)}


def _build_datafeed(
    prices: pl.DataFrame,
    signals: pl.DataFrame | None,
    config: BacktestRunConfig,
    date_col: str,
    entity_col: str,
) -> DataFeed:
    common_kwargs = {
        "timestamp_col": date_col,
        "entity_col": entity_col,
        "close_col": "close",
        "calendar": config.calendar,
        "timezone": config.timezone,
        "data_frequency": config.data_frequency,
    }
    ml4t_signals = _load_ml4t_signals_frame(config, date_col, entity_col)
    try:
        from ml4t.models.integration import backtest_datafeed_inputs, backtest_inputs_from_weights
    except ImportError:
        from ml4t.specs import FeedSpec

        feed_spec = FeedSpec(
            timestamp_col=date_col,
            entity_col=entity_col,
            close_col="close",
            price_col="close",
            calendar=config.calendar,
            timezone=config.timezone,
            data_frequency=config.data_frequency,
        )
        return DataFeed(prices_df=prices, signals_df=signals, feed_spec=feed_spec)

    weights_frame_path = _ml4t_weights_frame_path(config)
    if weights_frame_path is not None:
        try:
            weights_frame = pl.read_parquet(weights_frame_path)
            from ml4t.models.integration.surfaces import WeightsFrame

            rows = tuple(tuple(row) for row in weights_frame.iter_rows())
            weights = WeightsFrame(
                columns=tuple(weights_frame.columns),
                rows=rows,
                metadata={"source": str(weights_frame_path)},
            )
            inputs = backtest_inputs_from_weights(
                weights,
                prices_frame=prices,
                **common_kwargs,
            )
            log.info(
                "backtest.handoff",
                source="ml4t.models.integration.backtest_inputs_from_weights",
            )
            return DataFeed(**inputs.to_datafeed_kwargs())
        except Exception as exc:
            log.warning("backtest.handoff.weights_fallback", error=str(exc))

    signal_input = ml4t_signals
    if signal_input is None and signals is not None:
        signal_input = _signals_frame_from_polars(signals, date_col, entity_col)

    if signal_input is not None:
        inputs = backtest_datafeed_inputs(
            prices_frame=prices,
            signals=signal_input,
            **common_kwargs,
        )
    else:
        inputs = backtest_datafeed_inputs(prices_frame=prices, **common_kwargs)
    return DataFeed(**inputs.to_datafeed_kwargs())


def _ml4t_weights_frame_path(config: BacktestRunConfig) -> Path | None:
    if not config.signals_path:
        return None
    path = Path(config.signals_path)
    if path.is_dir():
        candidate = path / "weights_frame.parquet"
    else:
        candidate = path.parent / "ml4t_frames" / "weights_frame.parquet"
    return candidate if candidate.exists() else None


def _load_ml4t_signals_frame(
    config: BacktestRunConfig,
    date_col: str,
    entity_col: str,
):
    if not config.signals_path:
        return None
    path = Path(config.signals_path)
    if path.is_dir():
        candidate = path / "signals_frame.parquet"
    else:
        candidate = path.parent / "ml4t_frames" / "signals_frame.parquet"
    if not candidate.exists():
        return None
    from ml4t.models.integration.surfaces import SignalsFrame

    frame = pl.read_parquet(candidate)
    rows = tuple(tuple(row) for row in frame.iter_rows())
    return SignalsFrame(
        columns=tuple(frame.columns),
        rows=rows,
        metadata={"source": str(candidate), "frame_type": "signal"},
    )


def _signals_frame_from_polars(signals: pl.DataFrame, date_col: str, entity_col: str):
    from ml4t.models.integration.surfaces import SignalsFrame

    candidates = ("score", "signal_value", "weight", "prediction_value")
    value_col = next((c for c in candidates if c in signals.columns), signals.columns[-1])
    rows = []
    for row in signals.iter_rows(named=True):
        rows.append(
            (
                row[date_col],
                row[entity_col],
                float(row[value_col]),
                True,
            )
        )
    return SignalsFrame(
        columns=(date_col, entity_col, "signal_value", "selected"),
        rows=tuple(rows),
        metadata={"frame_type": "signal", "source": "sam_parquet"},
    )


def load_strategy_id(strategy_config: str, root: Path) -> str:
    from sam.config.loader import StrategyConfig, load_yaml

    cfg_path = (
        root / strategy_config if not Path(strategy_config).is_absolute() else strategy_config
    )
    raw = load_yaml(cfg_path)
    return StrategyConfig.model_validate(raw).strategy


def _filter_window(
    frame: pl.DataFrame,
    date_col: str,
    start_date: str | None,
    end_date: str | None,
) -> pl.DataFrame:
    if start_date:
        frame = frame.filter(pl.col(date_col) >= pl.lit(start_date).str.to_datetime())
    if end_date:
        frame = frame.filter(pl.col(date_col) <= pl.lit(end_date).str.to_datetime())
    return frame


def _lag_signals(
    signals: pl.DataFrame,
    date_col: str,
    entity_col: str,
    signal_lag_bars: int,
) -> pl.DataFrame:
    if signal_lag_bars <= 0:
        return signals
    signal_cols = [c for c in signals.columns if c not in (date_col, entity_col)]
    expressions = []
    for col in signal_cols:
        dtype = signals.schema[col]
        if dtype.is_numeric():
            expressions.append(
                pl.col(col).shift(signal_lag_bars).over(entity_col).fill_null(0.5).alias(col)
            )
    if not expressions:
        return signals
    return signals.sort([entity_col, date_col]).with_columns(expressions)


def _promotion_checks(result, config: BacktestRunConfig) -> tuple[dict, dict[str, str]]:
    gates = config.promotion_gates
    min_trades = gates.min_trades if gates.min_trades is not None else config.min_trades
    max_drawdown = gates.max_drawdown if gates.max_drawdown is not None else config.max_drawdown
    min_total_return = (
        gates.min_total_return if gates.min_total_return is not None else config.min_total_return
    )

    diagnostic_reports: dict[str, str] = {}
    metrics = dict(result.metrics)
    try:
        from ml4t.diagnostic.integration import analyze_backtest_result, compute_metrics_from_result

        profile = analyze_backtest_result(result, calendar=config.calendar)
        diagnostic_metrics = compute_metrics_from_result(result, calendar=config.calendar)
        metrics.update({k: v for k, v in diagnostic_metrics.items() if v is not None})
        diagnostic_reports["backtest_profile"] = str(
            Path(config.output_dir) / "diagnostic_profile.json"
        )
        write_json(
            Path(config.output_dir) / "diagnostic_profile.json",
            {"summary": profile.summary() if hasattr(profile, "summary") else str(profile)},
        )
    except ImportError:
        pass

    dsr_probability: float | None = None
    if gates.min_dsr_probability is not None:
        try:
            from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

            daily_returns = result.to_daily_returns(calendar=config.calendar).to_numpy()
            if len(daily_returns) > 1:
                kwargs = {"frequency": "daily"}
                if gates.n_trials > 1:
                    kwargs["effective_trials"] = float(gates.n_trials)
                dsr_result = deflated_sharpe_ratio(daily_returns, **kwargs)
                dsr_probability = float(dsr_result.probability)
                metrics["dsr_probability"] = dsr_probability
        except ImportError:
            pass

    failures: list[str] = []
    trades = int(metrics.get("num_trades") or metrics.get("num_fills") or 0)
    drawdown = float(metrics.get("max_drawdown") or 0)
    total_return = float(metrics.get("total_return") or 0)
    if trades < min_trades:
        failures.append(f"num_trades {trades} < min_trades {min_trades}")
    if max_drawdown is not None and drawdown > max_drawdown:
        failures.append(f"max_drawdown {drawdown} > limit {max_drawdown}")
    if min_total_return is not None and total_return < min_total_return:
        failures.append(f"total_return {total_return} < min_total_return {min_total_return}")
    if gates.min_dsr_probability is not None and dsr_probability is not None:
        if dsr_probability < gates.min_dsr_probability:
            failures.append(
                f"dsr_probability {dsr_probability:.3f} < min_dsr_probability "
                f"{gates.min_dsr_probability:.3f}"
            )
    return (
        {
            "passed": not failures,
            "failures": failures,
            "num_trades": trades,
            "max_drawdown": drawdown,
            "total_return": total_return,
            "dsr_probability": dsr_probability,
            "signal_lag_bars": config.signal_lag_bars,
            "source": "ml4t.diagnostic.integration" if diagnostic_reports else "sam.metrics",
        },
        diagnostic_reports,
    )
