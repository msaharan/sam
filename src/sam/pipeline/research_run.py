from __future__ import annotations

from pathlib import Path

import polars as pl

from sam.config.loader import ModelConfig, SamSettings, config_hash, ensure_dirs, write_json
from sam.logging import get_logger
from sam.manifests.run_manifest import PromotionState, RunManifest, installed_dependency_versions
from sam.pipeline.adapters import panel_to_training_batch, weights_to_signal_scores

log = get_logger(__name__)

def run_features(
    prices_path: str,
    output_dir: Path,
    feature_names: list[str] | None = None,
) -> Path:
    try:
        from ml4t.engineer import compute_features
    except ImportError as exc:
        raise ImportError(
            "sam[research] requires ml4t-engineer "
            "(needs CMake for numba/llvmlite on some platforms)"
        ) from exc

    names = feature_names or ["rsi", "macd", "atr"]
    prices = pl.read_parquet(prices_path)
    if "symbol" not in prices.columns and "ticker" in prices.columns:
        prices = prices.rename({"ticker": "symbol"})

    frames: list[pl.DataFrame] = []
    entity_col = "symbol" if "symbol" in prices.columns else "ticker"
    for symbol in prices[entity_col].unique().to_list():
        sym_df = prices.filter(pl.col(entity_col) == symbol).sort("date")
        feat = compute_features(sym_df, names)
        feat = feat.with_columns(pl.lit(symbol).alias(entity_col))
        frames.append(feat)

    out = pl.concat(frames, how="diagonal_relaxed")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "features.parquet"
    out.write_parquet(path)
    log.info(
        "research.features.complete",
        path=str(path),
        rows=out.height,
        symbols=out[entity_col].n_unique(),
    )
    return path


def run_diagnose(prices_path: str, features_path: str, output_dir: Path) -> dict:
    try:
        from ml4t.diagnostic import analyze_signal
    except ImportError as exc:
        raise ImportError("sam[research] requires ml4t-diagnostic") from exc

    prices = pl.read_parquet(prices_path)
    features = pl.read_parquet(features_path)
    entity = "symbol" if "symbol" in prices.columns else "ticker"
    date_col = "date" if "date" in prices.columns else "timestamp"

    factor_col = "rsi" if "rsi" in features.columns else features.columns[-1]
    factor = features.select(
        pl.col(date_col).alias("date"),
        pl.col(entity).alias("asset"),
        pl.col(factor_col).alias("factor"),
    )
    price_panel = prices.select(
        pl.col(date_col).alias("date"),
        pl.col(entity).alias("asset"),
        pl.col("close").alias("price"),
    )
    result = analyze_signal(factor=factor, prices=price_panel, periods=(1, 5))
    summary = {
        "ic_1d": float(result.ic.get("1D", 0)),
        "ic_t_1d": float(result.ic_t_stat.get("1D", 0)),
        "spread_1d": float(result.spread.get("1D", 0)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "diagnostics.json", summary)
    log.info("research.diagnose.complete", ic_1d=summary["ic_1d"], output_dir=str(output_dir))
    return summary


def run_train(
    prices_path: str,
    features_path: str,
    output_dir: Path,
    strategy_id: str = "signal_rank",
    model_config: ModelConfig | None = None,
) -> Path:
    """Train a baseline signal model via ML4T PortfolioAllocationPipeline."""
    try:
        from ml4t.models import (
            LinearFeaturePortfolioModel,
            LinearPortfolioConfig,
            PortfolioAllocationPipeline,
        )
        from ml4t.models.integration import (
            signals_frame_from_portfolio_weights,
            write_backtest_frames,
        )
    except ImportError as exc:
        raise ImportError("sam[research] requires ml4t-models") from exc

    cfg = model_config or ModelConfig()
    prices = pl.read_parquet(prices_path)
    features = pl.read_parquet(features_path)
    entity = "symbol" if "symbol" in prices.columns else "ticker"
    date_col = "date" if "date" in prices.columns else "timestamp"
    feature_entity = "symbol" if "symbol" in features.columns else "ticker"
    feature_date = "date" if "date" in features.columns else "timestamp"

    price_targets = (
        prices.select(
            pl.col(date_col).alias("date"),
            pl.col(entity).alias("symbol"),
            pl.col("close"),
        )
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("close").pct_change().shift(-1).over("symbol").alias("forward_return")
        )
    )
    panel = (
        features.rename({feature_date: "date", feature_entity: "symbol"})
        .join(price_targets, on=["date", "symbol"], how="inner")
        .sort(["symbol", "date"])
    )
    excluded = {"date", "symbol", "close", "forward_return"}
    feature_cols = [
        col
        for col, dtype in panel.schema.items()
        if col not in excluded and dtype.is_numeric()
    ]
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found in {features_path}")

    batch, train_dates, test_dates, _, _ = panel_to_training_batch(
        panel,
        feature_cols,
        entity_col="symbol",
        date_col="date",
        train_fraction=cfg.train_fraction,
        batch_type=cfg.batch_type,
    )

    pipeline = PortfolioAllocationPipeline(
        LinearFeaturePortfolioModel(
            LinearPortfolioConfig(
                ridge_alpha=cfg.ridge_alpha,
                fit_intercept=cfg.fit_intercept,
                gross_exposure=cfg.gross_exposure,
                net_exposure=cfg.net_exposure,
                max_abs_weight=cfg.max_abs_weight,
            )
        )
    )
    fit_result = pipeline.fit(batch)
    prediction = pipeline.predict(batch)
    signals_frame = signals_frame_from_portfolio_weights(prediction.processed_weights)

    output_dir.mkdir(parents=True, exist_ok=True)
    signals = weights_to_signal_scores(signals_frame)
    path = output_dir / "signals.parquet"
    signals.write_parquet(path)

    model_meta = {
        "model_type": cfg.model_type,
        "strategy_id": strategy_id,
        "features": feature_cols,
        "train_fraction": cfg.train_fraction,
        "train_end": str(train_dates[-1]) if train_dates else None,
        "test_start": str(test_dates[0]) if test_dates else None,
        "fit_summary": {
            "converged": fit_result.model_fit.converged,
            "train_metrics": fit_result.model_fit.train_metrics,
        },
    }
    model_path = output_dir / "model.json"
    write_json(model_path, model_meta)

    frames_dir = output_dir / "ml4t_frames"
    written = write_backtest_frames(frames_dir, predictions=None)
    signals_frame.write_parquet(frames_dir / "signals_frame.parquet")
    written["signals"] = frames_dir / "signals_frame.parquet"

    diagnostics = {
        "passed": fit_result.model_fit.converged,
        "train_rows": len(train_dates),
        "test_rows": len(test_dates),
        "model_type": cfg.model_type,
    }

    manifest = RunManifest(
        strategy_id=strategy_id,
        config_path="research/train",
        config_hash=config_hash(model_path),
        promotion_state=PromotionState.RESEARCH,
        environment="research",
        inputs={"prices": prices_path, "features": features_path},
        config_hashes={
            "prices": config_hash(prices_path),
            "features": config_hash(features_path),
            "model": config_hash(model_path),
        },
        dependency_versions=installed_dependency_versions(),
        checks=diagnostics,
        artifact_paths={
            "signals": str(path),
            "features": features_path,
            "model": str(model_path),
        },
        model_artifacts={
            "model": str(model_path),
            "signals_frame": str(written["signals"]),
        },
        ml4t_artifacts={key: str(value) for key, value in written.items()},
        diagnostic_reports={"train": str(output_dir / "diagnostics.json")},
    )
    manifest.save(output_dir / "run_manifest.json")
    log.info(
        "research.train.complete",
        strategy_id=strategy_id,
        signals=str(path),
        converged=fit_result.model_fit.converged,
        batch_type=cfg.batch_type,
    )
    return path


def research_pipeline(settings: SamSettings, prices_path: str, run_id: str = "default") -> dict:
    ensure_dirs(settings)
    base = settings.sam_artifacts_dir / "research" / run_id
    feat_path = run_features(prices_path, base / "features")
    diag = run_diagnose(prices_path, str(feat_path), base / "diagnostics")
    sig_path = run_train(prices_path, str(feat_path), base / "signals")
    return {
        "features": str(feat_path),
        "diagnostics": diag,
        "signals": str(sig_path),
    }
