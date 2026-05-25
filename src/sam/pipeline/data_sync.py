from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from sam.config.loader import (
    DataSyncConfig,
    DataValidateConfig,
    SamSettings,
    config_hash,
    ensure_dirs,
    load_universe,
    write_json,
)
from sam.logging import get_logger
from sam.manifests.run_manifest import (
    PromotionState,
    RunManifest,
    installed_dependency_versions,
)
from sam.pipeline.data_quality import validate_symbol_parquet, write_quality_report

log = get_logger(__name__)


def _build_market_data_spec(config: DataSyncConfig, out_dir: Path):
    from ml4t.specs import (
        ArtifactProvenance,
        ArtifactStorage,
        MarketDataSchema,
        MarketDataSemantics,
        MarketDataSpec,
    )

    fs = config.feed_spec
    artifact_id = f"sam_{config.output_subdir.replace('/', '_')}"
    return MarketDataSpec(
        artifact_id=artifact_id,
        storage=ArtifactStorage(path=str(out_dir), format="parquet"),
        provenance=ArtifactProvenance(created_by="sam.data.sync"),
        schema=MarketDataSchema(
            timestamp_col=fs.timestamp_col,
            entity_col=fs.entity_col,
            price_col=fs.price_col or fs.close_col,
            close_col=fs.close_col,
        ),
        semantics=MarketDataSemantics(
            data_frequency=fs.data_frequency,
            calendar=fs.calendar,
            timezone=fs.timezone,
        ),
    )


def _storage_data_manager(out_dir: Path):
    from ml4t.data import DataManager
    from ml4t.data.storage.backend import StorageConfig
    from ml4t.data.storage.hive import HiveStorage

    hive_dir = out_dir / ".hive"
    hive_dir.mkdir(parents=True, exist_ok=True)
    storage = HiveStorage(StorageConfig(base_path=str(hive_dir)))
    return DataManager(storage=storage)


def _sync_symbol_data(
    symbol: str,
    config: DataSyncConfig,
    out_dir: Path,
    end: str | None,
    fetch_dm,
    storage_dm,
) -> pl.DataFrame:
    path = out_dir / f"{symbol}.parquet"
    frequency = config.feed_spec.data_frequency or "daily"
    entity_col = config.feed_spec.entity_col or "symbol"

    use_storage = config.storage_mode == "hive" or config.incremental
    if use_storage and storage_dm is not None and config.incremental and path.exists():
        log.info("data.sync.incremental", symbol=symbol, path=str(path))
        try:
            storage_dm.update(symbol, provider=config.provider, frequency=frequency)
        except Exception:
            existing = pl.read_parquet(path)
            storage_dm.import_data(
                existing,
                symbol,
                config.provider,
                frequency=frequency,
            )
            storage_dm.update(symbol, provider=config.provider, frequency=frequency)
        end_date = end or datetime.now(tz=UTC).strftime("%Y-%m-%d")
        stacked = storage_dm.batch_load_from_storage(
            [symbol],
            config.start_date,
            end_date,
            frequency=frequency,
            provider=config.provider,
            fetch_missing=False,
        )
        if entity_col in stacked.columns:
            return stacked.filter(pl.col(entity_col) == symbol)
        return stacked

    df = fetch_dm.fetch(symbol, config.start_date, end, provider=config.provider)
    if use_storage and storage_dm is not None:
        storage_dm.import_data(df, symbol, config.provider, frequency=frequency)
    return df


def run_data_sync(config: DataSyncConfig, settings: SamSettings) -> dict:
    from ml4t.data import DataManager
    from ml4t.specs import FeedSpec, write_spec_payload

    ensure_dirs(settings)
    symbols = load_universe(config.universe_file)
    out_dir = settings.sam_data_dir / config.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = config.feed_spec
    feed_spec = FeedSpec(
        timestamp_col=fs.timestamp_col,
        entity_col=fs.entity_col,
        close_col=fs.close_col,
        price_col=fs.price_col,
        calendar=fs.calendar,
        timezone=fs.timezone,
        data_frequency=fs.data_frequency,
    )

    fetch_dm = DataManager()
    storage_dm = (
        _storage_data_manager(out_dir)
        if config.storage_mode == "hive" or config.incremental
        else None
    )
    end = config.end_date
    results: dict[str, str] = {}
    quality_reports: dict[str, str] = {}
    gate_failures: list[str] = []

    for symbol in symbols:
        df = _sync_symbol_data(symbol, config, out_dir, end, fetch_dm, storage_dm)
        path = out_dir / f"{symbol}.parquet"
        df.write_parquet(path)
        results[symbol] = str(path)
        log.info("data.sync.wrote", symbol=symbol, rows=df.height, path=str(path))

        report_payload = validate_symbol_parquet(
            path,
            symbol=symbol,
            provider=config.provider,
            frequency=fs.data_frequency or "daily",
            gates=config.quality_gates,
        )
        report_path = out_dir / "quality" / f"{symbol}.json"
        write_quality_report(report_path, report_payload)
        quality_reports[symbol] = str(report_path)
        if config.quality_gates.enforce and not report_payload["gates"]["passed"]:
            gate_failures.extend(
                f"{symbol}: {failure}" for failure in report_payload["gates"]["failures"]
            )

    feed_spec_path = write_spec_payload(feed_spec.to_dict(), out_dir / "data_spec.yaml")
    market_spec = _build_market_data_spec(config, out_dir)
    market_spec_path = write_spec_payload(market_spec.to_dict(), out_dir / "market_data_spec.yaml")

    summary = {
        "provider": config.provider,
        "start_date": config.start_date,
        "end_date": end,
        "symbols": symbols,
        "paths": results,
        "quality_reports": quality_reports,
        "data_spec": str(feed_spec_path),
        "market_data_spec": str(market_spec_path),
        "quality_gates": {
            "passed": not gate_failures,
            "failures": gate_failures,
        },
        "feed_spec": feed_spec.to_dict(),
        "storage_mode": config.storage_mode,
        "incremental": config.incremental,
    }
    write_json(out_dir / "sync_manifest.json", summary)

    manifest = RunManifest(
        strategy_id=config.output_subdir,
        config_path=f"data/sync:{config.output_subdir}",
        config_hash=config_hash(out_dir / "sync_manifest.json"),
        promotion_state=PromotionState.RESEARCH,
        environment="data",
        inputs={"universe": config.universe_file, **results},
        dependency_versions=installed_dependency_versions(),
        checks=summary["quality_gates"],
        quality_reports=quality_reports,
        ml4t_artifacts={
            "feed_spec": str(feed_spec_path),
            "market_data_spec": str(market_spec_path),
        },
        artifact_paths={"sync_dir": str(out_dir)},
    )
    manifest_path = out_dir / "run_manifest.json"
    manifest.save(manifest_path)
    summary["run_manifest"] = str(manifest_path)
    log.info(
        "data.sync.complete",
        symbols=len(symbols),
        passed=summary["quality_gates"]["passed"],
        manifest=str(manifest_path),
    )

    if gate_failures:
        raise RuntimeError(f"Data quality gates failed: {gate_failures}")
    return summary


def run_data_validate(config: DataValidateConfig, settings: SamSettings) -> dict:
    ensure_dirs(settings)
    data_dir = Path(config.data_dir) if config.data_dir else settings.sam_data_dir
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir

    symbols = load_universe(config.universe_file) if config.universe_file else []
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if symbols:
        symbol_paths = {symbol: data_dir / f"{symbol}.parquet" for symbol in symbols}
    else:
        symbol_paths = {path.stem: path for path in parquet_files}

    quality_reports: dict[str, str] = {}
    gate_failures: list[str] = []
    fs = config.feed_spec
    reports_dir = data_dir / "quality"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for symbol, path in symbol_paths.items():
        if not path.exists():
            gate_failures.append(f"{symbol}: missing parquet at {path}")
            continue
        report_payload = validate_symbol_parquet(
            path,
            symbol=symbol,
            provider=config.provider,
            frequency=fs.data_frequency or "daily",
            gates=config.quality_gates,
        )
        report_path = reports_dir / f"{symbol}.json"
        write_quality_report(report_path, report_payload)
        quality_reports[symbol] = str(report_path)
        if config.quality_gates.enforce and not report_payload["gates"]["passed"]:
            gate_failures.extend(
                f"{symbol}: {failure}" for failure in report_payload["gates"]["failures"]
            )

    summary = {
        "data_dir": str(data_dir),
        "symbols": list(symbol_paths.keys()),
        "quality_reports": quality_reports,
        "passed": not gate_failures,
        "failures": gate_failures,
    }
    write_json(reports_dir / "validation_summary.json", summary)

    manifest = RunManifest(
        strategy_id=str(data_dir),
        config_path=f"data/validate:{data_dir}",
        config_hash=config_hash(reports_dir / "validation_summary.json"),
        promotion_state=PromotionState.RESEARCH,
        environment="data",
        inputs={symbol: str(path) for symbol, path in symbol_paths.items()},
        dependency_versions=installed_dependency_versions(),
        checks={"passed": summary["passed"], "failures": gate_failures},
        quality_reports=quality_reports,
        artifact_paths={"validation_summary": str(reports_dir / "validation_summary.json")},
    )
    manifest.save(data_dir / "run_manifest.json")
    log.info("data.validate.complete", symbols=len(symbol_paths), passed=summary["passed"])

    if gate_failures and config.quality_gates.enforce:
        raise RuntimeError(f"Data validation failed: {gate_failures}")
    return summary
