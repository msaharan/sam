from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from sam.config.loader import QualityGatesConfig, write_json
from sam.logging import get_logger

log = get_logger(__name__)


def _normalize_ohlcv(df: pl.DataFrame, timestamp_col: str = "date") -> pl.DataFrame:
    frame = df
    if timestamp_col != "timestamp" and timestamp_col in frame.columns:
        frame = frame.rename({timestamp_col: "timestamp"})
    if "symbol" in frame.columns and "ticker" not in frame.columns:
        pass
    elif "ticker" in frame.columns and "symbol" not in frame.columns:
        frame = frame.rename({"ticker": "symbol"})
    return frame


def validate_symbol_parquet(
    path: Path,
    *,
    symbol: str,
    provider: str,
    frequency: str,
    gates: QualityGatesConfig,
) -> dict[str, Any]:
    """Run ML4T OHLCV validation and emit a DataQualityReport artifact."""
    try:
        from ml4t.diagnostic.integration import (
            DataAnomaly,
            DataQualityMetrics,
            DataQualityReport,
            Severity,
        )
    except ImportError as exc:
        raise ImportError(
            "Data quality reports require ml4t-diagnostic "
            "(install sam[research] or sam[all]; CMake may be required for llvmlite)"
        ) from exc
    from ml4t.data.validation import OHLCVValidator

    raw = pl.read_parquet(path)
    frame = _normalize_ohlcv(raw)
    validator = OHLCVValidator()
    result = validator.validate(frame)

    row_count = int(result.metadata.get("row_count") or frame.height or 0)
    null_issues = [i for i in result.issues if i.check == "null_values"]
    null_rows = sum(i.row_count or 0 for i in null_issues)
    completeness = 1.0 if row_count == 0 else max(0.0, 1.0 - (null_rows / row_count))

    ts_col = "timestamp" if "timestamp" in frame.columns else frame.columns[0]
    timestamps = frame[ts_col].drop_nulls().sort()
    if timestamps.is_empty():
        start = end = datetime.now(tz=UTC)
        timeliness_minutes = float("inf")
    else:
        start_val = timestamps.min()
        end_val = timestamps.max()
        start = _as_datetime(start_val)
        end = _as_datetime(end_val)
        timeliness_minutes = max(0.0, (datetime.now(tz=UTC) - end).total_seconds() / 60.0)

    severity_map = {
        "critical": Severity.CRITICAL,
        "error": Severity.ERROR,
        "warning": Severity.WARNING,
        "info": Severity.INFO,
    }
    anomalies: list[DataAnomaly] = []
    for issue in result.issues:
        anomalies.append(
            DataAnomaly(
                anomaly_type=_issue_to_anomaly_type(issue.check),
                severity=severity_map.get(issue.severity.value, Severity.WARNING),
                timestamp=issue.timestamp,
                symbol=symbol,
                description=issue.message,
            )
        )

    metrics = DataQualityMetrics(
        completeness=completeness,
        timeliness=timeliness_minutes,
        accuracy_score=1.0 if result.error_count == 0 and result.critical_count == 0 else 0.85,
        consistency_score=1.0 if result.passed else 0.7,
        n_records=row_count,
        n_anomalies=len(anomalies),
        n_critical=result.critical_count,
        n_error=result.error_count,
        n_warning=result.warning_count,
    )
    recommendations: list[str] = []
    if not result.passed:
        recommendations.append("Resolve validation errors before promotion.")
    if timeliness_minutes > gates.max_staleness_minutes:
        recommendations.append(
            f"Data staleness {timeliness_minutes:.1f}m exceeds limit "
            f"{gates.max_staleness_minutes:.1f}m."
        )

    report = DataQualityReport(
        symbol=symbol,
        source=provider,
        date_range=(start, end),
        frequency=frequency,
        metrics=metrics,
        anomalies=anomalies,
        recommendations=recommendations,
        is_production_ready=result.passed
        and metrics.completeness >= gates.min_completeness
        and metrics.n_critical <= gates.max_critical_anomalies
        and metrics.n_error <= gates.max_errors
        and timeliness_minutes <= gates.max_staleness_minutes,
    )
    payload = report.model_dump(mode="json")
    payload["gates"] = evaluate_quality_gates(report, gates)
    log.info(
        "data.quality.validated",
        symbol=symbol,
        path=str(path),
        passed=payload["gates"]["passed"],
        completeness=metrics.completeness,
    )
    return payload


def evaluate_quality_gates(report_payload: dict[str, Any] | Any, gates: QualityGatesConfig) -> dict:
    if hasattr(report_payload, "model_dump"):
        report_payload = report_payload.model_dump(mode="json")
    metrics = report_payload.get("metrics", {})
    failures: list[str] = []
    completeness = float(metrics.get("completeness", 0))
    n_critical = int(metrics.get("n_critical", 0))
    n_error = int(metrics.get("n_error", 0))
    timeliness = float(metrics.get("timeliness", 0))

    if completeness < gates.min_completeness:
        failures.append(
            f"completeness {completeness:.3f} < min_completeness {gates.min_completeness}"
        )
    if n_critical > gates.max_critical_anomalies:
        failures.append(
            f"n_critical {n_critical} > max_critical_anomalies {gates.max_critical_anomalies}"
        )
    if n_error > gates.max_errors:
        failures.append(f"n_error {n_error} > max_errors {gates.max_errors}")
    if timeliness > gates.max_staleness_minutes:
        failures.append(
            f"staleness_minutes {timeliness:.1f} > max_staleness_minutes "
            f"{gates.max_staleness_minutes:.1f}"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "min_completeness": gates.min_completeness,
        "max_errors": gates.max_errors,
        "max_critical_anomalies": gates.max_critical_anomalies,
        "max_staleness_minutes": gates.max_staleness_minutes,
    }


def write_quality_report(path: Path, payload: dict[str, Any]) -> Path:
    write_json(path, payload)
    return path


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


def _issue_to_anomaly_type(check: str) -> Any:
    from ml4t.diagnostic.integration import AnomalyType

    mapping = {
        "required_columns": AnomalyType.MISSING_DATA,
        "null_values": AnomalyType.MISSING_DATA,
        "price_consistency": AnomalyType.OHLC_VIOLATION,
        "negative_prices": AnomalyType.NEGATIVE_PRICE,
        "negative_volume": AnomalyType.ZERO_VOLUME,
        "duplicate_timestamps": AnomalyType.DUPLICATE_TIMESTAMP,
        "chronological_order": AnomalyType.TIMESTAMP_GAP,
        "price_staleness": AnomalyType.STALE_DATA,
        "extreme_returns": AnomalyType.PRICE_SPIKE,
    }
    return mapping.get(check, AnomalyType.OUTLIER)
