"""Experiment contracts and registry helpers for SAM research workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from sam.config import load_toml
from sam.io import ensure_dir, write_table

PRIVATE_PATH_MARKERS = {
    ".env",
    "api_key",
    "broker",
    "execution",
    "live",
    "orders",
    "personal",
    "private",
    "secret",
    "signals",
    "tax",
}


class DataContract(BaseModel):
    source: str
    universe: str | None = None
    symbols: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=lambda: ["adj_close"])
    adjusted_policy: str = "use adjusted prices for return and portfolio calculations"
    timestamp_convention: str = "features use information available at or before signal timestamp"
    missing_data_policy: str = "drop rows only after feature and target construction"
    start: str | None = None
    end: str | None = None

    @field_validator("symbols", "fields")
    @classmethod
    def require_non_empty_lists(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("contract list fields must not be empty")
        return values


class FeatureContract(BaseModel):
    families: list[str]
    lookback_windows_days: list[int] = Field(default_factory=list)
    leakage_policy: str = "features must be computed from data available no later than signal date"
    static_identity_policy: Literal["include", "exclude", "diagnostic_only"] = "diagnostic_only"
    feature_count: int | None = Field(default=None, ge=0)

    @field_validator("families")
    @classmethod
    def require_feature_families(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("at least one feature family is required")
        return values


class TargetContract(BaseModel):
    name: str
    task: Literal["classification", "regression", "ranking", "risk_scoring"]
    horizon_days: int = Field(ge=1)
    label_definition: str
    execution_convention: str
    target_timestamp: str = "future outcome after signal timestamp"
    allow_overlapping_labels: bool = True


class ValidationWindow(BaseModel):
    name: str
    start: str
    end: str
    purpose: str

    @model_validator(mode="after")
    def require_ordered_dates(self) -> ValidationWindow:
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError(f"validation window {self.name!r} starts after it ends")
        return self


class ValidationContract(BaseModel):
    split_policy: Literal["fixed_chronological", "walk_forward", "rolling", "expanding"]
    windows: list[ValidationWindow] = Field(default_factory=list)
    embargo_days: int = Field(default=0, ge=0)
    overlap_policy: str = "document overlapping targets and avoid random splits"
    train_window_days: int | None = Field(default=None, ge=1)
    test_window_days: int | None = Field(default=None, ge=1)
    step_days: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def require_chronological_windows(self) -> ValidationContract:
        ordered = sorted(self.windows, key=lambda item: pd.Timestamp(item.start))
        if ordered != self.windows:
            raise ValueError("validation windows must be listed chronologically")
        previous_end: pd.Timestamp | None = None
        for window in self.windows:
            start = pd.Timestamp(window.start)
            if previous_end is not None and start <= previous_end:
                raise ValueError("validation windows must not overlap")
            previous_end = pd.Timestamp(window.end)
        return self


class DecisionContract(BaseModel):
    workflow: Literal["alert_queue", "allocation", "ranking", "risk_dashboard", "research_only"]
    allocation_rule: str | None = None
    alert_threshold_policy: str | None = None
    cost_model_bps: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=0.0, ge=0)
    turnover_limit: float | None = Field(default=None, ge=0)
    risk_controls: list[str] = Field(default_factory=list)


class ModelContract(BaseModel):
    baselines: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    selection_policy: str = "compare against domain baselines before promoting model complexity"

    @model_validator(mode="after")
    def require_model_or_baseline(self) -> ModelContract:
        if not self.baselines and not self.candidates:
            raise ValueError("at least one baseline or candidate model is required")
        return self


class ExperimentSpec(BaseModel):
    id: str
    name: str
    hypothesis: str
    status: Literal["planned", "active", "published", "archived"] = "planned"
    owner: str = "Mohit Saharan"
    visibility: Literal["public", "private", "open_core"] = "open_core"
    blog_post: str | None = None
    notebook: str | None = None
    tags: list[str] = Field(default_factory=list)
    data: DataContract
    features: FeatureContract
    target: TargetContract
    validation: ValidationContract
    decision: DecisionContract
    models: ModelContract
    metrics: list[str]
    artifacts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sam_takeaway: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value or any(character.isspace() for character in value):
            raise ValueError("experiment id must be non-empty and contain no whitespace")
        return value

    @field_validator("metrics")
    @classmethod
    def require_metrics(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("at least one metric is required")
        return values


@dataclass(frozen=True)
class ExperimentBundle:
    out_dir: Path
    manifest: Path
    experiment_json: Path
    contracts_markdown: Path
    data_summary: Path
    split_summary: Path
    model_leaderboard: Path
    calibration_diagnostics: Path
    operating_points: Path
    decision_translation: Path
    cost_turnover_sensitivity: Path
    artifact_manifest: Path
    figure_manifest: Path
    limitations_markdown: Path
    publication_checklist: Path


def load_experiment(path: str | Path) -> ExperimentSpec:
    return ExperimentSpec.model_validate(load_toml(path))


class ExperimentRegistry:
    """File-backed registry of TOML experiment specifications."""

    def __init__(self, root: str | Path = "configs/experiments") -> None:
        self.root = Path(root)

    def paths(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("*.toml"))

    def specs(self) -> list[ExperimentSpec]:
        return [load_experiment(path) for path in self.paths()]

    def get(self, experiment_id: str) -> ExperimentSpec:
        for spec in self.specs():
            if spec.id == experiment_id:
                return spec
        raise KeyError(f"unknown experiment id: {experiment_id}")

    def index_frame(self) -> pd.DataFrame:
        rows = []
        for spec in self.specs():
            rows.append(
                {
                    "id": spec.id,
                    "name": spec.name,
                    "status": spec.status,
                    "visibility": spec.visibility,
                    "target": spec.target.name,
                    "task": spec.target.task,
                    "decision_workflow": spec.decision.workflow,
                    "blog_post": spec.blog_post,
                    "tags": ",".join(spec.tags),
                }
            )
        return pd.DataFrame(rows).sort_values("id") if rows else pd.DataFrame()

    def validation_issues(self) -> pd.DataFrame:
        return validate_experiment_specs(self.specs())


def validate_experiment_specs(specs: list[ExperimentSpec]) -> pd.DataFrame:
    """Return registry-level contract issues that are not visible in one TOML file."""

    rows = []
    ids = [spec.id for spec in specs]
    for experiment_id in sorted({item for item in ids if ids.count(item) > 1}):
        rows.append(
            {
                "experiment_id": experiment_id,
                "severity": "error",
                "field": "id",
                "message": "duplicate experiment id",
            }
        )
    for spec in specs:
        if spec.status == "published" and not spec.blog_post:
            rows.append(
                _issue(
                    spec.id,
                    "error",
                    "blog_post",
                    "published experiments need a blog link",
                )
            )
        if spec.visibility in {"public", "open_core"}:
            for field, value in _path_like_fields(spec).items():
                if value and _has_private_marker(value):
                    rows.append(
                        _issue(
                            spec.id,
                            "error",
                            field,
                            "public/open-core experiment references private, live, "
                            "or secret-like path",
                        )
                    )
        if spec.decision.workflow == "allocation" and spec.decision.allocation_rule is None:
            rows.append(
                _issue(
                    spec.id,
                    "error",
                    "decision.allocation_rule",
                    "allocation workflow needs an allocation rule",
                )
            )
        if spec.decision.workflow == "alert_queue" and spec.decision.alert_threshold_policy is None:
            rows.append(
                _issue(
                    spec.id,
                    "error",
                    "decision.alert_threshold_policy",
                    "alert workflow needs a threshold policy",
                )
            )
    return pd.DataFrame(rows, columns=["experiment_id", "severity", "field", "message"])


def validate_research_frame(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    symbol_column: str | None = "symbol",
    target_date_column: str | None = None,
    feature_available_column: str | None = None,
    expected_frequency: str | None = None,
) -> dict[str, bool | int]:
    """Validate common leakage and chronology invariants for research tables."""

    if date_column not in frame.columns:
        raise ValueError(f"missing date column: {date_column}")
    dates = pd.to_datetime(frame[date_column])
    duplicate_keys = 0
    missing_observations = 0
    monotonic_dates = bool(
        dates.sort_values(kind="stable").reset_index(drop=True).equals(dates.reset_index(drop=True))
    )
    if symbol_column and symbol_column in frame.columns:
        duplicate_keys = int(frame.duplicated([date_column, symbol_column]).sum())
        panel_frame = frame.assign(_sam_date=dates)
        monotonic_by_symbol = panel_frame.groupby(symbol_column, sort=False)["_sam_date"].apply(
            lambda item: item.is_monotonic_increasing
        )
        monotonic_dates = bool(monotonic_by_symbol.all())
        ordered = panel_frame.sort_values(
            [symbol_column, "_sam_date"],
            kind="stable",
        )
        if expected_frequency:
            for _, group in ordered.groupby(symbol_column):
                clean_dates = pd.DatetimeIndex(group["_sam_date"].dropna().unique()).sort_values()
                if len(clean_dates) < 2:
                    continue
                expected = pd.date_range(
                    clean_dates.min(),
                    clean_dates.max(),
                    freq=expected_frequency,
                )
                missing_observations += int(len(expected.difference(clean_dates)))
    result: dict[str, bool | int] = {
        "rows": int(len(frame)),
        "missing_dates": bool(dates.isna().any()),
        "monotonic_dates": monotonic_dates,
        "duplicate_keys": duplicate_keys,
        "missing_observations": missing_observations,
    }
    if target_date_column:
        if target_date_column not in frame.columns:
            raise ValueError(f"missing target date column: {target_date_column}")
        target_dates = pd.to_datetime(frame[target_date_column])
        comparable = target_dates.notna() & dates.notna()
        leakage = comparable & (target_dates <= dates)
        result["target_after_signal"] = bool(not leakage.any() and comparable.all())
        result["target_leakage_rows"] = int(leakage.sum())
    if feature_available_column:
        if feature_available_column not in frame.columns:
            raise ValueError(f"missing feature availability column: {feature_available_column}")
        available_dates = pd.to_datetime(frame[feature_available_column])
        comparable = available_dates.notna() & dates.notna()
        future_features = comparable & (available_dates > dates)
        result["features_available_by_signal"] = bool(
            not future_features.any() and comparable.all()
        )
        result["future_feature_rows"] = int(future_features.sum())
    return result


def _path_like_fields(spec: ExperimentSpec) -> dict[str, str | None]:
    return {
        "blog_post": spec.blog_post,
        "notebook": spec.notebook,
        **{f"artifact[{idx}]": artifact for idx, artifact in enumerate(spec.artifacts)},
    }


def _has_private_marker(value: str) -> bool:
    normalized = value.lower().replace("\\", "/")
    parts = {
        item
        for part in normalized.split("/")
        for item in part.replace("-", "_").replace(".", "_").split("_")
    }
    return bool(parts & PRIVATE_PATH_MARKERS)


def _issue(experiment_id: str, severity: str, field: str, message: str) -> dict[str, str]:
    return {
        "experiment_id": experiment_id,
        "severity": severity,
        "field": field,
        "message": message,
    }


def write_experiment_bundle(
    spec: ExperimentSpec,
    out_dir: str | Path,
    *,
    artifact_paths: list[str | Path] | None = None,
) -> ExperimentBundle:
    target = ensure_dir(out_dir)
    artifact_paths = artifact_paths or []
    artifact_tables = _load_artifact_tables(artifact_paths)
    experiment_json = target / "experiment.json"
    experiment_json.write_text(
        json.dumps(spec.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    contracts_markdown = target / "contracts.md"
    contracts_markdown.write_text(_contracts_markdown(spec), encoding="utf-8")

    data_summary = write_table(_data_summary(spec), target / "data_summary.csv")
    split_summary = write_table(_split_summary(spec), target / "split_summary.csv")
    model_leaderboard = write_table(
        _model_leaderboard(spec, artifact_tables),
        target / "model_leaderboard.csv",
    )
    calibration_diagnostics = write_table(
        _calibration_diagnostics(spec, artifact_tables),
        target / "calibration_diagnostics.csv",
    )
    operating_points = write_table(
        _operating_points(spec, artifact_tables),
        target / "operating_points.csv",
    )
    decision_translation = write_table(
        _decision_translation(spec, artifact_tables),
        target / "decision_translation.csv",
    )
    cost_turnover_sensitivity = write_table(
        _cost_turnover_sensitivity(spec, artifact_tables),
        target / "cost_turnover_sensitivity.csv",
    )
    artifact_manifest = write_table(
        _artifact_manifest(spec, artifact_paths),
        target / "artifact_manifest.csv",
    )
    figure_manifest = write_table(_figure_manifest(spec), target / "figure_manifest.csv")

    limitations_markdown = target / "limitations.md"
    limitations_markdown.write_text(_limitations_markdown(spec), encoding="utf-8")

    publication_checklist = target / "publication_checklist.md"
    publication_checklist.write_text(_publication_checklist(spec), encoding="utf-8")

    manifest = _write_bundle_manifest(spec, target)

    return ExperimentBundle(
        out_dir=target,
        manifest=manifest,
        experiment_json=experiment_json,
        contracts_markdown=contracts_markdown,
        data_summary=data_summary,
        split_summary=split_summary,
        model_leaderboard=model_leaderboard,
        calibration_diagnostics=calibration_diagnostics,
        operating_points=operating_points,
        decision_translation=decision_translation,
        cost_turnover_sensitivity=cost_turnover_sensitivity,
        artifact_manifest=artifact_manifest,
        figure_manifest=figure_manifest,
        limitations_markdown=limitations_markdown,
        publication_checklist=publication_checklist,
    )


def _data_summary(spec: ExperimentSpec) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": spec.id,
                "source": spec.data.source,
                "universe": spec.data.universe or ",".join(spec.data.symbols),
                "symbol_count": len(spec.data.symbols),
                "fields": ",".join(spec.data.fields),
                "start": spec.data.start,
                "end": spec.data.end,
                "adjusted_policy": spec.data.adjusted_policy,
                "timestamp_convention": spec.data.timestamp_convention,
                "missing_data_policy": spec.data.missing_data_policy,
            }
        ]
    )


def _split_summary(spec: ExperimentSpec) -> pd.DataFrame:
    return pd.DataFrame([window.model_dump() for window in spec.validation.windows])


def _load_artifact_tables(artifact_paths: list[str | Path]) -> list[tuple[str, pd.DataFrame]]:
    tables: list[tuple[str, pd.DataFrame]] = []
    for item in artifact_paths:
        path = Path(item)
        if not path.is_file():
            continue
        try:
            frame = _read_artifact_table(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not frame.empty:
            tables.append((str(item), frame))
    return tables


def _read_artifact_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            if isinstance(payload.get("validation_metrics"), list):
                return pd.DataFrame(payload["validation_metrics"])
            return pd.json_normalize(payload)
    raise ValueError(f"unsupported artifact table format: {path}")


def _model_leaderboard(
    spec: ExperimentSpec,
    artifact_tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    leaderboard_metrics = set(spec.metrics) - {
        "turnover",
        "cost_adjusted_return",
        "gross_exposure",
        "net_exposure",
    }
    metric_columns = leaderboard_metrics | {
        "accuracy",
        "alpha",
        "average_precision",
        "brier_score",
        "cagr",
        "calmar",
        "information_ratio",
        "log_loss",
        "mean_cross_sectional_rank_ic",
        "precision_at_k",
        "rank_ic",
        "rmse",
        "roc_auc",
        "sharpe",
        "top_k_excess_return",
        "top_quantile_hit_rate",
        "tracking_error",
    }
    for source, frame in artifact_tables:
        observed_metrics = [column for column in frame.columns if column in metric_columns]
        if not observed_metrics:
            continue
        model_name = Path(source).stem
        for _, row in frame.iterrows():
            payload = {
                "experiment_id": spec.id,
                "model_type": "observed",
                "model": str(row.get("model", row.get("name", model_name))),
                "selection_policy": spec.models.selection_policy,
                "status": "observed_results",
                "source": source,
            }
            for column in observed_metrics:
                payload[column] = row[column]
            rows.append(payload)
    if not rows:
        return _model_leaderboard_template(spec)
    return pd.DataFrame(rows).sort_values(["status", "model"]).reset_index(drop=True)


def _model_leaderboard_template(spec: ExperimentSpec) -> pd.DataFrame:
    rows = []
    for model_type, names in [
        ("baseline", spec.models.baselines),
        ("candidate", spec.models.candidates),
    ]:
        for name in names:
            rows.append(
                {
                    "experiment_id": spec.id,
                    "model_type": model_type,
                    "model": name,
                    "selection_policy": spec.models.selection_policy,
                    "status": "pending_results",
                }
            )
    return pd.DataFrame(
        rows,
        columns=["experiment_id", "model_type", "model", "selection_policy", "status"],
    )


def _calibration_diagnostics(
    spec: ExperimentSpec,
    artifact_tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    calibration_columns = ["brier_score", "log_loss", "ece"]
    for source, frame in artifact_tables:
        if not any(column in frame.columns for column in calibration_columns):
            continue
        model_name = Path(source).stem
        for _, row in frame.iterrows():
            rows.append(
                {
                    "experiment_id": spec.id,
                    "model": str(row.get("model", row.get("name", model_name))),
                    "window": str(row.get("window", row.get("fold_start", "observed"))),
                    "brier_score": row.get("brier_score"),
                    "log_loss": row.get("log_loss"),
                    "ece": row.get("ece"),
                    "calibration_method": row.get("calibration_method", ""),
                    "source": source,
                    "status": "observed_results",
                }
            )
    if not rows:
        return _calibration_diagnostics_template(spec)
    return pd.DataFrame(rows)


def _calibration_diagnostics_template(spec: ExperimentSpec) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": spec.id,
                "model": "",
                "window": "calibration",
                "brier_score": None,
                "log_loss": None,
                "ece": None,
                "calibration_method": "",
                "status": "pending_results",
            }
        ]
    )


def _operating_points(
    spec: ExperimentSpec,
    artifact_tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for source, frame in artifact_tables:
        if {"target_recall", "threshold", "precision"}.issubset(frame.columns):
            for _, row in frame.iterrows():
                rows.append(
                    {
                        "experiment_id": spec.id,
                        "model": str(row.get("score", row.get("model", Path(source).stem))),
                        "window": str(row.get("window", "observed")),
                        "target_recall": row.get("target_recall"),
                        "threshold": row.get("threshold"),
                        "alerts": row.get("alerts"),
                        "precision": row.get("precision"),
                        "source": source,
                        "status": "observed_results",
                    }
                )
        elif {"date", "symbol", "weight"}.issubset(frame.columns):
            latest = frame.copy()
            latest["date"] = pd.to_datetime(latest["date"])
            latest = latest.loc[latest["date"] == latest["date"].max()]
            rows.append(
                {
                    "experiment_id": spec.id,
                    "model": Path(source).stem,
                    "window": str(latest["date"].max().date()) if not latest.empty else "",
                    "top_k": int((latest["weight"] > 0).sum()),
                    "weight_rule": spec.decision.allocation_rule,
                    "gross_exposure": float(latest["weight"].abs().sum()),
                    "net_exposure": float(latest["weight"].sum()),
                    "source": source,
                    "status": "observed_results",
                }
            )
    if not rows:
        return _operating_points_template(spec)
    return pd.DataFrame(rows)


def _operating_points_template(spec: ExperimentSpec) -> pd.DataFrame:
    if spec.decision.workflow == "alert_queue":
        columns = [
            "experiment_id",
            "model",
            "window",
            "target_recall",
            "threshold",
            "alerts",
            "precision",
            "status",
        ]
    elif spec.decision.workflow in {"allocation", "ranking"}:
        columns = [
            "experiment_id",
            "model",
            "window",
            "top_k",
            "weight_rule",
            "gross_exposure",
            "net_exposure",
            "status",
        ]
    else:
        columns = ["experiment_id", "model", "window", "operating_point", "value", "status"]
    return pd.DataFrame([{column: None for column in columns}]).assign(
        experiment_id=spec.id,
        status="pending_results",
    )


def _decision_translation(
    spec: ExperimentSpec,
    artifact_tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    row = _decision_translation_template(spec).iloc[0].to_dict()
    weight_rows = 0
    turnover_rows = 0
    turnover_values = []
    total_cost = 0.0
    for _, frame in artifact_tables:
        if {"date", "symbol", "weight"}.issubset(frame.columns):
            weight_rows += int(len(frame))
        if {"turnover", "cost"}.issubset(frame.columns):
            turnover_rows += int(len(frame))
            turnover_values.extend(pd.to_numeric(frame["turnover"], errors="coerce").dropna())
            total_cost += float(pd.to_numeric(frame["cost"], errors="coerce").fillna(0.0).sum())
    row.update(
        {
            "observed_weight_rows": weight_rows,
            "observed_turnover_rows": turnover_rows,
            "observed_average_turnover": float(pd.Series(turnover_values).mean())
            if turnover_values
            else None,
            "observed_total_cost": total_cost if turnover_rows else None,
            "status": "observed_artifacts" if weight_rows or turnover_rows else "contract_only",
        }
    )
    return pd.DataFrame([row])


def _decision_translation_template(spec: ExperimentSpec) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": spec.id,
                "workflow": spec.decision.workflow,
                "allocation_rule": spec.decision.allocation_rule,
                "alert_threshold_policy": spec.decision.alert_threshold_policy,
                "cost_model_bps": spec.decision.cost_model_bps,
                "slippage_bps": spec.decision.slippage_bps,
                "turnover_limit": spec.decision.turnover_limit,
                "risk_controls": ",".join(spec.decision.risk_controls),
                "status": "contract_only",
            }
        ]
    )


def _cost_turnover_sensitivity(
    spec: ExperimentSpec,
    artifact_tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for source, frame in artifact_tables:
        if not {"turnover", "cost"}.issubset(frame.columns):
            continue
        turnover = pd.to_numeric(frame["turnover"], errors="coerce")
        cost = pd.to_numeric(frame["cost"], errors="coerce")
        scenario = Path(source).stem
        if "scenario" in frame.columns and frame["scenario"].nunique(dropna=True) == 1:
            scenario = str(frame["scenario"].dropna().iloc[0])
        rows.append(
            {
                "experiment_id": spec.id,
                "scenario": scenario,
                "cost_bps": spec.decision.cost_model_bps,
                "slippage_bps": spec.decision.slippage_bps,
                "turnover": float(turnover.mean()),
                "cost_adjusted_return": None,
                "total_cost": float(cost.sum()),
                "source": source,
                "status": "observed_results",
            }
        )
    if not rows:
        return _cost_turnover_sensitivity_template(spec)
    return pd.DataFrame(rows)


def _cost_turnover_sensitivity_template(spec: ExperimentSpec) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": spec.id,
                "scenario": "base_contract",
                "cost_bps": spec.decision.cost_model_bps,
                "slippage_bps": spec.decision.slippage_bps,
                "turnover": None,
                "cost_adjusted_return": None,
                "status": "pending_results",
            }
        ]
    )


def _artifact_manifest(spec: ExperimentSpec, artifact_paths: list[str | Path]) -> pd.DataFrame:
    rows = []
    for item in [*spec.artifacts, *[str(path) for path in artifact_paths]]:
        path = Path(item)
        rows.append(
            {
                "path": item,
                "name": path.name,
                "suffix": path.suffix,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            }
        )
    return pd.DataFrame(rows, columns=["path", "name", "suffix", "exists", "size_bytes"])


def _figure_manifest(spec: ExperimentSpec) -> pd.DataFrame:
    figure_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    rows = []
    for item in spec.artifacts:
        path = Path(item)
        if path.suffix.lower() in figure_suffixes:
            rows.append(
                {
                    "experiment_id": spec.id,
                    "figure": item,
                    "publication_ready": False,
                    "caption": "",
                    "status": "pending_review",
                }
            )
    return pd.DataFrame(
        rows,
        columns=["experiment_id", "figure", "publication_ready", "caption", "status"],
    )


def _write_bundle_manifest(spec: ExperimentSpec, target: Path) -> Path:
    manifest = target / "manifest.json"
    files = sorted({path.name for path in target.iterdir() if path.is_file()} | {"manifest.json"})
    payload = {
        "bundle_version": 1,
        "experiment_id": spec.id,
        "experiment_name": spec.name,
        "files": files,
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _limitations_markdown(spec: ExperimentSpec) -> str:
    limitations = "\n".join(f"- {item}" for item in spec.limitations) or "- None recorded"
    return f"""# Limitations: {spec.name}

{limitations}

## Publication Rule

Do not present this experiment as a trading edge unless the decision layer, costs, turnover,
data timing, and relevant production caveats have been tested and reported.
"""


def _contracts_markdown(spec: ExperimentSpec) -> str:
    lookback_windows = (
        ", ".join(str(item) for item in spec.features.lookback_windows_days) or "not specified"
    )
    windows = "\n".join(
        f"- {item.name}: {item.start} to {item.end} ({item.purpose})"
        for item in spec.validation.windows
    )
    limitations = "\n".join(f"- {item}" for item in spec.limitations) or "- None recorded"
    return f"""# {spec.name}

## Hypothesis

{spec.hypothesis}

## Data Contract

- Source: {spec.data.source}
- Universe: {spec.data.universe or "custom symbol list"}
- Symbols: {", ".join(spec.data.symbols)}
- Fields: {", ".join(spec.data.fields)}
- Adjusted policy: {spec.data.adjusted_policy}
- Timestamp convention: {spec.data.timestamp_convention}
- Missing-data policy: {spec.data.missing_data_policy}

## Feature Contract

- Families: {", ".join(spec.features.families)}
- Lookback windows: {lookback_windows}
- Static identity policy: {spec.features.static_identity_policy}
- Leakage policy: {spec.features.leakage_policy}

## Target Contract

- Name: {spec.target.name}
- Task: {spec.target.task}
- Horizon days: {spec.target.horizon_days}
- Label definition: {spec.target.label_definition}
- Execution convention: {spec.target.execution_convention}
- Overlapping labels allowed: {spec.target.allow_overlapping_labels}

## Validation Contract

- Split policy: {spec.validation.split_policy}
- Embargo days: {spec.validation.embargo_days}
- Overlap policy: {spec.validation.overlap_policy}
{windows}

## Decision Contract

- Workflow: {spec.decision.workflow}
- Allocation rule: {spec.decision.allocation_rule or "n/a"}
- Alert threshold policy: {spec.decision.alert_threshold_policy or "n/a"}
- Cost model bps: {spec.decision.cost_model_bps}
- Slippage bps: {spec.decision.slippage_bps}
- Risk controls: {", ".join(spec.decision.risk_controls) or "not specified"}

## Models and Metrics

- Baselines: {", ".join(spec.models.baselines) or "none"}
- Candidates: {", ".join(spec.models.candidates) or "none"}
- Metrics: {", ".join(spec.metrics)}

## Known Limitations

{limitations}

## SAM Takeaway

{spec.sam_takeaway or "Not recorded"}
"""


def _publication_checklist(spec: ExperimentSpec) -> str:
    return f"""# Publication Checklist: {spec.name}

- [ ] Problem is framed as a professional workflow, not a generic prediction demo.
- [ ] Data, feature, target, validation, and decision contracts are stated.
- [ ] Simple domain baselines are reported before complex models.
- [ ] Calibration, operating thresholds, or ranking diagnostics are included where relevant.
- [ ] Decision utility is translated into alert queue, allocation, risk, cost, or turnover terms.
- [ ] Failure modes and limitations are explicit.
- [ ] Blog post links back to SAM experiment id `{spec.id}`.
- [ ] SAM takeaway explains what reusable platform capability this experiment added.
"""


__all__ = [
    "DataContract",
    "DecisionContract",
    "ExperimentBundle",
    "ExperimentRegistry",
    "ExperimentSpec",
    "FeatureContract",
    "ModelContract",
    "TargetContract",
    "ValidationContract",
    "ValidationWindow",
    "load_experiment",
    "validate_experiment_specs",
    "validate_research_frame",
    "write_experiment_bundle",
]
