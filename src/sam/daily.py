"""Daily research brief workflow for SAM."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from sam.allocation import scores_to_weights
from sam.config import Market, load_ml_config, load_toml, load_universe
from sam.data import fetch_prices, validate_prices
from sam.io import ensure_dir, read_table, write_table
from sam.ml import predict, train_model
from sam.risk import build_volatility_regime_frame


class DailyBriefConfig(BaseModel):
    """Configuration for the daily ETF/risk research brief."""

    universe: Path = Path("configs/universes/etf_core.toml")
    market: Market = "us"
    data_source: Literal["yfinance"] = "yfinance"
    start: str = "2018-01-01"
    price_output: Path = Path("data/raw/etf_core.parquet")
    refresh_data: bool = True
    as_of: str | None = None

    ml_config: Path = Path("configs/ml/baseline.toml")
    model_dir: Path = Path("models/latest")
    model_path: Path = Path("models/latest/model.pkl")
    train_if_missing: bool = True
    retrain: bool = False

    risk_symbol: str = "SPY"
    risk_horizon_days: int = Field(default=20, ge=2)
    risk_threshold_quantile: float = Field(default=0.8, gt=0, lt=1)
    risk_threshold_end: str | None = "2017-12-29"

    allocation_top_k: int = Field(default=5, ge=1)
    allocation_max_weight: float = Field(default=0.20, gt=0, le=1)
    allocation_weighting: Literal["equal", "rank"] = "equal"
    transaction_cost_bps: float = Field(default=5.0, ge=0)
    slippage_bps: float = Field(default=1.0, ge=0)
    turnover_limit: float = Field(default=2.0, ge=0)
    top_scores_count: int = Field(default=10, ge=1)

    report_root: Path = Path("reports/daily")
    output_dir: Path | None = None

    @model_validator(mode="after")
    def require_feasible_weight_cap(self) -> DailyBriefConfig:
        if self.allocation_top_k * self.allocation_max_weight < 1 - 1e-12:
            raise ValueError("allocation_top_k * allocation_max_weight must be at least 1")
        return self


@dataclass(frozen=True)
class DailyBriefResult:
    out_dir: Path
    latest_dir: Path
    brief_markdown: Path
    summary_json: Path
    paths: dict[str, Path]
    summary: dict[str, Any]


def load_daily_brief_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> DailyBriefConfig:
    payload = load_toml(path)
    payload.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return DailyBriefConfig.model_validate(payload)


def run_daily_brief(config: DailyBriefConfig) -> DailyBriefResult:
    cfg = DailyBriefConfig.model_validate(config)
    run_date = _run_date(cfg)
    out_dir = ensure_dir(cfg.output_dir or cfg.report_root / run_date.isoformat())
    latest_dir = cfg.report_root / "latest"
    previous_weights = _read_previous_latest_weights(latest_dir)

    prices = _load_or_fetch_prices(cfg)
    validation = validate_prices(prices)
    model_path, model_status = _ensure_model(prices, cfg)
    predictions = predict(prices, model_path)
    allocation_scores = _latest_allocation_scores(predictions, cfg)
    target_weights = _target_weights(allocation_scores, cfg)
    turnover = _turnover_against_previous(target_weights, previous_weights, cfg)
    risk_snapshot = _risk_snapshot(prices, cfg)

    paths = {
        "data_validation": write_table(validation, out_dir / "data_validation.csv"),
        "risk_snapshot": write_table(risk_snapshot, out_dir / "risk_snapshot.csv"),
        "allocation_scores": write_table(allocation_scores, out_dir / "allocation_scores.csv"),
        "target_weights": write_table(target_weights, out_dir / "target_weights.csv"),
        "turnover": write_table(turnover, out_dir / "turnover.csv"),
    }

    summary = _build_summary(
        cfg,
        run_date=run_date,
        prices=prices,
        validation=validation,
        model_path=model_path,
        model_status=model_status,
        risk_snapshot=risk_snapshot,
        target_weights=target_weights,
        turnover=turnover,
        out_dir=out_dir,
        latest_dir=latest_dir,
        paths=paths,
    )
    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps(summary, default=str, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    brief_markdown = out_dir / "brief.md"
    brief_markdown.write_text(
        _render_brief(
            summary,
            risk_snapshot,
            allocation_scores,
            target_weights,
            turnover,
            top_scores_count=cfg.top_scores_count,
        ),
        encoding="utf-8",
    )
    paths["summary"] = summary_json
    paths["brief"] = brief_markdown
    paths["manifest"] = _write_daily_manifest(out_dir, cfg, summary)

    _mirror_latest(out_dir, latest_dir)
    return DailyBriefResult(
        out_dir=out_dir,
        latest_dir=latest_dir,
        brief_markdown=brief_markdown,
        summary_json=summary_json,
        paths=paths,
        summary=summary,
    )


def _run_date(config: DailyBriefConfig) -> date:
    if config.as_of:
        return pd.Timestamp(config.as_of).date()
    return datetime.now().date()


def _load_or_fetch_prices(config: DailyBriefConfig) -> pd.DataFrame:
    as_of = pd.Timestamp(config.as_of) if config.as_of else None
    if config.refresh_data:
        universe = load_universe(config.universe)
        end = None
        if as_of is not None:
            end = (as_of + pd.Timedelta(days=1)).date().isoformat()
        prices = fetch_prices(
            universe.normalized_symbols,
            start=config.start,
            end=end,
            market=config.market,
            out=config.price_output,
        )
    else:
        prices = read_table(config.price_output)
    if as_of is not None:
        frame = prices.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        prices = frame.loc[frame["date"] <= as_of].reset_index(drop=True)
    if prices.empty:
        raise ValueError("daily brief has no price rows after applying configuration")
    return prices


def _ensure_model(prices: pd.DataFrame, config: DailyBriefConfig) -> tuple[Path, str]:
    model_path = config.model_path
    if config.retrain:
        return _train_daily_model(prices, config), "retrained"
    if model_path.is_file():
        return model_path, "reused"
    if config.train_if_missing:
        return _train_daily_model(prices, config), "trained_missing"
    raise FileNotFoundError(model_path)


def _train_daily_model(prices: pd.DataFrame, config: DailyBriefConfig) -> Path:
    result = train_model(prices, config=load_ml_config(config.ml_config), out_dir=config.model_dir)
    if result.model_path is None:
        raise RuntimeError("daily brief could not train a model; no walk-forward folds were found")
    return result.model_path


def _latest_allocation_scores(
    predictions: pd.DataFrame,
    config: DailyBriefConfig,
) -> pd.DataFrame:
    if predictions.empty:
        raise ValueError("daily brief model produced no predictions")
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    latest_date = frame["date"].max()
    latest = frame.loc[frame["date"] == latest_date, ["date", "symbol", "prediction"]].copy()
    latest = latest.sort_values(["prediction", "symbol"], ascending=[False, True]).reset_index(
        drop=True
    )
    latest["score_rank"] = range(1, len(latest) + 1)
    latest["selected"] = latest["score_rank"] <= config.allocation_top_k
    return latest[["date", "symbol", "prediction", "score_rank", "selected"]]


def _target_weights(scores: pd.DataFrame, config: DailyBriefConfig) -> pd.DataFrame:
    weights = scores_to_weights(
        scores,
        top_k=config.allocation_top_k,
        max_weight=config.allocation_max_weight,
        weighting=config.allocation_weighting,
    )
    enriched = weights.merge(
        scores[["date", "symbol", "score_rank"]],
        on=["date", "symbol"],
        how="left",
    )
    return enriched.sort_values(["weight", "score"], ascending=[False, False]).reset_index(
        drop=True
    )


def _read_previous_latest_weights(latest_dir: Path) -> pd.DataFrame | None:
    path = latest_dir / "target_weights.csv"
    if not path.is_file():
        return None
    return read_table(path)


def _turnover_against_previous(
    current_weights: pd.DataFrame,
    previous_weights: pd.DataFrame | None,
    config: DailyBriefConfig,
) -> pd.DataFrame:
    current = current_weights.groupby("symbol")["weight"].sum().astype(float)
    previous_date = None
    if previous_weights is None or previous_weights.empty:
        previous = pd.Series(dtype=float)
    else:
        previous_frame = previous_weights.copy()
        previous_frame["date"] = pd.to_datetime(previous_frame["date"])
        previous_date = previous_frame["date"].max()
        previous = (
            previous_frame.loc[previous_frame["date"] == previous_date]
            .groupby("symbol")["weight"]
            .sum()
            .astype(float)
        )
    symbols = sorted(set(current.index) | set(previous.index))
    current = current.reindex(symbols, fill_value=0.0)
    previous = previous.reindex(symbols, fill_value=0.0)
    turnover = float((current - previous).abs().sum())
    cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000
    return pd.DataFrame(
        [
            {
                "date": pd.to_datetime(current_weights["date"]).max(),
                "previous_weight_date": previous_date,
                "turnover": turnover,
                "cost": turnover * cost_rate,
                "gross_exposure": float(current.abs().sum()),
                "net_exposure": float(current.sum()),
                "turnover_limit": config.turnover_limit,
                "turnover_breach": turnover > config.turnover_limit,
            }
        ]
    )


def _risk_snapshot(prices: pd.DataFrame, config: DailyBriefConfig) -> pd.DataFrame:
    result = build_volatility_regime_frame(
        prices,
        symbol=config.risk_symbol,
        horizon_days=config.risk_horizon_days,
        threshold_quantile=config.risk_threshold_quantile,
        threshold_end=config.risk_threshold_end,
        include_unlabeled=True,
    )
    vol_column = f"realized_vol_{config.risk_horizon_days}d"
    usable = result.frame.dropna(subset=[vol_column]).copy()
    if usable.empty:
        raise ValueError("daily brief has no labelled or current volatility rows")
    latest = usable.tail(1).copy()
    threshold = float(latest["target_threshold"].iloc[0])
    current_vol = float(latest[vol_column].iloc[0])
    ratio = current_vol / threshold if threshold > 0 else float("nan")
    latest["risk_score_to_threshold"] = ratio
    latest["risk_level"] = _risk_level(ratio)
    vix_metrics = _vix_snapshot_metrics(prices)
    for column, value in vix_metrics.items():
        latest[column] = value
    columns = [
        "date",
        "symbol",
        "risk_level",
        vol_column,
        "target_threshold",
        "risk_score_to_threshold",
        "vix_close",
        "vix_zscore_252d",
        "drawdown_20d",
        "drawdown_63d",
        "ma_distance_20d",
        "ma_distance_63d",
        f"future_realized_vol_{config.risk_horizon_days}d",
        "target_high_vol",
        "feature_available_date",
        "target_date",
    ]
    for column in columns:
        if column not in latest.columns:
            latest[column] = pd.NA
    return latest[columns].reset_index(drop=True)


def _vix_snapshot_metrics(prices: pd.DataFrame) -> dict[str, float]:
    """Attach simple VIX rule inputs when a VIX proxy exists in the price table."""

    if "symbol" not in prices.columns:
        return {}
    vix_symbol = next(
        (item for item in prices["symbol"].unique() if str(item).upper() in {"^VIX", "VIX"}),
        None,
    )
    if vix_symbol is None:
        return {}
    frame = prices[prices["symbol"] == vix_symbol].sort_values("date")
    if frame.empty:
        return {}
    close = frame["adj_close"].astype(float)
    latest_close = float(close.iloc[-1])
    mean = float(close.rolling(252, min_periods=60).mean().iloc[-1])
    std = float(close.rolling(252, min_periods=60).std(ddof=1).iloc[-1])
    zscore = (latest_close - mean) / std if std > 0 else float("nan")
    return {"vix_close": latest_close, "vix_zscore_252d": zscore}


def _risk_level(ratio: float) -> str:
    if pd.isna(ratio):
        return "unknown"
    if ratio >= 1.0:
        return "elevated"
    if ratio >= 0.75:
        return "watch"
    return "normal"


def _build_summary(
    config: DailyBriefConfig,
    *,
    run_date: date,
    prices: pd.DataFrame,
    validation: pd.DataFrame,
    model_path: Path,
    model_status: str,
    risk_snapshot: pd.DataFrame,
    target_weights: pd.DataFrame,
    turnover: pd.DataFrame,
    out_dir: Path,
    latest_dir: Path,
    paths: dict[str, Path],
) -> dict[str, Any]:
    price_last_date = pd.to_datetime(prices["date"]).max().date()
    selected = (
        target_weights.sort_values(["weight", "score"], ascending=[False, False])["symbol"]
        .astype(str)
        .tolist()
    )
    output_paths = {key: str(value) for key, value in paths.items()}
    output_paths.update(
        {
            "out_dir": str(out_dir),
            "latest_dir": str(latest_dir),
            "summary": str(out_dir / "summary.json"),
            "brief": str(out_dir / "brief.md"),
            "manifest": str(out_dir / "manifest.json"),
        }
    )
    row = turnover.iloc[0]
    return {
        "run_date": run_date.isoformat(),
        "price_last_date": price_last_date.isoformat(),
        "data_freshness_days": int((run_date - price_last_date).days),
        "validation_failures": int((validation["status"] == "fail").sum()),
        "model_status": model_status,
        "model_path": str(model_path),
        "risk_level": str(risk_snapshot["risk_level"].iloc[0]),
        "top_selected_symbols": selected,
        "gross_exposure": float(row["gross_exposure"]),
        "net_exposure": float(row["net_exposure"]),
        "turnover": float(row["turnover"]),
        "estimated_cost": float(row["cost"]),
        "output_paths": output_paths,
        "research_only": True,
    }


def _render_brief(
    summary: dict[str, Any],
    risk_snapshot: pd.DataFrame,
    allocation_scores: pd.DataFrame,
    target_weights: pd.DataFrame,
    turnover: pd.DataFrame,
    *,
    top_scores_count: int,
) -> str:
    risk = risk_snapshot.iloc[0]
    turn = turnover.iloc[0]
    top_scores = allocation_scores.head(top_scores_count).copy()
    return f"""# Daily SAM Brief - {summary["run_date"]}

Research only. Not investment advice.

## Data Freshness and Validation

- Price last date: {summary["price_last_date"]}
- Data freshness days: {summary["data_freshness_days"]}
- Validation failures: {summary["validation_failures"]}
- Model status: {summary["model_status"]}

## SPY Risk Regime

- Risk level: {summary["risk_level"]}
- Realized volatility 20d: {_format_value(risk.get("realized_vol_20d"))}
- Threshold: {_format_value(risk.get("target_threshold"))}
- Volatility / threshold: {_format_value(risk.get("risk_score_to_threshold"))}
- Drawdown 20d: {_format_value(risk.get("drawdown_20d"))}
- MA distance 20d: {_format_value(risk.get("ma_distance_20d"))}

## ETF Ranking

{_markdown_table(top_scores[["score_rank", "symbol", "prediction", "selected"]])}

## Target Research Weights

{_markdown_table(target_weights[["symbol", "weight", "score", "score_rank"]])}

## Turnover and Cost Diagnostics

- Previous weight date: {turn.get("previous_weight_date")}
- Turnover: {_format_value(turn.get("turnover"))}
- Estimated cost: {_format_value(turn.get("cost"))}
- Gross exposure: {_format_value(turn.get("gross_exposure"))}
- Net exposure: {_format_value(turn.get("net_exposure"))}
- Turnover breach: {bool(turn.get("turnover_breach"))}

## Limitations

- Public market data can be revised and may differ from institutional data.
- Costs are simple basis-point estimates, not a market-impact model.
- The allocation view is a daily research snapshot for a monthly-horizon ETF workflow.
- TabPFN and TabICL are not required for this CPU-first daily brief.
"""


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for column in text.columns:
        text[column] = text[column].map(_format_value)
    header = "| " + " | ".join(text.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows])


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(value)
    if isinstance(value, Real):
        return f"{float(value):.4f}"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value)


def _write_daily_manifest(
    out_dir: Path,
    config: DailyBriefConfig,
    summary: dict[str, Any],
) -> Path:
    manifest = out_dir / "manifest.json"
    files = sorted({path.name for path in out_dir.iterdir() if path.is_file()} | {"manifest.json"})
    payload = {
        "command": "daily brief",
        "created_at": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "summary": summary,
        "files": files,
    }
    manifest.write_text(
        json.dumps(payload, default=str, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _mirror_latest(out_dir: Path, latest_dir: Path) -> None:
    if out_dir.resolve() == latest_dir.resolve():
        return
    ensure_dir(latest_dir.parent)
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(out_dir, latest_dir)


__all__ = [
    "DailyBriefConfig",
    "DailyBriefResult",
    "load_daily_brief_config",
    "run_daily_brief",
]
