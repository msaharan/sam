"""Score-driven allocation utilities for tactical ETF and ranking workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocationDiagnostics:
    weights: pd.DataFrame
    turnover: pd.DataFrame


def scores_to_weights(
    scores: pd.DataFrame,
    *,
    date_column: str = "date",
    symbol_column: str = "symbol",
    score_column: str = "prediction",
    top_k: int | None = None,
    top_quantile: float | None = None,
    max_weight: float = 1.0,
    weighting: Literal["equal", "rank"] = "equal",
) -> pd.DataFrame:
    """Convert cross-sectional scores into long-only target weights by date."""

    _validate_score_inputs(scores, date_column, symbol_column, score_column)
    if (top_k is None) == (top_quantile is None):
        raise ValueError("provide exactly one of top_k or top_quantile")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")
    if top_quantile is not None and not 0 < top_quantile <= 1:
        raise ValueError("top_quantile must be in the interval (0, 1]")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in the interval (0, 1]")

    frame = scores[[date_column, symbol_column, score_column]].dropna().copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    rows = []
    for date, group in frame.groupby(date_column, sort=True):
        selected = _select_top_group(
            group.sort_values([score_column, symbol_column], ascending=[False, True]),
            top_k=top_k,
            top_quantile=top_quantile,
        )
        raw = _raw_weights(selected, symbol_column, score_column, weighting)
        weights = _cap_and_normalize(raw, max_weight)
        for symbol, weight in weights.items():
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "weight": float(weight),
                    "score": float(selected.set_index(symbol_column).loc[symbol, score_column]),
                }
            )
    return pd.DataFrame(rows, columns=["date", "symbol", "weight", "score"])


def allocation_turnover(
    weights: pd.DataFrame,
    *,
    date_column: str = "date",
    symbol_column: str = "symbol",
    weight_column: str = "weight",
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    turnover_limit: float | None = None,
) -> pd.DataFrame:
    """Compute turnover and cost diagnostics for target-weight panels."""

    required = {date_column, symbol_column, weight_column}
    missing = sorted(required - set(weights.columns))
    if missing:
        raise ValueError(f"weights missing columns: {missing}")
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("transaction costs and slippage must be non-negative")
    if turnover_limit is not None and turnover_limit < 0:
        raise ValueError("turnover_limit must be non-negative")

    frame = weights[[date_column, symbol_column, weight_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    panel = (
        frame.pivot_table(
            index=date_column,
            columns=symbol_column,
            values=weight_column,
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )
    previous = panel.shift(1).fillna(0.0)
    turnover = (panel - previous).abs().sum(axis=1)
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
    diagnostics = pd.DataFrame(
        {
            "date": panel.index,
            "turnover": turnover.to_numpy(dtype=float),
            "cost": (turnover * cost_rate).to_numpy(dtype=float),
            "gross_exposure": panel.abs().sum(axis=1).to_numpy(dtype=float),
            "net_exposure": panel.sum(axis=1).to_numpy(dtype=float),
        }
    )
    if turnover_limit is not None:
        diagnostics["turnover_limit"] = float(turnover_limit)
        diagnostics["turnover_breach"] = diagnostics["turnover"] > float(turnover_limit)
    return diagnostics.reset_index(drop=True)


def build_score_allocation(
    scores: pd.DataFrame,
    *,
    top_k: int | None = None,
    top_quantile: float | None = None,
    max_weight: float = 1.0,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    turnover_limit: float | None = None,
    score_column: str = "prediction",
    weighting: Literal["equal", "rank"] = "equal",
) -> AllocationDiagnostics:
    weights = scores_to_weights(
        scores,
        score_column=score_column,
        top_k=top_k,
        top_quantile=top_quantile,
        max_weight=max_weight,
        weighting=weighting,
    )
    turnover = allocation_turnover(
        weights,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        turnover_limit=turnover_limit,
    )
    return AllocationDiagnostics(weights=weights, turnover=turnover)


def _validate_score_inputs(
    scores: pd.DataFrame,
    date_column: str,
    symbol_column: str,
    score_column: str,
) -> None:
    required = {date_column, symbol_column, score_column}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"scores missing columns: {missing}")


def _select_top_group(
    group: pd.DataFrame,
    *,
    top_k: int | None,
    top_quantile: float | None,
) -> pd.DataFrame:
    if top_k is not None:
        count = min(top_k, len(group))
    else:
        count = int(np.ceil(len(group) * float(top_quantile)))
    return group.head(max(count, 1))


def _raw_weights(
    selected: pd.DataFrame,
    symbol_column: str,
    score_column: str,
    weighting: Literal["equal", "rank"],
) -> pd.Series:
    if weighting == "equal":
        return pd.Series(1.0, index=selected[symbol_column].to_numpy())
    if weighting == "rank":
        ranks = selected[score_column].rank(method="first", ascending=True)
        return pd.Series(ranks.to_numpy(dtype=float), index=selected[symbol_column].to_numpy())
    raise ValueError(f"unknown weighting: {weighting}")


def _cap_and_normalize(weights: pd.Series, max_weight: float) -> pd.Series:
    weights = weights.astype(float).clip(lower=0)
    if weights.empty or weights.sum() <= 0:
        raise ValueError("cannot normalize empty or non-positive weights")
    if len(weights) * max_weight < 1 - 1e-12:
        raise ValueError("max_weight is too low for the selected asset count")
    weights = weights / weights.sum()
    for _ in range(100):
        over = weights > max_weight
        if not over.any():
            break
        capped = weights[over].copy()
        weights[over] = max_weight
        residual = 1.0 - float(weights[over].sum())
        under = ~over
        if residual < -1e-12 or not under.any():
            break
        under_sum = float(weights[under].sum())
        if under_sum <= 0:
            weights[under] = residual / int(under.sum())
        else:
            weights[under] = weights[under] / under_sum * residual
        if capped.equals(weights[over]):
            break
    return weights / weights.sum()


__all__ = [
    "AllocationDiagnostics",
    "allocation_turnover",
    "build_score_allocation",
    "scores_to_weights",
]
