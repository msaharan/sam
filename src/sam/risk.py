"""Risk-scoring primitives for volatility regime research."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class VolatilityRegimeResult:
    frame: pd.DataFrame
    threshold: float
    threshold_quantile: float
    threshold_end: str | None


def build_volatility_regime_frame(
    prices: pd.DataFrame,
    *,
    symbol: str,
    horizon_days: int = 20,
    threshold_quantile: float = 0.8,
    threshold_end: str | None = None,
    windows: Iterable[int] = (5, 20, 63, 126, 252),
    include_unlabeled: bool = False,
) -> VolatilityRegimeResult:
    """Build a leakage-aware volatility-regime research table for one asset.

    Features are computed through the signal date. The target uses realized volatility from
    subsequent returns only, so it can be used for time-ordered risk-scoring experiments.
    """

    if horizon_days < 2:
        raise ValueError("horizon_days must be at least 2 for realized-volatility targets")
    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be between 0 and 1")

    asset = _asset_frame(prices, symbol)
    close = asset["adj_close"].astype(float)
    raw_close = asset["close"].astype(float) if "close" in asset else close
    returns = close.pct_change(fill_method=None)
    frame = pd.DataFrame({"date": asset["date"], "symbol": symbol})
    for window in windows:
        if window < 2:
            raise ValueError("volatility windows must be at least 2 days")
        frame[f"return_{window}d"] = close.pct_change(window)
        frame[f"realized_vol_{window}d"] = annualized_realized_volatility(returns, window)
        frame[f"drawdown_{window}d"] = close / close.rolling(window).max() - 1.0
        frame[f"ma_distance_{window}d"] = close / close.rolling(window).mean() - 1.0
        if {"high", "low"}.issubset(asset.columns):
            frame[f"parkinson_vol_{window}d"] = parkinson_volatility(
                asset["high"],
                asset["low"],
                window,
            )
        if {"open", "high", "low", "close"}.issubset(asset.columns):
            frame[f"garman_klass_vol_{window}d"] = garman_klass_volatility(
                asset["open"],
                asset["high"],
                asset["low"],
                raw_close,
                window,
            )

    target_col = f"future_realized_vol_{horizon_days}d"
    frame[target_col] = forward_realized_volatility(returns, horizon_days)
    threshold_source = frame[target_col]
    if threshold_end is not None:
        threshold_source = frame.loc[
            pd.to_datetime(frame["date"]) <= pd.Timestamp(threshold_end),
            target_col,
        ]
    threshold = float(threshold_source.dropna().quantile(threshold_quantile))
    if not np.isfinite(threshold):
        raise ValueError("cannot estimate volatility threshold from the selected data")
    target = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    valid_target = frame[target_col].notna()
    target.loc[valid_target] = (frame.loc[valid_target, target_col] > threshold).astype("Int64")
    frame["target_high_vol"] = target
    frame["target_threshold"] = threshold
    frame["feature_available_date"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.to_datetime(frame["date"]).shift(-horizon_days)
    if not include_unlabeled:
        frame = frame.dropna(subset=[target_col])
    frame = frame.reset_index(drop=True)
    return VolatilityRegimeResult(
        frame=frame,
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        threshold_end=threshold_end,
    )


def annualized_realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def forward_realized_volatility(returns: pd.Series, horizon_days: int) -> pd.Series:
    values = returns.to_numpy(dtype=float)
    output = np.full(len(values), np.nan)
    for idx in range(len(values)):
        sample = values[idx + 1 : idx + 1 + horizon_days]
        sample = sample[np.isfinite(sample)]
        if len(sample) == horizon_days:
            output[idx] = sample.std(ddof=1) * math.sqrt(TRADING_DAYS)
    return pd.Series(output, index=returns.index)


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    ratio = (high.astype(float) / low.astype(float)).where(lambda item: item > 0)
    variance = (np.log(ratio) ** 2).rolling(window).mean() / (4 * math.log(2))
    return np.sqrt(variance * TRADING_DAYS)


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    high_low = np.log((high.astype(float) / low.astype(float)).where(lambda item: item > 0))
    close_open = np.log((close.astype(float) / open_.astype(float)).where(lambda item: item > 0))
    variance = 0.5 * high_low.pow(2) - (2 * math.log(2) - 1) * close_open.pow(2)
    variance = variance.clip(lower=0).rolling(window).mean()
    return np.sqrt(variance * TRADING_DAYS)


def alert_operating_points(
    scores: pd.DataFrame,
    *,
    score_columns: list[str],
    target_column: str = "target_high_vol",
    recalls: Iterable[float] = (0.5, 0.8, 0.9),
) -> pd.DataFrame:
    """Translate risk scores into alert-queue operating points."""

    if target_column not in scores.columns:
        raise ValueError(f"missing target column: {target_column}")
    missing_scores = sorted(set(score_columns) - set(scores.columns))
    if missing_scores:
        raise ValueError(f"missing score columns: {missing_scores}")

    rows = []
    for score_column in score_columns:
        usable = scores[[score_column, target_column]].dropna().sort_values(
            score_column,
            ascending=False,
        )
        positives = usable[target_column].astype(int)
        total_positives = int(positives.sum())
        for recall in recalls:
            if not 0 < recall <= 1:
                raise ValueError("recalls must be in the interval (0, 1]")
            if total_positives == 0 or usable.empty:
                rows.append(_empty_alert_row(score_column, recall, total_positives))
                continue
            required_hits = int(math.ceil(total_positives * recall))
            cumulative_hits = positives.cumsum()
            hit_positions = np.flatnonzero(cumulative_hits.to_numpy() >= required_hits)
            if len(hit_positions) == 0:
                rows.append(_empty_alert_row(score_column, recall, total_positives))
                continue
            position = int(hit_positions[0])
            alerts = position + 1
            hits = int(cumulative_hits.iloc[position])
            rows.append(
                {
                    "score": score_column,
                    "target_recall": float(recall),
                    "threshold": float(usable[score_column].iloc[position]),
                    "alerts": alerts,
                    "hits": hits,
                    "total_positives": total_positives,
                    "precision": float(hits / alerts),
                    "realized_recall": float(hits / total_positives),
                }
            )
    return pd.DataFrame(rows)


def _asset_frame(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"date", "symbol", "adj_close"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"price data missing columns: {missing}")
    frame = prices[prices["symbol"] == symbol].copy()
    if frame.empty:
        raise ValueError(f"symbol not found in price data: {symbol}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "adj_close"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "close" not in frame.columns:
        frame["close"] = frame["adj_close"]
    frame = _apply_ohlc_adjustment(frame)
    return frame


def _apply_ohlc_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"open", "high", "low", "close", "adj_close"}.issubset(frame.columns):
        return frame
    adjusted = frame.copy()
    factor = (adjusted["adj_close"] / adjusted["close"]).replace([np.inf, -np.inf], np.nan)
    factor = factor.where(factor > 0)
    for column in ["open", "high", "low", "close"]:
        adjusted[column] = adjusted[column] * factor
    adjusted["close"] = adjusted["adj_close"]
    return adjusted


def _empty_alert_row(score_column: str, recall: float, total_positives: int) -> dict:
    return {
        "score": score_column,
        "target_recall": float(recall),
        "threshold": float("nan"),
        "alerts": 0,
        "hits": 0,
        "total_positives": total_positives,
        "precision": float("nan"),
        "realized_recall": float("nan"),
    }


__all__ = [
    "TRADING_DAYS",
    "VolatilityRegimeResult",
    "alert_operating_points",
    "annualized_realized_volatility",
    "build_volatility_regime_frame",
    "forward_realized_volatility",
    "garman_klass_volatility",
    "parkinson_volatility",
]
