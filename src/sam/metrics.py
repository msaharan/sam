"""Traditional quant finance performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def calculate_metrics(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    series = _as_return_series(returns).dropna()
    if series.empty:
        return _empty_metrics()
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = series - daily_rf
    equity = (1.0 + series).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    downside = excess[excess < 0]
    volatility = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS))
    cagr = _cagr(equity)
    sharpe = _safe_div(float(excess.mean() * TRADING_DAYS), volatility)
    sortino = _safe_div(
        float(excess.mean() * TRADING_DAYS),
        float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else np.nan,
    )
    max_drawdown = float(drawdown.min())
    calmar = _safe_div(cagr, abs(max_drawdown))
    payload = {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "var_95": float(series.quantile(0.05)),
        "cvar_95": float(series[series <= series.quantile(0.05)].mean()),
        "hit_rate": float((series > 0).mean()),
        "drawdown_days": float(_max_drawdown_duration(drawdown)),
        "total_return": float(equity.iloc[-1] - 1.0),
    }
    if benchmark is not None:
        aligned = pd.concat([series, benchmark.rename("benchmark")], axis=1).dropna()
        if not aligned.empty:
            active = aligned.iloc[:, 0] - aligned["benchmark"]
            beta = _safe_div(
                float(aligned.iloc[:, 0].cov(aligned["benchmark"])),
                float(aligned["benchmark"].var()),
            )
            alpha_daily = float(aligned.iloc[:, 0].mean() - beta * aligned["benchmark"].mean())
            payload.update(
                {
                    "beta": beta,
                    "alpha": alpha_daily * TRADING_DAYS,
                    "tracking_error": float(active.std(ddof=1) * np.sqrt(TRADING_DAYS)),
                    "information_ratio": _safe_div(
                        float(active.mean() * TRADING_DAYS),
                        float(active.std(ddof=1) * np.sqrt(TRADING_DAYS)),
                    ),
                }
            )
    return payload


def _as_return_series(returns: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(returns, pd.Series):
        return returns.astype(float)
    if "portfolio_return" in returns.columns:
        return returns["portfolio_return"].astype(float)
    if returns.shape[1] == 1:
        return returns.iloc[:, 0].astype(float)
    raise ValueError(
        "returns must be a Series, a one-column DataFrame, or include portfolio_return"
    )


def _cagr(equity: pd.Series) -> float:
    years = max(len(equity) / TRADING_DAYS, 1 / TRADING_DAYS)
    return float(equity.iloc[-1] ** (1 / years) - 1)


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    longest = current = 0
    for value in drawdown:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return float("nan")
    return float(numerator / denominator)


def _empty_metrics() -> dict[str, float]:
    keys = [
        "cagr",
        "volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "hit_rate",
        "drawdown_days",
        "total_return",
    ]
    return dict.fromkeys(keys, float("nan"))
