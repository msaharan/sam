"""Risk-control policy diagnostics for volatility-regime scores."""

from __future__ import annotations

import pandas as pd

from sam.data import price_panel


def run_risk_policy_diagnostic(
    frame: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    score_column: str,
    calibration_mask: pd.Series,
    holdout_mask: pd.Series,
    action_rate: float = 0.2,
    high_risk_weight: float = 0.5,
    cost_bps: float = 1.0,
    symbol: str = "SPY",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a simple exposure-reduction policy and return daily returns and summary."""

    close = price_panel(prices)[symbol]
    returns = close.pct_change(fill_method=None)
    aligned = frame[["date", score_column]].copy()
    aligned["date"] = pd.to_datetime(aligned["date"])
    aligned = aligned.set_index("date").sort_index()
    cal_mask = calibration_mask.reindex(aligned.index, fill_value=False)
    calibration_scores = aligned.loc[cal_mask, score_column]
    threshold = float(calibration_scores.quantile(1 - action_rate))
    weights = pd.Series(1.0, index=aligned.index)
    weights.loc[aligned[score_column] >= threshold] = high_risk_weight
    weights = weights.reindex(returns.index).ffill().fillna(1.0)
    shifted_weights = weights.shift(1).fillna(1.0)
    policy_returns = shifted_weights * returns
    turnover = shifted_weights.diff().abs().fillna(shifted_weights.abs())
    costs = turnover * (cost_bps / 10_000)
    net_returns = policy_returns - costs
    holdout_index = holdout_mask.reindex(net_returns.index, fill_value=False)
    holdout_returns = net_returns.loc[holdout_index].dropna()
    equity = (1 + holdout_returns).cumprod()
    summary = pd.DataFrame(
        [
            {
                "score": score_column,
                "threshold": threshold,
                "action_rate_target": action_rate,
                "high_risk_weight": high_risk_weight,
                "cost_bps": cost_bps,
                "sharpe": _sharpe(holdout_returns),
                "final_equity": float(equity.iloc[-1]) if not equity.empty else float("nan"),
                "max_drawdown": _max_drawdown(equity),
            }
        ]
    )
    daily = pd.DataFrame(
        {
            "date": holdout_returns.index,
            "policy_return": holdout_returns.to_numpy(),
            "equity": equity.to_numpy(),
            "weight": shifted_weights.loc[holdout_index].to_numpy(),
        }
    )
    return daily, summary


def _sharpe(returns: pd.Series) -> float:
    if returns.std(ddof=1) == 0 or returns.empty:
        return float("nan")
    return float(returns.mean() / returns.std(ddof=1) * (252**0.5))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


__all__ = ["run_risk_policy_diagnostic"]
