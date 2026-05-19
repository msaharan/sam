"""Long-only portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sam.config import DEFAULT_ETFS, PortfolioConfig, PortfolioMethod


def build_portfolio(
    prices: pd.DataFrame,
    *,
    method: PortfolioMethod = "inverse_volatility",
    config: PortfolioConfig | None = None,
) -> pd.DataFrame:
    cfg = config or PortfolioConfig(method=method)
    panel = _price_panel(prices).dropna(axis=1, thresh=cfg.min_history_days)
    if cfg.method == "etf_baseline":
        etfs = [symbol for symbol in DEFAULT_ETFS if symbol in panel.columns]
        if not etfs:
            raise ValueError("ETF baseline requires at least one default ETF in price data")
        panel = panel[etfs]
    returns = panel.pct_change(fill_method=None).dropna(how="all")
    if returns.empty:
        raise ValueError("not enough price history to build portfolio")
    weights = _weights_for_returns(returns.tail(cfg.lookback_days), cfg.method, cfg.max_weight)
    return weights.rename("weight").reset_index().rename(columns={"index": "symbol"})


def portfolio_diagnostics(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    config: PortfolioConfig | None = None,
) -> pd.DataFrame:
    cfg = config or PortfolioConfig()
    panel = _price_panel(prices)
    returns = panel.pct_change(fill_method=None)
    lookback = returns.tail(cfg.lookback_days)
    rows = []
    for symbol in sorted(panel.columns):
        asset_returns = lookback[symbol].dropna()
        rows.append(
            {
                "symbol": symbol,
                "weight": float(
                    weights.set_index("symbol")["weight"].reindex([symbol]).fillna(0.0).iloc[0]
                ),
                "history_days": int(panel[symbol].dropna().shape[0]),
                "lookback_return_days": int(asset_returns.shape[0]),
                "lookback_total_return": float((1.0 + asset_returns).prod() - 1.0)
                if not asset_returns.empty
                else float("nan"),
                "lookback_volatility": float(asset_returns.std(ddof=1))
                if len(asset_returns) > 1
                else float("nan"),
                "eligible": bool(panel[symbol].dropna().shape[0] >= cfg.min_history_days),
            }
        )
    return pd.DataFrame(rows).sort_values(["weight", "symbol"], ascending=[False, True])


def _weights_for_returns(
    returns: pd.DataFrame,
    method: PortfolioMethod,
    max_weight: float,
) -> pd.Series:
    returns = returns.dropna(axis=1, how="any")
    if returns.empty:
        raise ValueError("no assets with complete return history")
    if method in {"equal_weight", "etf_baseline"}:
        raw = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    elif method == "inverse_volatility":
        vol = returns.std(ddof=1).replace(0, np.nan)
        raw = (1 / vol).replace([np.inf, -np.inf], np.nan).dropna()
    elif method == "minimum_variance":
        raw = _minimum_variance_weights(returns)
    elif method == "risk_parity":
        raw = _risk_parity_weights(returns)
    elif method == "momentum_tilt":
        momentum = (1 + returns).prod() - 1
        positive = momentum.clip(lower=0)
        raw = positive if positive.sum() > 0 else momentum.rank(pct=True)
    else:
        raise ValueError(f"unknown portfolio method: {method}")
    capped = _cap_and_normalize(raw, max_weight)
    return capped.sort_index()


def _price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    if {"date", "symbol", "adj_close"}.issubset(prices.columns):
        frame = prices.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    return prices.copy().sort_index()


def _minimum_variance_weights(returns: pd.DataFrame) -> pd.Series:
    covariance = returns.cov().to_numpy()
    n_assets = len(returns.columns)
    x0 = np.full(n_assets, 1 / n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    result = minimize(
        lambda weights: weights @ covariance @ weights,
        x0,
        bounds=bounds,
        constraints=constraints,
    )
    weights = result.x if result.success else x0
    return pd.Series(weights, index=returns.columns)


def _risk_parity_weights(returns: pd.DataFrame) -> pd.Series:
    covariance = returns.cov().to_numpy()
    n_assets = len(returns.columns)
    x0 = np.full(n_assets, 1 / n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    def objective(weights: np.ndarray) -> float:
        portfolio_var = weights @ covariance @ weights
        if portfolio_var <= 0:
            return 1e6
        marginal = covariance @ weights
        contribution = weights * marginal / portfolio_var
        target = np.full(n_assets, 1 / n_assets)
        return float(((contribution - target) ** 2).sum())

    result = minimize(objective, x0, bounds=bounds, constraints=constraints)
    weights = result.x if result.success else x0
    return pd.Series(weights, index=returns.columns)


def _cap_and_normalize(weights: pd.Series, max_weight: float) -> pd.Series:
    weights = weights.astype(float).clip(lower=0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=weights.index)
    weights = weights / weights.sum()
    for _ in range(20):
        over = weights > max_weight
        if not over.any():
            break
        excess = float((weights[over] - max_weight).sum())
        weights[over] = max_weight
        under = ~over
        if not under.any() or weights[under].sum() <= 0:
            break
        weights[under] += excess * weights[under] / weights[under].sum()
    return weights / weights.sum()
