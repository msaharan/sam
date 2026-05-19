"""Feature engineering for cross-sectional return models."""

from __future__ import annotations

import pandas as pd

from sam.data import price_panel


def build_feature_frame(
    prices: pd.DataFrame,
    *,
    benchmark_symbol: str | None = None,
    horizon_days: int = 21,
    include_target: bool = True,
) -> pd.DataFrame:
    panel = price_panel(prices)
    volume_panel = (
        price_panel(prices, value="volume") if "volume" in prices.columns else panel * 0 + 1
    )
    returns = panel.pct_change(fill_method=None)
    features = []
    benchmark_returns = (
        returns[benchmark_symbol] if benchmark_symbol in returns.columns else returns.mean(axis=1)
    )
    for symbol in panel.columns:
        series = panel[symbol]
        asset_returns = returns[symbol]
        frame = pd.DataFrame(
            {
                "date": panel.index,
                "symbol": symbol,
                "ret_21d": series.pct_change(21),
                "ret_63d": series.pct_change(63),
                "ret_126d": series.pct_change(126),
                "vol_21d": asset_returns.rolling(21).std(),
                "vol_63d": asset_returns.rolling(63).std(),
                "drawdown_63d": series / series.rolling(63).max() - 1.0,
                "ma_ratio_20_100": series.rolling(20).mean() / series.rolling(100).mean() - 1.0,
                "beta_63d": asset_returns.rolling(63).cov(benchmark_returns)
                / benchmark_returns.rolling(63).var(),
                "volume_21d": volume_panel[symbol].rolling(21).mean(),
                "dollar_volume_21d": (series * volume_panel[symbol]).rolling(21).mean(),
            }
        )
        if include_target:
            frame["target_forward_return"] = series.shift(-horizon_days) / series - 1.0
        features.append(frame)
    result = pd.concat(features, ignore_index=True)
    rank_cols = ["ret_21d", "ret_63d", "ret_126d", "vol_63d", "dollar_volume_21d"]
    for column in rank_cols:
        result[f"{column}_rank"] = result.groupby("date")[column].rank(pct=True)
    result["regime_vol_63d"] = result.groupby("date")["vol_63d"].transform("median")
    required = [
        "ret_21d",
        "ret_63d",
        "ret_126d",
        "vol_21d",
        "vol_63d",
        "volume_21d",
        "dollar_volume_21d",
        "drawdown_63d",
        "ma_ratio_20_100",
        "beta_63d",
        "ret_21d_rank",
        "ret_63d_rank",
        "ret_126d_rank",
        "vol_63d_rank",
        "dollar_volume_21d_rank",
        "regime_vol_63d",
    ]
    if include_target:
        required.append("target_forward_return")
    return result.dropna(subset=required).reset_index(drop=True)
