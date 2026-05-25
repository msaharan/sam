from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl


def panel_to_training_batch(
    panel: pl.DataFrame,
    feature_cols: list[str],
    *,
    entity_col: str = "symbol",
    date_col: str = "date",
    train_fraction: float = 0.7,
    batch_type: Literal["cross_section", "persistent_panel"] = "persistent_panel",
    schema: Any | None = None,
) -> tuple[Any, list[Any], list[Any], list[Any], list[Any]]:
    """Build ML4T training batches from a SAM feature panel."""
    from ml4t.models import (
        PortfolioSequenceBatch,
        cross_section_batch_from_long_frame,
        persistent_panel_batch_from_long_frame,
    )

    dates = panel[date_col].unique().sort().to_list()
    if len(dates) < 3:
        raise ValueError("Need at least three dates to build a training batch")
    split_idx = max(1, min(len(dates) - 1, int(len(dates) * train_fraction)))
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    common_kwargs = {
        "feature_cols": feature_cols,
        "return_col": "forward_return",
        "timestamp_col": date_col,
        "entity_col": entity_col,
        "schema": schema,
    }

    if batch_type == "cross_section":
        batch = cross_section_batch_from_long_frame(panel, **common_kwargs)
        return batch, train_dates, test_dates, train_dates, test_dates

    panel_batch = persistent_panel_batch_from_long_frame(panel, **common_kwargs)
    returns = panel_batch.returns
    characteristics = panel_batch.characteristics
    if returns is None or characteristics is None:
        raise ValueError("persistent_panel batch requires returns and feature columns")

    n_periods, n_assets, n_features = characteristics.shape
    mask = np.isfinite(returns)
    batch = PortfolioSequenceBatch(
        features=characteristics[np.newaxis, ...],
        returns=returns[np.newaxis, ...],
        vol_scale=np.ones((1, n_periods, n_assets), dtype=np.float64),
        mask=mask[np.newaxis, ...],
        timestamps=tuple(str(d) for d in panel_batch.timestamps),
        asset_ids=panel_batch.asset_ids,
    )
    return batch, train_dates, test_dates, train_dates, test_dates


def weights_to_signal_scores(weights_frame: Any, entity_col: str = "symbol") -> pl.DataFrame:
    """Convert ML4T signal/weight frames to SAM signal_rank parquet schema."""
    frame = weights_frame.to_polars()
    rename = {}
    if "asset" in frame.columns:
        rename["asset"] = entity_col
    if "timestamp" in frame.columns:
        rename["timestamp"] = "date"
    if "signal_value" in frame.columns:
        rename["signal_value"] = "score"
    elif "weight" in frame.columns:
        rename["weight"] = "score"
    elif "prediction_value" in frame.columns:
        rename["prediction_value"] = "score"
    frame = frame.rename(rename)
    keep = ["date", entity_col, "score"]
    return frame.select([c for c in keep if c in frame.columns])
