import pytest


def test_research_features_runs():
    pytest.importorskip("ml4t.engineer", reason="Install sam[research] (requires CMake for numba)")

    from pathlib import Path

    from sam.pipeline.research_run import run_features

    out = run_features(
        "tests/fixtures/ohlcv_ma_baseline.parquet",
        Path("artifacts/test_research/features"),
        ["rsi"],
    )
    assert out.exists()


def test_research_train_writes_model_and_manifest(tmp_path):
    pytest.importorskip("ml4t.models", reason="Install sam[research] (requires ml4t-models)")
    import json

    import polars as pl
    from ml4t.models.types import PortfolioSequenceBatch

    from sam.pipeline.adapters import panel_to_training_batch
    from sam.pipeline.research_run import run_train

    prices_path = "tests/fixtures/ohlcv_ma_baseline.parquet"
    prices = pl.read_parquet(prices_path)
    features = (
        prices.sort(["symbol", "date"])
        .with_columns(
            pl.col("close").pct_change().over("symbol").fill_null(0.0).alias("ret1"),
            pl.col("close").rolling_mean(3).over("symbol").fill_null(pl.col("close")).alias("ma3"),
        )
        .select("date", "symbol", "ret1", "ma3")
    )
    features_path = tmp_path / "features.parquet"
    features.write_parquet(features_path)

    panel = (
        features.join(
            prices.select("date", "symbol", "close")
            .sort(["symbol", "date"])
            .with_columns(
                pl.col("close").pct_change().shift(-1).over("symbol").alias("forward_return")
            ),
            on=["date", "symbol"],
            how="inner",
        )
    )
    batch, _, _, _, _ = panel_to_training_batch(
        panel,
        ["ret1", "ma3"],
        batch_type="persistent_panel",
    )
    assert isinstance(batch, PortfolioSequenceBatch)

    cross_batch, _, _, _, _ = panel_to_training_batch(
        panel,
        ["ret1", "ma3"],
        batch_type="cross_section",
    )
    from ml4t.models.types import CrossSectionBatch

    assert isinstance(cross_batch, CrossSectionBatch)

    signals_path = run_train(prices_path, str(features_path), tmp_path / "signals")

    assert signals_path.exists()
    assert (tmp_path / "signals" / "model.json").exists()
    assert (tmp_path / "signals" / "run_manifest.json").exists()
    model = json.loads((tmp_path / "signals" / "model.json").read_text())
    assert model["model_type"] == "linear_feature_portfolio"
