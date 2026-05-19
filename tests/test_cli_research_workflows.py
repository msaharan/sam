from __future__ import annotations

from pathlib import Path

import pandas as pd

from sam.cli import main
from sam.io import write_table


def test_cli_builds_volatility_alert_and_allocation_workflows(
    tmp_path: Path,
    sample_prices: pd.DataFrame,
) -> None:
    prices_path = tmp_path / "prices.parquet"
    volatility_path = tmp_path / "volatility.csv"
    alerts_path = tmp_path / "alerts.csv"
    scores_path = tmp_path / "scores.csv"
    weights_path = tmp_path / "weights.csv"
    turnover_path = tmp_path / "turnover.csv"
    write_table(sample_prices, prices_path)

    assert (
        main(
            [
                "risk",
                "volatility-frame",
                "--prices",
                str(prices_path),
                "--symbol",
                "SPY",
                "--horizon-days",
                "20",
                "--threshold-end",
                "2020-12-31",
                "--out",
                str(volatility_path),
            ]
        )
        == 0
    )
    volatility = pd.read_csv(volatility_path)
    assert "realized_vol_20d" in volatility.columns

    assert (
        main(
            [
                "risk",
                "alerts",
                "--scores",
                str(volatility_path),
                "--score-columns",
                "realized_vol_20d",
                "parkinson_vol_20d",
                "--out",
                str(alerts_path),
            ]
        )
        == 0
    )
    assert not pd.read_csv(alerts_path).empty

    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 5 + ["2020-02-29"] * 5),
            "symbol": ["A", "B", "C", "D", "E"] * 2,
            "prediction": [5, 4, 3, 2, 1, 1, 5, 4, 3, 2],
        }
    )
    write_table(scores, scores_path)
    assert (
        main(
            [
                "allocation",
                "weights",
                "--scores",
                str(scores_path),
                "--top-k",
                "3",
                "--max-weight",
                "0.5",
                "--out",
                str(weights_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "allocation",
                "turnover",
                "--weights",
                str(weights_path),
                "--transaction-cost-bps",
                "5",
                "--slippage-bps",
                "1",
                "--out",
                str(turnover_path),
            ]
        )
        == 0
    )
    assert pd.read_csv(turnover_path)["cost"].sum() > 0
