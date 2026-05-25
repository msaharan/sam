#!/usr/bin/env python3
"""Generate committed OHLCV and signal fixtures for offline tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _ohlcv(symbols: list[str], days: int = 60) -> pl.DataFrame:
    start = datetime(2024, 1, 2)
    rows: list[dict] = []
    for i in range(days):
        d = start + timedelta(days=i)
        for j, symbol in enumerate(symbols):
            base = 100.0 + j * 10 + i * 0.15
            rows.append(
                {
                    "date": d,
                    "symbol": symbol,
                    "open": base - 0.3,
                    "high": base + 0.5,
                    "low": base - 0.6,
                    "close": base + (0.2 if (i + j) % 7 < 4 else -0.2),
                    "volume": 1_000_000 + i * 1000,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    etf_symbols = ["SPY", "QQQ", "IWM"]
    prices = _ohlcv(etf_symbols)
    prices_path = FIXTURES / "ohlcv_ma_baseline.parquet"
    prices.write_parquet(prices_path)

    signals = (
        prices.sort(["symbol", "date"])
        .with_columns(pl.col("close").pct_change().over("symbol").fill_null(0).alias("ret1"))
        .with_columns((0.5 + pl.col("ret1").clip(-0.05, 0.05) * 5).alias("score"))
        .select("date", "symbol", "score")
    )
    signals_path = FIXTURES / "signals_etf_rank.parquet"
    signals.write_parquet(signals_path)

    equity_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    equity_prices = _ohlcv(equity_symbols)
    equity_prices_path = FIXTURES / "ohlcv_equity_sample.parquet"
    equity_prices.write_parquet(equity_prices_path)

    equity_signals = (
        equity_prices.sort(["symbol", "date"])
        .with_columns(pl.col("close").pct_change().over("symbol").fill_null(0).alias("ret1"))
        .with_columns((0.5 + pl.col("ret1").clip(-0.05, 0.05) * 5).alias("score"))
        .select("date", "symbol", "score")
    )
    equity_signals_path = FIXTURES / "signals_equity_rank.parquet"
    equity_signals.write_parquet(equity_signals_path)

    print(f"Wrote {prices_path} ({prices.height} rows)")
    print(f"Wrote {signals_path} ({signals.height} rows)")
    print(f"Wrote {equity_prices_path} ({equity_prices.height} rows)")
    print(f"Wrote {equity_signals_path} ({equity_signals.height} rows)")


if __name__ == "__main__":
    main()
