"""Offline live replay using committed parquet fixtures and ML4T shadow mode."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from ml4t.backtest.types import Order, Position


class OfflineBrokerStub:
    """Minimal broker stub required by SafeBroker in offline shadow mode."""

    def __init__(self) -> None:
        self._connected = False
        self._positions: dict[str, Position] = {}
        self._pending_orders: list[Order] = []
        self._cash = 100_000.0

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def pending_orders(self) -> list[Order]:
        return list(self._pending_orders)

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def is_connected_async(self) -> bool:
        return self._connected

    async def get_positions_async(self) -> dict[str, Position]:
        return dict(self._positions)

    async def get_pending_orders_async(self) -> list[Order]:
        return list(self._pending_orders)

    async def get_position_async(self, asset: str) -> Position | None:
        return self._positions.get(asset)

    async def get_account_value_async(self) -> float:
        return self._cash

    async def get_cash_async(self) -> float:
        return self._cash

    async def submit_order_async(self, *args, **kwargs):
        raise RuntimeError("OfflineBrokerStub should not receive orders in shadow mode")

    async def cancel_order_async(self, order_id: str) -> bool:
        return False

    async def close_position_async(self, asset: str):
        return None


class ParquetReplayFeed:
    """Replay OHLCV bars from a parquet fixture for offline shadow/preview runs."""

    def __init__(
        self,
        parquet_path: str | Path,
        symbols: list[str],
        *,
        max_bars: int | None = None,
        sleep_seconds: float = 0.05,
        date_col: str = "date",
        entity_col: str = "symbol",
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.symbols = symbols
        self.max_bars = max_bars
        self.sleep_seconds = sleep_seconds
        self.date_col = date_col
        self.entity_col = entity_col
        self._running = False
        self._index = 0
        self._bars: list[tuple[datetime, dict, dict]] = []
        self._load_bars()

    def _load_bars(self) -> None:
        frame = pl.read_parquet(self.parquet_path)
        if self.entity_col not in frame.columns and "ticker" in frame.columns:
            frame = frame.rename({"ticker": self.entity_col})
        if self.date_col not in frame.columns and "timestamp" in frame.columns:
            frame = frame.rename({"timestamp": self.date_col})
        frame = frame.filter(pl.col(self.entity_col).is_in(self.symbols)).sort(
            [self.date_col, self.entity_col]
        )
        timestamps = frame[self.date_col].unique().sort().to_list()
        if self.max_bars is not None:
            timestamps = timestamps[: self.max_bars]

        for ts in timestamps:
            slice_df = frame.filter(pl.col(self.date_col) == ts)
            data: dict[str, dict] = {}
            for row in slice_df.iter_rows(named=True):
                symbol = row[self.entity_col]
                data[symbol] = {
                    "open": float(row.get("open", row["close"])),
                    "high": float(row.get("high", row["close"])),
                    "low": float(row.get("low", row["close"])),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0)),
                }
            timestamp = _as_utc_datetime(ts)
            self._bars.append((timestamp, data, {}))

    async def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def __aiter__(self) -> AsyncIterator[tuple[datetime, dict, dict]]:
        return self

    async def __anext__(self) -> tuple[datetime, dict, dict]:
        if not self._running or self._index >= len(self._bars):
            raise StopAsyncIteration
        bar = self._bars[self._index]
        self._index += 1
        if self.sleep_seconds:
            await asyncio.sleep(self.sleep_seconds)
        return bar


def _as_utc_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


def default_fixture_path(root: Path) -> Path:
    return root / "tests/fixtures/ohlcv_ma_baseline.parquet"
