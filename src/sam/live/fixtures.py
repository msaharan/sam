"""Deprecated local live fixtures retained for unit tests only."""

from __future__ import annotations

import warnings

from sam.live.offline import OfflineBrokerStub, ParquetReplayFeed

warnings.warn(
    "sam.live.fixtures is deprecated; use sam.live.offline for production replay paths",
    DeprecationWarning,
    stacklevel=2,
)

DemoBroker = OfflineBrokerStub
SyntheticBarFeed = ParquetReplayFeed


class PreviewBroker:
    """Deprecated dry-run broker; use sam.live.runner.run_preview instead."""

    def __init__(self) -> None:
        self.orders: list[dict] = []
        self._positions: dict = {}

    def get_position(self, asset):
        return self._positions.get(asset)

    def submit_order(self, asset, qty, side=None, **kwargs):
        self.orders.append(
            {
                "asset": asset,
                "quantity": qty,
                "side": getattr(side, "value", str(side)),
                **kwargs,
            }
        )
