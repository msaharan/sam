from __future__ import annotations

from ml4t.backtest import Strategy
from ml4t.backtest.types import OrderSide


class SignalRankStrategy(Strategy):
    """Trade on precomputed per-bar signal scores (research pipeline output)."""

    def __init__(
        self,
        symbols: list[str],
        signal_column: str = "score",
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        top_k: int = 3,
        order_quantity: float = 5,
    ) -> None:
        self.symbols = list(symbols)
        self.signal_column = signal_column
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.top_k = top_k
        self.order_quantity = order_quantity

    def _score(self, bar: dict) -> float | None:
        signals = bar.get("signals") or {}
        if self.signal_column in signals:
            return float(signals[self.signal_column])
        if self.signal_column in bar:
            return float(bar[self.signal_column])
        return None

    def on_data(self, timestamp, data, context, broker) -> None:
        ranked: list[tuple[str, float]] = []
        for symbol in self.symbols:
            bar = data.get(symbol)
            if bar is None:
                continue
            score = self._score(bar)
            if score is not None:
                ranked.append((symbol, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        targets = {sym for sym, _ in ranked[: self.top_k] if _ >= self.buy_threshold}

        for symbol in self.symbols:
            bar = data.get(symbol)
            if bar is None:
                continue
            score = self._score(bar)
            position = broker.get_position(symbol)
            qty = position.quantity if position else 0

            if symbol in targets and qty <= 0:
                broker.submit_order(symbol, self.order_quantity, side=OrderSide.BUY)
            elif (score is not None and score < self.sell_threshold) or symbol not in targets:
                if qty > 0:
                    broker.submit_order(symbol, qty, side=OrderSide.SELL)
