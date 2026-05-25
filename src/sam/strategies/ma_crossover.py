from __future__ import annotations

from ml4t.backtest import Strategy
from ml4t.backtest.types import OrderSide


class MACrossoverStrategy(Strategy):
    """Moving-average crossover; identical logic for backtest and live."""

    def __init__(
        self,
        symbols: list[str],
        ma_period: int = 10,
        order_quantity: float = 10,
    ) -> None:
        self.symbols = list(symbols)
        self.ma_period = ma_period
        self.order_quantity = order_quantity
        self.price_history: dict[str, list[float]] = {s: [] for s in self.symbols}
        self.last_signal: dict[str, str | None] = dict.fromkeys(self.symbols)

    def on_data(self, timestamp, data, context, broker) -> None:
        for symbol in self.symbols:
            bar = data.get(symbol)
            if bar is None:
                continue

            close = float(bar.get("close", bar.get("price", 0)))
            history = self.price_history[symbol]
            history.append(close)
            if len(history) > self.ma_period:
                del history[:-self.ma_period]

            if len(history) < self.ma_period:
                continue

            ma = sum(history[-self.ma_period :]) / self.ma_period
            signal = "BUY" if close > ma else "SELL"
            if signal == self.last_signal[symbol]:
                continue
            self.last_signal[symbol] = signal

            position = broker.get_position(symbol)
            current_qty = position.quantity if position else 0

            if signal == "BUY" and current_qty <= 0:
                qty = (
                    self.order_quantity
                    if current_qty == 0
                    else abs(current_qty) + self.order_quantity
                )
                broker.submit_order(symbol, qty, side=OrderSide.BUY)
            elif signal == "SELL" and current_qty > 0:
                broker.submit_order(symbol, current_qty, side=OrderSide.SELL)
