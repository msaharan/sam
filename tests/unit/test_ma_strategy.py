from sam.strategies.ma_crossover import MACrossoverStrategy


class _Broker:
    def __init__(self):
        self.orders = []
        self._positions = {}

    def get_position(self, asset):
        return self._positions.get(asset)

    def submit_order(self, asset, qty, side=None, **kwargs):
        self.orders.append((asset, qty, side))


def test_ma_signal_change_triggers_order():
    strategy = MACrossoverStrategy(["SPY"], ma_period=3, order_quantity=5)
    broker = _Broker()
    ts = None
    for close in [100, 101, 102, 103, 99, 98]:
        strategy.on_data(ts, {"SPY": {"close": close}}, {}, broker)
    assert broker.orders
