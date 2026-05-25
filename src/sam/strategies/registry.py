from __future__ import annotations

from pathlib import Path

from sam.config.loader import StrategyConfig, load_yaml, resolve_symbols
from sam.strategies.ma_crossover import MACrossoverStrategy
from sam.strategies.signal_rank import SignalRankStrategy


def build_strategy(config_path: str | Path, root: Path | None = None):
    cfg_path = Path(config_path)
    if root and not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    raw = load_yaml(cfg_path)
    cfg = StrategyConfig.model_validate(raw)
    symbols = resolve_symbols(cfg, root=root)
    if cfg.strategy == "ma_crossover":
        return MACrossoverStrategy(
            symbols=symbols,
            ma_period=cfg.ma_period,
            order_quantity=cfg.order_quantity,
        )
    if cfg.strategy in ("signal_rank", "etf_tactical"):
        return SignalRankStrategy(
            symbols=symbols,
            signal_column=cfg.signal_column,
            buy_threshold=cfg.buy_threshold,
            sell_threshold=cfg.sell_threshold,
            top_k=cfg.top_k,
            order_quantity=cfg.order_quantity,
        )
    raise ValueError(f"Unknown strategy: {cfg.strategy}")
