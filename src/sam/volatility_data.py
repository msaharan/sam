"""Market panel ingestion for volatility-regime research."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from sam.config import load_universe
from sam.data import CANONICAL_COLUMNS, fetch_prices

logger = logging.getLogger(__name__)

FRED_SETUP_HINT = (
    "Macro (FRED) series are required for a full live run. Use one of:\n"
    "  export FRED_API_KEY=<key>   # free: https://fred.stlouisfed.org/docs/api/api_key.html\n"
    "  sam experiment run volatility-regime-scoring --fred-csv path/to/fred.csv\n"
    "  place fred.csv in --cache-dir (e.g. data/research/volatility_regime/fred.csv)\n"
    "  sam experiment run volatility-regime-scoring --synthetic --fast  # offline smoke, no FRED"
)

VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

ETF_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "HYG", "EEM", "VNQ"]
DEFAULT_VOLATILITY_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "tests/fixtures/volatility_regime"
)


def fetch_vix_history(
    *,
    start: str | None = None,
    end: str | None = None,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load Cboe VIX daily history as date/close."""

    if csv_path is not None:
        raw = pd.read_csv(csv_path)
    else:
        logger.info("fetching VIX history from Cboe")
        with urlopen(VIX_HISTORY_URL, timeout=60) as response:
            raw = pd.read_csv(io.BytesIO(response.read()))
    date_col = "DATE" if "DATE" in raw.columns else "Date"
    close_col = "CLOSE" if "CLOSE" in raw.columns else "Close"
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]),
            "vix_close": pd.to_numeric(raw[close_col], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["date", "vix_close"]).sort_values("date")
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    return frame.reset_index(drop=True)


def fetch_fred_series(
    series_map: dict[str, str],
    *,
    start: str,
    end: str | None = None,
    api_key: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Fetch FRED series and return one column per logical feature name."""

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(FRED_SETUP_HINT)
    import json

    from sam.progress import progress_iter

    rows: dict[str, list] = {"date": []}
    for name, series_id in progress_iter(
        series_map.items(),
        desc="FRED series",
        total=len(series_map),
        enabled=show_progress,
        unit="series",
    ):
        params = (
            f"series_id={series_id}&api_key={key}&file_type=json"
            f"&observation_start={start}"
        )
        if end:
            params += f"&observation_end={end}"
        url = f"{FRED_API_BASE}?{params}"
        with urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        observations = payload.get("observations", [])
        dates = []
        values = []
        for item in observations:
            value = pd.to_numeric(item.get("value"), errors="coerce")
            if np.isfinite(value):
                dates.append(pd.Timestamp(item["date"]))
                values.append(float(value))
        series = pd.Series(values, index=pd.DatetimeIndex(dates), name=name).sort_index()
        if not rows["date"]:
            rows["date"] = list(series.index)
        aligned = series.reindex(pd.DatetimeIndex(rows["date"]))
        rows[name] = aligned.to_numpy()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_fred_from_csv(path: str | Path) -> pd.DataFrame:
    """Load pre-downloaded FRED features (columns: date + feature names)."""

    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def build_market_panel(
    *,
    prices: pd.DataFrame,
    vix: pd.DataFrame,
    fred: pd.DataFrame | None = None,
    calendar_symbol: str = "SPY",
) -> pd.DataFrame:
    """Align VIX and FRED onto the SPY trading calendar with forward-fill."""

    calendar = (
        prices.loc[prices["symbol"] == calendar_symbol, ["date"]]
        .drop_duplicates()
        .assign(date=lambda item: pd.to_datetime(item["date"]))
        .sort_values("date")
    )
    panel = calendar.copy()
    vix_aligned = (
        vix.assign(date=pd.to_datetime(vix["date"]))
        .sort_values("date")
        .drop_duplicates("date")
    )
    panel = panel.merge(vix_aligned, on="date", how="left")
    if fred is not None and not fred.empty:
        fred_aligned = (
            fred.assign(date=pd.to_datetime(fred["date"]))
            .sort_values("date")
            .drop_duplicates("date")
        )
        panel = panel.merge(fred_aligned, on="date", how="left")
    feature_cols = [column for column in panel.columns if column != "date"]
    panel[feature_cols] = panel[feature_cols].ffill()
    return panel.reset_index(drop=True)


def ingest_volatility_research_data(
    *,
    universe_path: str | Path = "configs/universes/volatility_regime.toml",
    run_config_path: str | Path = "configs/risk/volatility_regime_run.toml",
    cache_dir: str | Path | None = "data/research/volatility_regime",
    vix_csv: str | Path | None = None,
    fred_csv: str | Path | None = None,
    refresh: bool = False,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch or load cached prices, VIX, and FRED inputs."""

    from sam.volatility_config import load_volatility_run_config

    run_cfg = load_volatility_run_config(run_config_path)
    universe = load_universe(universe_path)
    cache = Path(cache_dir) if cache_dir else None
    prices_path = cache / "prices.parquet" if cache else None
    vix_path = cache / "vix.parquet" if cache else None
    fred_path = cache / "fred.parquet" if cache else None

    if prices_path is not None and prices_path.is_file() and not refresh:
        prices = pd.read_parquet(prices_path)
    else:
        prices = fetch_prices(
            universe.normalized_symbols,
            start=run_cfg.data_start,
            end=run_cfg.data_end,
            market=universe.market,
        )
        if prices_path is not None:
            prices_path.parent.mkdir(parents=True, exist_ok=True)
            prices.to_parquet(prices_path, index=False)

    if vix_path is not None and vix_path.is_file() and not refresh:
        vix = pd.read_parquet(vix_path)
    else:
        vix = fetch_vix_history(
            start=run_cfg.data_start,
            end=run_cfg.data_end,
            csv_path=vix_csv,
        )
        if vix_path is not None:
            vix.to_parquet(vix_path, index=False)

    if fred_csv is None and cache is not None:
        default_fred_csv = cache / "fred.csv"
        if default_fred_csv.is_file():
            fred_csv = default_fred_csv
            logger.info("loading FRED features from %s", default_fred_csv)

    if fred_csv is not None:
        fred = load_fred_from_csv(fred_csv)
        if fred_path is not None and refresh:
            fred.to_parquet(fred_path, index=False)
    elif fred_path is not None and fred_path.is_file() and not refresh:
        logger.info("loading cached FRED features from %s", fred_path)
        fred = pd.read_parquet(fred_path)
    elif run_cfg.fred_series:
        if not os.environ.get("FRED_API_KEY"):
            raise ValueError(FRED_SETUP_HINT)
        logger.info("fetching FRED series from API")
        fred = fetch_fred_series(
            run_cfg.fred_series,
            start=run_cfg.data_start,
            end=run_cfg.data_end,
            show_progress=show_progress,
        )
        if fred_path is not None:
            fred_path.parent.mkdir(parents=True, exist_ok=True)
            fred.to_parquet(fred_path, index=False)
            logger.info("cached FRED features to %s", fred_path)
    else:
        fred = pd.DataFrame(columns=["date"])

    panel = build_market_panel(prices=prices, vix=vix, fred=fred)
    return {"prices": prices, "vix": vix, "fred": fred, "panel": panel}


def load_volatility_regime_fixtures(
    fixture_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load frozen CSV fixtures for offline volatility-regime tests."""

    root = Path(fixture_dir) if fixture_dir is not None else DEFAULT_VOLATILITY_FIXTURE_DIR
    prices = pd.read_csv(root / "prices.csv")
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    vix = pd.read_csv(root / "vix.csv")
    vix["date"] = pd.to_datetime(vix["date"])
    fred = load_fred_from_csv(root / "fred.csv")
    panel = build_market_panel(prices=prices, vix=vix, fred=fred)
    return {"prices": prices, "vix": vix, "fred": fred, "panel": panel}


def synthetic_volatility_fixtures(
    *,
    n_days: int = 600,
    start: str = "2018-01-01",
) -> dict[str, pd.DataFrame]:
    """Build deterministic research fixtures for offline tests."""

    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for symbol in ETF_SYMBOLS:
        base = 100 + ETF_SYMBOLS.index(symbol) * 5
        noise = np.sin(np.arange(len(dates)) / 13 + ETF_SYMBOLS.index(symbol)) * 0.01
        prices = base * (1 + pd.Series(noise)).cumprod()
        for date, price in zip(dates, prices, strict=True):
            rows.append(
                {
                    "date": date.date(),
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
    prices = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    vix = pd.DataFrame(
        {
            "date": dates,
            "vix_close": 12 + 8 * np.sin(np.arange(len(dates)) / 40),
        }
    )
    fred = pd.DataFrame(
        {
            "date": dates,
            "treasury_10y_yield": 4.0 + 0.5 * np.sin(np.arange(len(dates)) / 100),
            "treasury_2y_yield": 3.5 + 0.4 * np.sin(np.arange(len(dates)) / 90),
            "yield_curve_10y_2y": 0.5 + 0.1 * np.sin(np.arange(len(dates)) / 80),
            "high_yield_oas": 4.0,
            "investment_grade_oas": 1.5,
            "fed_funds_rate": 2.0,
        }
    )
    panel = build_market_panel(prices=prices, vix=vix, fred=fred)
    return {"prices": prices, "vix": vix, "fred": fred, "panel": panel}


__all__ = [
    "DEFAULT_VOLATILITY_FIXTURE_DIR",
    "ETF_SYMBOLS",
    "VIX_HISTORY_URL",
    "build_market_panel",
    "fetch_fred_series",
    "fetch_vix_history",
    "ingest_volatility_research_data",
    "load_fred_from_csv",
    "load_volatility_regime_fixtures",
    "synthetic_volatility_fixtures",
]
