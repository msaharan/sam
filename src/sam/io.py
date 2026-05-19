"""Filesystem and tabular IO helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if target.suffix == ".parquet":
        return pd.read_parquet(target)
    if target.suffix == ".csv":
        return pd.read_csv(target)
    raise ValueError(f"Unsupported table format: {target}")


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    if target.suffix == ".parquet":
        frame.to_parquet(target, index=False)
    elif target.suffix == ".csv":
        frame.to_csv(target, index=False)
    else:
        raise ValueError(f"Unsupported table format: {target}")
    return target


def write_partitioned_table(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    partition_cols: list[str],
) -> Path:
    """Write a Parquet dataset directory partitioned by stable research keys."""

    target = ensure_dir(path)
    missing = sorted(set(partition_cols) - set(frame.columns))
    if missing:
        raise ValueError(f"partition columns missing from frame: {missing}")
    frame.to_parquet(target, index=False, partition_cols=partition_cols)
    return target


def write_manifest(out_dir: str | Path, *, command: str, config: dict[str, Any]) -> Path:
    target = ensure_dir(out_dir) / "manifest.json"
    payload = {
        "command": command,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
    }
    target.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True), encoding="utf-8")
    return target


def query_parquet(sql: str, data_dir: str | Path = "data") -> pd.DataFrame:
    root = Path(data_dir)
    parquet_path = str(root / "**" / "*.parquet") if root.is_dir() else str(root)
    parquet_literal = parquet_path.replace("'", "''")
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET enable_object_cache=true")
    connection.execute(
        f"CREATE OR REPLACE VIEW parquet_data AS SELECT * FROM read_parquet('{parquet_literal}')"
    )
    return connection.execute(sql).fetch_df()
