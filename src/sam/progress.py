"""Lightweight tqdm helpers for long-running SAM workflows."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    enabled: bool = True,
    unit: str = "it",
    leave: bool = False,
) -> Iterator[T]:
    """Iterate with an optional tqdm bar."""

    if not enabled:
        yield from iterable
        return
    from tqdm import tqdm

    yield from tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


def progress_range(
    stop: int,
    *,
    desc: str,
    enabled: bool = True,
    unit: str = "it",
    leave: bool = False,
) -> Iterator[int]:
    """Range with an optional tqdm bar."""

    if not enabled:
        return range(stop)
    from tqdm import tqdm

    return tqdm(range(stop), desc=desc, unit=unit, leave=leave)


@contextmanager
def progress_task(desc: str, *, enabled: bool = True):
    """Context manager for a single long step (no sub-iterations)."""

    if not enabled:
        logger.info("%s", desc)
        yield
        return
    from tqdm import tqdm

    with tqdm(total=1, desc=desc, bar_format="{desc} ...", leave=False) as bar:
        yield
        bar.update(1)


def progress_write(message: str, *, enabled: bool = True) -> None:
    """Print a status line without breaking active tqdm bars."""

    if enabled:
        from tqdm import tqdm

        tqdm.write(message)
    else:
        logger.info("%s", message)
