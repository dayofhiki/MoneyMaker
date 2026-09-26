"""Strict bridge from provider one-second aggregates to causal replay bars."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

import pandas as pd

from .causal_second_execution import SECOND_MS, SecondBar


@dataclass(frozen=True)
class HaltInterval:
    start_t: int  # UTC milliseconds, inclusive
    resume_t: int | None  # exclusive; None means no observed resume in session


@dataclass(frozen=True)
class AdaptedSecondPath:
    bars: tuple[SecondBar, ...]
    provider_rows: int
    session_rows: int
    halt_overlap_rows: int


def adapt_second_bars(
    frame: pd.DataFrame,
    *,
    session_open_t: int,
    session_close_t: int,
    halts: Sequence[HaltInterval] = (),
) -> AdaptedSecondPath:
    """Keep only regular-session rows and reject ambiguous provider duplicates.

    Provider `t` is the start of a one-second aggregate. A bar overlapping any
    portion of a supplied halt interval is unusable for fills or triggers.
    Missing seconds remain missing. Unknown halt coverage must be reported by
    the caller; an empty `halts` sequence is not evidence that no halt occurred.
    """
    if session_open_t < 0 or session_close_t <= session_open_t:
        raise ValueError("invalid session boundaries")
    required = {"t", "o", "h", "l", "c"}
    if not required.issubset(frame.columns):
        raise ValueError(f"second path missing columns: {sorted(required - set(frame.columns))}")
    ordered_halts = sorted(halts, key=lambda item: item.start_t)
    previous_end = -1
    for interval in ordered_halts:
        end = interval.resume_t if interval.resume_t is not None else session_close_t
        if interval.start_t < 0 or end <= interval.start_t or interval.start_t < previous_end:
            raise ValueError("halt intervals must be positive and non-overlapping")
        previous_end = end

    rows: list[tuple[int, float, float, float, float]] = []
    for raw in frame.loc[:, ["t", "o", "h", "l", "c"]].itertuples(index=False, name=None):
        timestamp = raw[0]
        if pd.isna(timestamp) or not isfinite(float(timestamp)):
            raise ValueError("second timestamp must be finite")
        t = int(timestamp)
        if float(timestamp) != t or t % SECOND_MS:
            raise ValueError("second timestamp must be whole-second UTC milliseconds")
        if not session_open_t <= t < session_close_t:
            continue
        prices = tuple(float(value) for value in raw[1:])
        rows.append((t, *prices))

    rows.sort(key=lambda item: item[0])
    if any(a[0] == b[0] for a, b in pairwise(rows)):
        raise ValueError("duplicate second timestamp; resolve provider conflict first")
    bars: list[SecondBar] = []
    halted_count = 0
    for t, open_price, high, low, close in rows:
        halted = any(
            interval.start_t < t + SECOND_MS
            and t < (interval.resume_t if interval.resume_t is not None else session_close_t)
            for interval in ordered_halts
        )
        halted_count += halted
        if not halted:
            if any(not isfinite(x) or x <= 0 for x in (open_price, high, low, close)):
                raise ValueError("active second bar has invalid price")
            if low > min(open_price, close) or high < max(open_price, close):
                raise ValueError("active second bar has inconsistent OHLC")
        bars.append(SecondBar(t, open_price, high, low, close, halted))
    return AdaptedSecondPath(tuple(bars), len(frame), len(rows), halted_count)

