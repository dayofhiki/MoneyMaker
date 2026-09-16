from __future__ import annotations

from typing import Any

import pandas as pd


MASSIVE_BAR_COLUMNS = ("t", "o", "h", "l", "c", "v")


def bars_from_massive_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Normalize a Massive aggregate response into a chronological DataFrame.

    The returned frame keeps Massive's compact column names because they are
    convenient for high-volume research: t/o/h/l/c/v. `t` is Unix milliseconds.
    """
    results = payload.get("results") or []
    if not results:
        return pd.DataFrame(columns=MASSIVE_BAR_COLUMNS)

    frame = pd.DataFrame(results)
    missing = set(MASSIVE_BAR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Massive payload missing bar fields: {sorted(missing)}")

    frame = frame.loc[:, MASSIVE_BAR_COLUMNS].copy()
    for col in ("o", "h", "l", "c", "v"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame["t"] = pd.to_numeric(frame["t"], errors="raise").astype("int64")
    return frame.sort_values("t").drop_duplicates(subset=["t"], keep="last").reset_index(drop=True)
