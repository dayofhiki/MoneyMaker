from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_BAR_COLUMNS = ("t", "o", "h", "l", "c", "v")
OPTIONAL_BAR_COLUMNS = ("vw", "n")


def bars_from_massive_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Normalize a Massive aggregate response into a chronological DataFrame.

    Required columns are t/o/h/l/c/v. Massive also exposes optional per-bar VWAP
    (`vw`) and transaction count (`n`); preserve those when present so feature
    engineering does not need another API request.
    """
    results = payload.get("results") or []
    if not results:
        return pd.DataFrame(columns=REQUIRED_BAR_COLUMNS + OPTIONAL_BAR_COLUMNS)

    frame = pd.DataFrame(results)
    missing = set(REQUIRED_BAR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Massive payload missing bar fields: {sorted(missing)}")

    selected = list(REQUIRED_BAR_COLUMNS) + [
        col for col in OPTIONAL_BAR_COLUMNS if col in frame.columns
    ]
    frame = frame.loc[:, selected].copy()
    for col in ("o", "h", "l", "c", "v", "vw", "n"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce" if col in OPTIONAL_BAR_COLUMNS else "raise")
    frame["t"] = pd.to_numeric(frame["t"], errors="raise").astype("int64")
    return frame.sort_values("t").drop_duplicates(subset=["t"], keep="last").reset_index(drop=True)
