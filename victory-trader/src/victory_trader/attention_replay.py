"""Chronological replay and coverage audit for hierarchical attention."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .attention_runtime import (
    AttentionConfig,
    AttentionEvidence,
    AttentionRuntime,
    AttentionState,
    ObservationResolution,
)

MINUTE_MS = 60_000
SCAN_REQUIRED_COLUMNS = {
    "trading_day",
    "ticker",
    "t",
    "attention_score",
}
MARKET_BAR_REQUIRED_COLUMNS = {"trading_day", "ticker", "t", "c"}
PRIOR_REQUIRED_COLUMNS = {
    "trading_day",
    "ticker",
    "previous_close",
    "eligible",
}


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def build_market_scan_frame(
    minute_bars: pd.DataFrame,
    prior_close: pd.DataFrame,
    *,
    min_price: float = 0.50,
    max_price: float = 20.0,
    runner_threshold_pct: float = 10.0,
) -> pd.DataFrame:
    """Build a point-in-time broad-scan baseline from market-wide minute bars.

    The v0.1 score is the same-timestamp cross-sectional percentile of return
    from the nominal prior close. It deliberately uses no future outcome or
    completed-day field. This is an orchestration baseline, not a validated
    entry edge.
    """

    _require_columns(minute_bars, MARKET_BAR_REQUIRED_COLUMNS, "minute bars")
    _require_columns(prior_close, PRIOR_REQUIRED_COLUMNS, "prior close")
    if min_price <= 0 or max_price <= min_price:
        raise ValueError("price bounds must satisfy 0 < min_price < max_price")

    bars = minute_bars.copy()
    prior = prior_close.copy()
    for frame in (bars, prior):
        frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
        frame["trading_day"] = frame["trading_day"].astype(str)

    duplicate_prior = prior.duplicated(["trading_day", "ticker"], keep=False)
    if duplicate_prior.any():
        raise ValueError("prior close has duplicate ticker/day rows")
    duplicate_bars = bars.duplicated(
        ["trading_day", "ticker", "t"], keep=False
    )
    if duplicate_bars.any():
        raise ValueError("minute bars have duplicate ticker/timestamp rows")

    prior["previous_close"] = pd.to_numeric(
        prior["previous_close"], errors="coerce"
    )
    prior["eligible"] = prior["eligible"].fillna(False).astype(bool)
    prior = prior.loc[
        prior["eligible"]
        & prior["previous_close"].between(min_price, max_price, inclusive="both")
    ]

    frame = bars.merge(
        prior.loc[:, ["trading_day", "ticker", "previous_close"]],
        on=["trading_day", "ticker"],
        how="inner",
        validate="many_to_one",
    )
    frame["t"] = pd.to_numeric(frame["t"], errors="coerce")
    frame["c"] = pd.to_numeric(frame["c"], errors="coerce")
    valid = (
        frame["t"].notna()
        & frame["c"].gt(0)
        & frame["previous_close"].gt(0)
    )
    frame = frame.loc[valid].copy()
    frame["t"] = frame["t"].astype("int64")
    frame["return_from_previous_close_pct"] = (
        frame["c"] / frame["previous_close"] - 1.0
    ) * 100.0
    frame["attention_score"] = frame.groupby(
        ["trading_day", "t"], sort=False
    )["return_from_previous_close_pct"].rank(method="average", pct=True)

    above = frame["return_from_previous_close_pct"].ge(runner_threshold_pct)
    already_crossed = above.groupby(
        [frame["trading_day"], frame["ticker"]], sort=False
    ).transform(lambda values: values.shift(fill_value=False).cummax())
    frame["runner_cross_now"] = above & ~already_crossed
    return frame.sort_values(
        ["trading_day", "t", "ticker"], kind="stable"
    ).reset_index(drop=True)


def replay_attention(
    scan_frame: pd.DataFrame,
    config: AttentionConfig | None = None,
) -> pd.DataFrame:
    """Replay broad scans in strict session/timestamp order."""

    _require_columns(scan_frame, SCAN_REQUIRED_COLUMNS, "scan frame")
    duplicate = scan_frame.duplicated(
        ["trading_day", "ticker", "t"], keep=False
    )
    if duplicate.any():
        raise ValueError("scan frame has duplicate ticker/timestamp rows")

    frame = scan_frame.copy()
    frame["trading_day"] = frame["trading_day"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["t"] = pd.to_numeric(frame["t"], errors="raise").astype("int64")
    frame["attention_score"] = pd.to_numeric(
        frame["attention_score"], errors="raise"
    )
    if not np.isfinite(frame["attention_score"].to_numpy(dtype=float)).all():
        raise ValueError("scan frame attention scores must be finite")

    rows: list[dict[str, object]] = []
    for trading_day, day_frame in frame.groupby("trading_day", sort=True):
        runtime = AttentionRuntime(config)
        for timestamp, batch in day_frame.groupby("t", sort=True):
            def optional_bool(row: object, name: str, default: bool) -> bool:
                value = getattr(row, name, default)
                return default if pd.isna(value) else bool(value)

            evidence = [
                AttentionEvidence(
                    ticker=str(row.ticker),
                    attention_score=float(row.attention_score),
                    position_open=optional_bool(row, "position_open", False),
                    position_just_closed=optional_bool(
                        row, "position_just_closed", False
                    ),
                    data_valid=optional_bool(row, "data_valid", True),
                    eligible=optional_bool(row, "eligible", True),
                )
                for row in batch.itertuples(index=False)
            ]
            plans = runtime.step(int(timestamp), evidence)
            for plan in plans.values():
                rows.append(
                    {
                        "trading_day": str(trading_day),
                        "t": int(timestamp),
                        "ticker": plan.ticker,
                        "state": plan.state.value,
                        "resolution": plan.resolution.value,
                        "attention_score": plan.attention_score,
                        "attention_rank": plan.rank,
                        "reason": plan.reason,
                        "feeds": ",".join(plan.feeds),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["trading_day", "t", "ticker"], kind="stable"
    ).reset_index(drop=True)


def _runner_capture(
    trace: pd.DataFrame,
    scan_frame: pd.DataFrame,
    states: set[str],
) -> tuple[int, list[float]]:
    if "runner_cross_now" not in scan_frame.columns:
        return 0, []
    crossings = scan_frame.loc[
        scan_frame["runner_cross_now"].fillna(False).astype(bool),
        ["trading_day", "ticker", "t"],
    ]
    captured = 0
    leads: list[float] = []
    for row in crossings.itertuples(index=False):
        eligible = trace.loc[
            trace["trading_day"].eq(str(row.trading_day))
            & trace["ticker"].eq(str(row.ticker).upper())
            & trace["t"].le(int(row.t))
            & trace["state"].isin(states)
        ]
        if eligible.empty:
            continue
        captured += 1
        first_t = int(eligible["t"].min())
        leads.append((int(row.t) - first_t) / MINUTE_MS)
    return captured, leads


def summarize_attention_replay(
    trace: pd.DataFrame,
    scan_frame: pd.DataFrame,
) -> dict[str, int | float | None]:
    """Summarize coverage and feed demand without inspecting trade returns."""

    if trace.empty:
        return {
            "trace_rows": 0,
            "runner_episodes": 0,
        }
    crossings = (
        int(scan_frame["runner_cross_now"].fillna(False).astype(bool).sum())
        if "runner_cross_now" in scan_frame.columns
        else 0
    )
    watch_capture, watch_leads = _runner_capture(
        trace,
        scan_frame,
        {
            AttentionState.WATCH.value,
            AttentionState.HOT.value,
            AttentionState.POSITION.value,
        },
    )
    hot_capture, hot_leads = _runner_capture(
        trace,
        scan_frame,
        {AttentionState.HOT.value, AttentionState.POSITION.value},
    )

    occupancy = (
        trace.groupby(["trading_day", "t", "state"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    ordered = trace.sort_values(["trading_day", "ticker", "t"])
    previous = ordered.groupby(
        ["trading_day", "ticker"], sort=False
    )["state"].shift()
    state_changes = previous.notna() & ordered["state"].ne(previous)

    resolution = trace["resolution"]
    return {
        "trace_rows": len(trace),
        "sessions": int(trace["trading_day"].nunique()),
        "unique_symbols": int(trace["ticker"].nunique()),
        "runner_episodes": crossings,
        "runner_watch_or_better_by_cross": watch_capture,
        "runner_hot_or_position_by_cross": hot_capture,
        "runner_watch_or_better_capture_rate": (
            watch_capture / crossings if crossings else None
        ),
        "runner_hot_or_position_capture_rate": (
            hot_capture / crossings if crossings else None
        ),
        "median_watch_lead_minutes": (
            float(np.median(watch_leads)) if watch_leads else None
        ),
        "median_hot_lead_minutes": (
            float(np.median(hot_leads)) if hot_leads else None
        ),
        "max_watch_occupancy": int(
            occupancy.get(AttentionState.WATCH.value, pd.Series([0])).max()
        ),
        "max_hot_occupancy": int(
            occupancy.get(AttentionState.HOT.value, pd.Series([0])).max()
        ),
        "state_changes": int(state_changes.sum()),
        "grouped_minute_requests": int(
            resolution.eq(ObservationResolution.GROUPED_MINUTE.value).sum()
        ),
        "minute_bar_requests": int(
            resolution.eq(ObservationResolution.MINUTE_BARS.value).sum()
        ),
        "second_bar_requests": int(
            resolution.eq(ObservationResolution.SECOND_BARS.value).sum()
        ),
        "trade_nbbo_requests": int(
            resolution.eq(ObservationResolution.TRADES_NBBO.value).sum()
        ),
        "dropped_rows": int(resolution.eq(ObservationResolution.NONE.value).sum()),
    }
