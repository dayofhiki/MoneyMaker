from __future__ import annotations

import pandas as pd


# Features are deliberately opt-in. New dataset columns never become model inputs
# merely because they exist in the same research table.
MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "threshold_pct",
    "signal_price",
    "previous_close",
    "event_volume",
    "cumulative_volume",
    "volume_5m",
    "volume_15m",
    "volume_30m",
    "volume_accel_1m_vs_prior20m",
    "volume_accel_5m_vs_prior20m",
    "rvol_cumulative_20d",
    "rvol_5m_20d",
    "rvol_history_days",
    "session_vwap",
    "vwap_distance_pct",
    "regular_vwap",
    "regular_vwap_distance_pct",
    "hod_distance_pct",
    "trailing_return_5m_pct",
    "trailing_return_15m_pct",
    "trailing_return_30m_pct",
    "volatility_5m_pct",
    "volatility_15m_pct",
    "volatility_30m_pct",
    "premarket_return_pct",
    "premarket_volume",
    "minutes_from_regular_open",
    "is_premarket",
    "is_regular_session",
    "is_after_hours",
    "security_type",
    "primary_exchange",
)


FORBIDDEN_MODEL_EXACT = {
    "trading_day",
    "ticker",
    "entry_price",
    "entry_timestamp_ms",
    "mfe_pct",
    "mae_pct",
    "day_high",
    "day_close",
    "day_volume",
    "day_vwap",
    "day_dollar_volume",
    "day_high_return_pct",
    "market_cap",
    "shares_outstanding",
    "halt_within_5m",
    "halt_within_15m",
    "halt_within_30m",
    "halt_within_60m",
    "first_halt_minutes_after_event",
    "first_halt_reason_code",
    "first_halt_resume_minutes",
    "first_halt_is_volatility_pause",
}

FORBIDDEN_MODEL_PREFIXES = (
    "return_",
    "signal_return_",
    "delay1_",
    "delay2_",
    "tp",
)


def validate_feature_allowlist(columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS) -> None:
    duplicates = {column for column in columns if columns.count(column) > 1}
    if duplicates:
        raise ValueError(f"duplicate model feature columns: {sorted(duplicates)}")
    forbidden = [
        column
        for column in columns
        if column in FORBIDDEN_MODEL_EXACT
        or any(column.startswith(prefix) for prefix in FORBIDDEN_MODEL_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"future/outcome columns present in model feature allowlist: {sorted(forbidden)}")


def build_model_frame(
    frame: pd.DataFrame,
    target_column: str = "return_5m_base_net_return_pct",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return an explicit point-in-time X/y pair for downstream ML.

    This function is the only supported bridge from the rich research table into
    model training. Outcome, discovery, halt-after-event, identity/memorization,
    and execution columns are not inherited implicitly.
    """
    validate_feature_allowlist()
    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"dataset missing model feature columns: {missing}")
    if target_column not in frame.columns:
        raise ValueError(f"dataset missing target column: {target_column}")

    target = pd.to_numeric(frame[target_column], errors="coerce")
    usable = target.notna()
    return frame.loc[usable, list(MODEL_FEATURE_COLUMNS)].copy(), target.loc[usable].copy()
