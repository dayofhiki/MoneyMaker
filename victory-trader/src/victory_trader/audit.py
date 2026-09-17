from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .model_schema import MODEL_FEATURE_COLUMNS, validate_feature_allowlist


FORWARD_RETURN_RE = re.compile(r"^return_(\d+)m_pct$")
EXPECTED_HORIZONS = (1, 2, 5, 10, 15, 30, 60)
EXECUTION_SCENARIOS = ("light", "base", "stress")
EVENT_KEY = ("trading_day", "ticker", "timestamp_ms", "threshold_pct")
FORBIDDEN_LEGACY_COLUMNS = {
    "day_high",
    "day_close",
    "day_volume",
    "day_vwap",
    "day_dollar_volume",
    "day_high_return_pct",
    "market_cap",
    "shares_outstanding",
}


@dataclass(frozen=True)
class AuditResult:
    issues: tuple[str, ...]
    summary: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def render(self) -> str:
        lines = ["=== MoneyMaker Dataset Audit ==="]
        lines.extend(self.summary)
        if self.ok:
            lines.append("STATUS: PASS")
        else:
            lines.append("STATUS: FAIL")
            lines.extend(f"- {issue}" for issue in self.issues)
        return "\n".join(lines)


def audit_event_dataset(frame: pd.DataFrame, *, require_full_universe: bool = True) -> AuditResult:
    issues: list[str] = []
    summary: list[str] = []

    if frame.empty:
        return AuditResult(("dataset is empty",), ("rows=0",))

    required = set(EVENT_KEY) | {
        "signal_price",
        "entry_price",
        "entry_timestamp_ms",
        "entry_model",
        "entry_delay_minutes",
        "previous_close",
        "dataset_schema_version",
        "source_prices_adjusted",
        "discovery_high_return_threshold_pct",
    }
    missing_required = sorted(required - set(frame.columns))
    if missing_required:
        issues.append(f"missing required columns: {missing_required}")

    if all(column in frame.columns for column in EVENT_KEY):
        duplicate_count = int(frame.duplicated(list(EVENT_KEY)).sum())
        if duplicate_count:
            issues.append(f"duplicate event keys: {duplicate_count}")

    rows = len(frame)
    tickers = int(frame["ticker"].nunique()) if "ticker" in frame.columns else 0
    days = int(frame["trading_day"].nunique()) if "trading_day" in frame.columns else 0
    clusters = (
        int(frame[["trading_day", "ticker"]].drop_duplicates().shape[0])
        if {"trading_day", "ticker"}.issubset(frame.columns)
        else 0
    )
    summary.append(f"rows={rows} tickers={tickers} trading_days={days} ticker_day_clusters={clusters}")

    if "threshold_pct" in frame.columns:
        thresholds = sorted(pd.to_numeric(frame["threshold_pct"], errors="coerce").dropna().unique())
        summary.append(f"thresholds={','.join(str(float(value)) for value in thresholds)}")
        if thresholds and "discovery_high_return_threshold_pct" in frame.columns:
            discovery = pd.to_numeric(frame["discovery_high_return_threshold_pct"], errors="coerce")
            if discovery.isna().any():
                issues.append("missing discovery threshold provenance")
            elif float(discovery.max()) > float(min(thresholds)):
                issues.append(
                    "discovery threshold exceeds lowest studied event threshold and creates future selection bias"
                )

    if require_full_universe:
        if "debug_candidate_limit" not in frame.columns:
            issues.append("missing debug_candidate_limit provenance column")
        elif frame["debug_candidate_limit"].notna().any():
            limits = sorted(pd.to_numeric(frame["debug_candidate_limit"], errors="coerce").dropna().unique())
            issues.append(f"debug candidate limit present in research dataset: {limits}")

    legacy = sorted(FORBIDDEN_LEGACY_COLUMNS & set(frame.columns))
    if legacy:
        issues.append(f"future/leak-prone legacy columns present: {legacy}")

    if "source_prices_adjusted" in frame.columns:
        adjusted = frame["source_prices_adjusted"].fillna(True).astype(bool)
        if adjusted.any():
            issues.append("research universe contains split-adjusted source prices")

    if "entry_model" in frame.columns:
        bad_models = sorted(set(frame["entry_model"].dropna().astype(str)) - {"next_minute_open"})
        if bad_models:
            issues.append(f"unsupported primary entry models: {bad_models}")
    if "entry_delay_minutes" in frame.columns:
        delays = pd.to_numeric(frame["entry_delay_minutes"], errors="coerce")
        if delays.isna().any() or (delays != 0).any():
            issues.append("primary dataset entry_delay_minutes must be 0")

    for price_col in ("signal_price", "previous_close"):
        if price_col in frame.columns:
            prices = pd.to_numeric(frame[price_col], errors="coerce")
            invalid = int((prices <= 0).fillna(True).sum())
            if invalid:
                issues.append(f"invalid {price_col} rows: {invalid}")
    if "entry_price" in frame.columns:
        entries = pd.to_numeric(frame["entry_price"], errors="coerce")
        invalid = int((entries.notna() & (entries <= 0)).sum())
        if invalid:
            issues.append(f"invalid entry_price rows: {invalid}")
        summary.append(f"executable_entry_rate={float(entries.notna().mean()):.3f}")

    forward_columns = {
        int(match.group(1)): column
        for column in frame.columns
        if (match := FORWARD_RETURN_RE.match(column))
    }
    missing_horizons = [h for h in EXPECTED_HORIZONS if h not in forward_columns]
    if missing_horizons:
        issues.append(f"missing forward-return horizons: {missing_horizons}")

    entry_present = (
        pd.to_numeric(frame["entry_price"], errors="coerce").notna()
        if "entry_price" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    for horizon, gross_col in sorted(forward_columns.items()):
        gross_present = pd.to_numeric(frame[gross_col], errors="coerce").notna()
        impossible = int((gross_present & ~entry_present).sum())
        if impossible:
            issues.append(f"executable return exists without executable entry at {horizon}m: {impossible} rows")
        summary.append(f"return_{horizon}m_n={int(gross_present.sum())}")
        for scenario in EXECUTION_SCENARIOS:
            net_col = f"return_{horizon}m_{scenario}_net_return_pct"
            if net_col not in frame.columns:
                issues.append(f"missing execution-cost column: {net_col}")
                continue
            net_present = pd.to_numeric(frame[net_col], errors="coerce").notna()
            mismatch = int((gross_present != net_present).sum())
            if mismatch:
                issues.append(f"gross/net missingness mismatch at {horizon}m {scenario}: {mismatch} rows")

    for delay in (1, 2):
        if f"delay{delay}_entry_price" not in frame.columns:
            issues.append(f"missing latency sensitivity entry: delay{delay}_entry_price")
        if f"delay{delay}_return_5m_base_net_return_pct" not in frame.columns:
            issues.append(f"missing latency sensitivity endpoint: delay{delay}_return_5m_base_net_return_pct")

    try:
        validate_feature_allowlist()
    except ValueError as exc:
        issues.append(str(exc))
    missing_model_features = sorted(set(MODEL_FEATURE_COLUMNS) - set(frame.columns))
    if missing_model_features:
        issues.append(f"missing model allowlist columns: {missing_model_features}")

    if {"is_premarket", "is_regular_session", "is_after_hours"}.issubset(frame.columns):
        flags = frame[["is_premarket", "is_regular_session", "is_after_hours"]].fillna(False).astype(int)
        overlapping = int((flags.sum(axis=1) > 1).sum())
        if overlapping:
            issues.append(f"overlapping session labels: {overlapping} rows")

    if "halt_data_available" in frame.columns:
        available = frame["halt_data_available"].fillna(False).astype(bool)
        summary.append(f"halt_data_available_rate={float(available.mean()):.3f}")

    barrier_status_columns = [column for column in frame.columns if column.startswith("tp") and column.endswith("_status")]
    if barrier_status_columns:
        statuses = pd.concat([frame[column].astype("string") for column in barrier_status_columns], ignore_index=True)
        summary.append(f"barrier_ambiguous_rate={float((statuses == 'ambiguous').mean()):.3f}")
        summary.append(f"barrier_stop_gap_rate={float((statuses == 'stop_gap').mean()):.3f}")
        summary.append(f"barrier_unresolved_missing_rate={float((statuses == 'unresolved_missing').mean()):.3f}")

    return AuditResult(tuple(issues), tuple(summary))
