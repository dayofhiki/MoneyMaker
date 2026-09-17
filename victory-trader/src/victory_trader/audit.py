from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


FORWARD_RETURN_RE = re.compile(r"^return_(\d+)m_pct$")
EXPECTED_HORIZONS = (1, 2, 5, 10, 15, 30, 60)
EXECUTION_SCENARIOS = ("light", "base", "stress")
EVENT_KEY = ("trading_day", "ticker", "timestamp_ms", "threshold_pct")


@dataclass(frozen=True)
class AuditResult:
    issues: tuple[str, ...]
    summary: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def render(self) -> str:
        lines = ["=== Victory Trader Dataset Audit ==="]
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

    required = set(EVENT_KEY) | {"entry_price", "previous_close"}
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
    summary.append(f"rows={rows} tickers={tickers} trading_days={days}")

    if "threshold_pct" in frame.columns:
        thresholds = sorted(pd.to_numeric(frame["threshold_pct"], errors="coerce").dropna().unique())
        summary.append(f"thresholds={','.join(str(float(value)) for value in thresholds)}")

    if require_full_universe:
        if "debug_candidate_limit" not in frame.columns:
            issues.append("missing debug_candidate_limit provenance column")
        elif frame["debug_candidate_limit"].notna().any():
            limits = sorted(
                pd.to_numeric(frame["debug_candidate_limit"], errors="coerce").dropna().unique()
            )
            issues.append(f"debug candidate limit present in research dataset: {limits}")

    forward_columns = {
        int(match.group(1)): column
        for column in frame.columns
        if (match := FORWARD_RETURN_RE.match(column))
    }
    missing_horizons = [h for h in EXPECTED_HORIZONS if h not in forward_columns]
    if missing_horizons:
        issues.append(f"missing forward-return horizons: {missing_horizons}")

    for horizon, gross_col in sorted(forward_columns.items()):
        gross_present = pd.to_numeric(frame[gross_col], errors="coerce").notna()
        summary.append(f"return_{horizon}m_n={int(gross_present.sum())}")
        for scenario in EXECUTION_SCENARIOS:
            net_col = f"return_{horizon}m_{scenario}_net_return_pct"
            if net_col not in frame.columns:
                issues.append(f"missing execution-cost column: {net_col}")
                continue
            net_present = pd.to_numeric(frame[net_col], errors="coerce").notna()
            mismatch = int((gross_present != net_present).sum())
            if mismatch:
                issues.append(
                    f"gross/net missingness mismatch at {horizon}m {scenario}: {mismatch} rows"
                )

    trailing_expected = {
        "trailing_return_5m_pct",
        "trailing_return_15m_pct",
        "trailing_return_30m_pct",
    }
    missing_trailing = sorted(trailing_expected - set(frame.columns))
    if missing_trailing:
        issues.append(f"missing trailing-return feature columns: {missing_trailing}")

    for price_col in ("entry_price", "previous_close"):
        if price_col in frame.columns:
            prices = pd.to_numeric(frame[price_col], errors="coerce")
            invalid = int((prices <= 0).fillna(True).sum())
            if invalid:
                issues.append(f"invalid {price_col} rows: {invalid}")

    if "halt_data_available" in frame.columns:
        available = frame["halt_data_available"].fillna(False).astype(bool)
        summary.append(f"halt_data_available_rate={float(available.mean()):.3f}")

    return AuditResult(tuple(issues), tuple(summary))
