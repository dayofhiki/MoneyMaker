from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .execution_costs import (
    DEFAULT_EXECUTION_SCENARIOS,
    modeled_buy_fill,
    modeled_sell_fill,
)
from .expanded_opportunity_shared_q import (
    predict_opportunity,
    train_opportunity_model,
)
from .expanded_path_aware_continuation import (
    HOLD_THRESHOLD,
    KEYS,
    MAX_HOLD_MINUTES,
    MINUTE_MS,
    build_continuation_rows,
)
from .expanded_path_transition_continuation import (
    predict_transition_continuation,
    train_transition_model,
)
from .expanded_supply_hurdle_ev import load_fold

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261044
INITIAL_CASH = 10_000.0
ORDER_BUDGET = 1_000.0
SEVERE_LOSS_PCT = -5.0


def _positive(value: object) -> bool:
    return bool(
        pd.notna(value)
        and np.isfinite(float(value))
        and float(value) > 0
    )


def _month_of(frame: pd.DataFrame) -> pd.Series:
    return frame["trading_day"].astype(str).str.slice(0, 7)


def _normalize_state(state: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["t", "o", "c"])
    if not required.issubset(state.columns):
        raise ValueError("state panel requires trading_day/ticker/t/o/c")
    result = state.copy()
    result["trading_day"] = result["trading_day"].astype(str)
    result["ticker"] = result["ticker"].astype(str)
    result["t"] = pd.to_numeric(result["t"], errors="coerce")
    if result.duplicated(KEYS + ["t"]).any():
        raise ValueError("state panel must be unique by ticker-day/t")
    return result.sort_values(KEYS + ["t"], kind="stable").reset_index(drop=True)


def _group_state(state: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    return {
        (str(day), str(ticker)): group.reset_index(drop=True)
        for (day, ticker), group in state.groupby(KEYS, sort=False)
    }


def _exact_open(group: pd.DataFrame | None, timestamp: int) -> float:
    if group is None or group.empty:
        return np.nan
    rows = group.loc[pd.to_numeric(group["t"], errors="coerce").eq(timestamp)]
    if rows.empty:
        return np.nan
    value = pd.to_numeric(rows.iloc[0]["o"], errors="coerce")
    return float(value) if _positive(value) else np.nan


def _first_open_at_or_after(
    group: pd.DataFrame | None,
    timestamp: int,
) -> tuple[float, float]:
    if group is None or group.empty:
        return np.nan, np.nan
    t = pd.to_numeric(group["t"], errors="coerce")
    o = pd.to_numeric(group["o"], errors="coerce")
    mask = t.ge(timestamp) & o.gt(0)
    rows = group.loc[mask].copy()
    if rows.empty:
        return np.nan, np.nan
    row = rows.sort_values("t", kind="stable").iloc[0]
    return float(row["t"]), float(row["o"])


def _add_returns(record: dict[str, object]) -> None:
    entry = record.get("entry_price", np.nan)
    exit_price = record.get("exit_price", np.nan)
    if not (_positive(entry) and _positive(exit_price)):
        record["gross_return_pct"] = np.nan
        for scenario in DEFAULT_EXECUTION_SCENARIOS:
            record[f"{scenario.name}_net_return_pct"] = np.nan
        return

    entry = float(entry)
    exit_price = float(exit_price)
    record["gross_return_pct"] = (exit_price / entry - 1.0) * 100.0
    for scenario in DEFAULT_EXECUTION_SCENARIOS:
        buy = modeled_buy_fill(entry, scenario)
        sell = modeled_sell_fill(exit_price, scenario)
        sell *= 1.0 - scenario.sell_fee_bps / 10_000.0
        record[f"{scenario.name}_net_return_pct"] = (
            sell / buy - 1.0
        ) * 100.0


def build_recurrent_trajectories(
    evaluation_anchors: pd.DataFrame,
    scored_rows: pd.DataFrame,
    state: pd.DataFrame,
    month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compose the frozen v3.7 one-step signal into one reached path per entry."""
    anchors = evaluation_anchors.copy()
    anchors["trading_day"] = anchors["trading_day"].astype(str)
    anchors["ticker"] = anchors["ticker"].astype(str)
    anchors["t"] = pd.to_numeric(anchors["t"], errors="coerce")
    anchors = anchors.loc[
        pd.to_numeric(
            anchors["opportunity_probability"],
            errors="coerce",
        ).gt(0.5)
    ].copy()
    if anchors.duplicated(KEYS).any():
        raise ValueError("evaluation anchors must be unique per ticker-day")

    state = _normalize_state(state)
    state_groups = _group_state(state)

    rows = scored_rows.copy()
    rows["trading_day"] = rows["trading_day"].astype(str)
    rows["ticker"] = rows["ticker"].astype(str)
    rows["t"] = pd.to_numeric(rows["t"], errors="coerce")
    row_groups = {
        (str(day), str(ticker)): group.sort_values(
            "minutes_held",
            kind="stable",
        ).reset_index(drop=True)
        for (day, ticker), group in rows.groupby(KEYS, sort=False)
    }

    trajectories: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    for anchor in anchors.sort_values(
        ["trading_day", "t", "ticker"],
        kind="stable",
    ).to_dict("records"):
        day = str(anchor["trading_day"])
        ticker = str(anchor["ticker"])
        key = (day, ticker)
        anchor_t = int(anchor["t"])
        entry_t = anchor_t + MINUTE_MS
        state_group = state_groups.get(key)
        entry_price = _exact_open(state_group, entry_t)
        record: dict[str, object] = {
            "month": month,
            "trading_day": day,
            "ticker": ticker,
            "anchor_t": anchor_t,
            "opportunity_probability": float(
                anchor["opportunity_probability"]
            ),
            "entry_t": entry_t,
            "entry_price": entry_price,
            "status": "missing_entry",
            "exit_reason": "missing_exact_entry_open",
            "exit_t": np.nan,
            "exit_price": np.nan,
            "holding_minutes": np.nan,
            "hold_decisions": 0,
            "decision_count": 0,
        }
        if not _positive(entry_price):
            _add_returns(record)
            trajectories.append(record)
            continue

        group = row_groups.get(key, pd.DataFrame())
        by_minute = {}
        if not group.empty:
            for row in group.to_dict("records"):
                held = pd.to_numeric(
                    pd.Series([row.get("minutes_held")]),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(held) and float(held).is_integer():
                    by_minute[int(held)] = row

        exit_t = np.nan
        exit_price = np.nan
        exit_reason = "unresolved"
        hold_decisions = 0
        decision_count = 0

        for minute in range(1, MAX_HOLD_MINUTES):
            expected_row_t = anchor_t + minute * MINUTE_MS
            decision_open_t = expected_row_t + MINUTE_MS
            row = by_minute.get(minute)

            if row is None or int(float(row["t"])) != expected_row_t:
                later_t, later_open = _first_open_at_or_after(
                    state_group,
                    decision_open_t,
                )
                decisions.append(
                    {
                        "month": month,
                        "trading_day": day,
                        "ticker": ticker,
                        "minutes_held": minute,
                        "decision_t": decision_open_t,
                        "hold_probability": np.nan,
                        "action": "EXIT_PENDING",
                        "reason": "missing_exact_decision_state",
                    }
                )
                if _positive(later_open):
                    exit_t = later_t
                    exit_price = later_open
                    exit_reason = (
                        "gap_exit_exact_open"
                        if int(later_t) == decision_open_t
                        else "gap_exit_delayed_open"
                    )
                else:
                    exit_reason = "unresolved_after_missing_state"
                break

            current_open = pd.to_numeric(
                pd.Series([row.get("exit_open")]),
                errors="coerce",
            ).iloc[0]
            probability = pd.to_numeric(
                pd.Series([row.get("hold_probability")]),
                errors="coerce",
            ).iloc[0]

            if not _positive(current_open):
                later_t, later_open = _first_open_at_or_after(
                    state_group,
                    decision_open_t,
                )
                decisions.append(
                    {
                        "month": month,
                        "trading_day": day,
                        "ticker": ticker,
                        "minutes_held": minute,
                        "decision_t": decision_open_t,
                        "hold_probability": probability,
                        "action": "EXIT_PENDING",
                        "reason": "missing_exact_decision_open",
                    }
                )
                if _positive(later_open):
                    exit_t = later_t
                    exit_price = later_open
                    exit_reason = "gap_exit_delayed_open"
                else:
                    exit_reason = "unresolved_after_missing_open"
                break

            decision_count += 1
            if not pd.notna(probability) or not np.isfinite(float(probability)):
                exit_t = decision_open_t
                exit_price = float(current_open)
                exit_reason = "model_score_missing_exit"
                decisions.append(
                    {
                        "month": month,
                        "trading_day": day,
                        "ticker": ticker,
                        "minutes_held": minute,
                        "decision_t": decision_open_t,
                        "hold_probability": probability,
                        "action": "EXIT",
                        "reason": exit_reason,
                    }
                )
                break

            if float(probability) <= HOLD_THRESHOLD:
                exit_t = decision_open_t
                exit_price = float(current_open)
                exit_reason = "classifier_exit"
                decisions.append(
                    {
                        "month": month,
                        "trading_day": day,
                        "ticker": ticker,
                        "minutes_held": minute,
                        "decision_t": decision_open_t,
                        "hold_probability": float(probability),
                        "action": "EXIT",
                        "reason": exit_reason,
                    }
                )
                break

            hold_decisions += 1
            decisions.append(
                {
                    "month": month,
                    "trading_day": day,
                    "ticker": ticker,
                    "minutes_held": minute,
                    "decision_t": decision_open_t,
                    "hold_probability": float(probability),
                    "action": "HOLD",
                    "reason": (
                        "classifier_hold"
                        if minute < MAX_HOLD_MINUTES - 1
                        else "classifier_hold_to_forced_cap"
                    ),
                }
            )

            if minute == MAX_HOLD_MINUTES - 1:
                forced_t = entry_t + MAX_HOLD_MINUTES * MINUTE_MS
                exact_forced = _exact_open(state_group, forced_t)
                if _positive(exact_forced):
                    exit_t = forced_t
                    exit_price = exact_forced
                    exit_reason = "forced_30m_exact_open"
                else:
                    later_t, later_open = _first_open_at_or_after(
                        state_group,
                        forced_t,
                    )
                    if _positive(later_open):
                        exit_t = later_t
                        exit_price = later_open
                        exit_reason = "forced_30m_delayed_open"
                    else:
                        exit_reason = "unresolved_after_forced_cap"
                break

        record.update(
            status="completed" if _positive(exit_price) else "unresolved",
            exit_reason=exit_reason,
            exit_t=exit_t,
            exit_price=exit_price,
            holding_minutes=(
                (float(exit_t) - entry_t) / MINUTE_MS
                if _positive(exit_t)
                else np.nan
            ),
            hold_decisions=hold_decisions,
            decision_count=decision_count,
        )
        _add_returns(record)
        trajectories.append(record)

    return pd.DataFrame(trajectories), pd.DataFrame(decisions)


def build_always_hold_comparator(
    recurrent: pd.DataFrame,
    state: pd.DataFrame,
    month: str,
) -> pd.DataFrame:
    state = _normalize_state(state)
    groups = _group_state(state)
    records: list[dict[str, object]] = []

    for source in recurrent.to_dict("records"):
        record = {
            "month": month,
            "trading_day": str(source["trading_day"]),
            "ticker": str(source["ticker"]),
            "anchor_t": source["anchor_t"],
            "opportunity_probability": source["opportunity_probability"],
            "entry_t": source["entry_t"],
            "entry_price": source["entry_price"],
            "status": "missing_entry",
            "exit_reason": "missing_exact_entry_open",
            "exit_t": np.nan,
            "exit_price": np.nan,
            "holding_minutes": np.nan,
            "hold_decisions": MAX_HOLD_MINUTES,
            "decision_count": 0,
        }
        if not _positive(source.get("entry_price")):
            _add_returns(record)
            records.append(record)
            continue

        target = int(source["entry_t"]) + MAX_HOLD_MINUTES * MINUTE_MS
        group = groups.get(
            (str(source["trading_day"]), str(source["ticker"]))
        )
        exact = _exact_open(group, target)
        if _positive(exact):
            exit_t, exit_price = float(target), exact
            reason = "always_hold_30m_exact_open"
        else:
            exit_t, exit_price = _first_open_at_or_after(group, target)
            reason = (
                "always_hold_30m_delayed_open"
                if _positive(exit_price)
                else "unresolved_after_30m"
            )

        record.update(
            status="completed" if _positive(exit_price) else "unresolved",
            exit_reason=reason,
            exit_t=exit_t,
            exit_price=exit_price,
            holding_minutes=(
                (float(exit_t) - float(source["entry_t"])) / MINUTE_MS
                if _positive(exit_t)
                else np.nan
            ),
        )
        _add_returns(record)
        records.append(record)

    return pd.DataFrame(records)


def _daily_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    daily = pd.DataFrame(
        {
            "day": frame["trading_day"].astype(str),
            "value": values,
        }
    ).dropna().groupby("day")["value"].mean()
    return float(daily.mean()) if len(daily) else np.nan


def _bootstrap_daily(
    frame: pd.DataFrame,
    column: str,
) -> tuple[int, float, float, float]:
    values = pd.to_numeric(frame[column], errors="coerce")
    daily = pd.DataFrame(
        {
            "day": frame["trading_day"].astype(str),
            "value": values,
        }
    ).dropna().groupby("day")["value"].mean()
    if len(daily) < 2:
        return int(len(daily)), float(daily.mean()) if len(daily) else np.nan, np.nan, np.nan
    arr = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = arr[
        rng.integers(
            0,
            len(arr),
            size=(BOOTSTRAP_SAMPLES, len(arr)),
        )
    ].mean(axis=1)
    return (
        int(len(arr)),
        float(arr.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def trade_summary(
    recurrent: pd.DataFrame,
    comparator: pd.DataFrame,
    month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_entries = recurrent.loc[
        pd.to_numeric(recurrent["entry_price"], errors="coerce").gt(0)
    ].copy()
    completed = recurrent.loc[recurrent["status"].eq("completed")].copy()
    comp_completed = comparator.loc[
        comparator["status"].eq("completed")
    ].copy()

    row: dict[str, object] = {
        "month": month,
        "gated_attempts": int(len(recurrent)),
        "valid_entries": int(len(valid_entries)),
        "completed_trades": int(len(completed)),
        "unresolved_trades": int(
            valid_entries["status"].eq("unresolved").sum()
        ),
        "completion_coverage": (
            float(len(completed) / len(valid_entries))
            if len(valid_entries)
            else np.nan
        ),
    }
    for column in (
        "gross_return_pct",
        "light_net_return_pct",
        "base_net_return_pct",
        "stress_net_return_pct",
    ):
        row[f"recurrent_{column}_mean"] = (
            float(pd.to_numeric(completed[column], errors="coerce").mean())
            if len(completed)
            else np.nan
        )
        row[f"comparator_{column}_mean"] = (
            float(
                pd.to_numeric(
                    comp_completed[column],
                    errors="coerce",
                ).mean()
            )
            if len(comp_completed)
            else np.nan
        )

    base = pd.to_numeric(
        completed["base_net_return_pct"],
        errors="coerce",
    ).dropna()
    row["recurrent_day_balanced_base_mean_pct"] = _daily_mean(
        completed,
        "base_net_return_pct",
    )
    row["comparator_day_balanced_base_mean_pct"] = _daily_mean(
        comp_completed,
        "base_net_return_pct",
    )
    row["recurrent_base_median_pct"] = (
        float(base.median()) if len(base) else np.nan
    )
    row["recurrent_base_p05_pct"] = (
        float(base.quantile(0.05)) if len(base) else np.nan
    )
    row["recurrent_base_positive_rate"] = (
        float(base.gt(0).mean()) if len(base) else np.nan
    )
    row["recurrent_base_severe_loss_rate"] = (
        float(base.le(SEVERE_LOSS_PCT).mean()) if len(base) else np.nan
    )

    holds = pd.to_numeric(
        completed["holding_minutes"],
        errors="coerce",
    ).dropna()
    row["holding_mean_min"] = float(holds.mean()) if len(holds) else np.nan
    row["holding_median_min"] = (
        float(holds.median()) if len(holds) else np.nan
    )
    row["holding_p10_min"] = (
        float(holds.quantile(0.10)) if len(holds) else np.nan
    )
    row["holding_p90_min"] = (
        float(holds.quantile(0.90)) if len(holds) else np.nan
    )
    row["holding_max_min"] = float(holds.max()) if len(holds) else np.nan
    row["mean_hold_decisions"] = (
        float(
            pd.to_numeric(
                completed["hold_decisions"],
                errors="coerce",
            ).mean()
        )
        if len(completed)
        else np.nan
    )

    match_keys = ["trading_day", "ticker", "anchor_t"]
    left = completed.loc[
        :,
        match_keys + ["base_net_return_pct"],
    ].rename(
        columns={
            "base_net_return_pct": "recurrent_base_net_return_pct"
        }
    )
    right = comp_completed.loc[
        :,
        match_keys + ["base_net_return_pct"],
    ].rename(
        columns={
            "base_net_return_pct": "comparator_base_net_return_pct"
        }
    )
    matched = left.merge(
        right,
        on=match_keys,
        how="inner",
        validate="one_to_one",
    )
    matched["base_difference_pct"] = (
        pd.to_numeric(
            matched["recurrent_base_net_return_pct"],
            errors="coerce",
        )
        - pd.to_numeric(
            matched["comparator_base_net_return_pct"],
            errors="coerce",
        )
    )
    row["matched_trades"] = int(len(matched))
    row["matched_base_difference_mean_pct"] = (
        float(matched["base_difference_pct"].mean())
        if len(matched)
        else np.nan
    )
    row["matched_day_balanced_difference_pct"] = _daily_mean(
        matched,
        "base_difference_pct",
    )

    days, mean, low, high = _bootstrap_daily(
        completed,
        "base_net_return_pct",
    )
    diff_days, diff_mean, diff_low, diff_high = _bootstrap_daily(
        matched,
        "base_difference_pct",
    )
    bootstrap = pd.DataFrame(
        [
            {
                "month": month,
                "recurrent_days": days,
                "recurrent_day_mean_pct": mean,
                "recurrent_ci_low_pct": low,
                "recurrent_ci_high_pct": high,
                "difference_days": diff_days,
                "difference_day_mean_pct": diff_mean,
                "difference_ci_low_pct": diff_low,
                "difference_ci_high_pct": diff_high,
            }
        ]
    )
    return pd.DataFrame([row]), bootstrap


def account_replay(
    trajectories: pd.DataFrame,
    state: pd.DataFrame,
    policy: str,
    month: str,
) -> pd.DataFrame:
    state = _normalize_state(state)
    state_groups = _group_state(state)
    rows = trajectories.reset_index(drop=True).copy()
    rows["_trajectory_id"] = np.arange(len(rows))
    output: list[dict[str, object]] = []

    for scenario in DEFAULT_EXECUTION_SCENARIOS:
        cash = INITIAL_CASH
        positions: dict[str, dict[str, object]] = {}
        ledger_status: dict[int, str] = {}
        events: list[tuple[int, int, str, int, str]] = []

        missing_entries = 0
        for item in rows.to_dict("records"):
            tid = int(item["_trajectory_id"])
            ticker = str(item["ticker"])
            if not _positive(item.get("entry_price")):
                ledger_status[tid] = "missing_entry_reference"
                missing_entries += 1
                continue
            events.append(
                (
                    int(item["entry_t"]),
                    1,
                    ticker,
                    tid,
                    "entry",
                )
            )
            if (
                item.get("status") == "completed"
                and _positive(item.get("exit_t"))
                and _positive(item.get("exit_price"))
            ):
                events.append(
                    (
                        int(float(item["exit_t"])),
                        0,
                        ticker,
                        tid,
                        "exit",
                    )
                )

        events.sort()
        accepted = 0
        closed = 0
        blocked_cash = 0
        blocked_position = 0
        realized_pnl = 0.0
        max_open_positions = 0

        by_id = {
            int(item["_trajectory_id"]): item
            for item in rows.to_dict("records")
        }

        for timestamp, _priority, ticker, tid, kind in events:
            item = by_id[tid]
            if kind == "exit":
                pos = positions.get(ticker)
                if pos is None or int(pos["trajectory_id"]) != tid:
                    continue
                proceeds = float(pos["qty"]) * modeled_sell_fill(
                    float(item["exit_price"]),
                    scenario,
                )
                proceeds *= 1.0 - scenario.sell_fee_bps / 10_000.0
                pnl = proceeds - float(pos["cost"])
                cash += proceeds
                realized_pnl += pnl
                closed += 1
                ledger_status[tid] = "closed"
                del positions[ticker]
                continue

            if ticker in positions:
                ledger_status[tid] = "blocked_open_position"
                blocked_position += 1
                continue
            if cash + 1e-9 < ORDER_BUDGET:
                ledger_status[tid] = "blocked_cash"
                blocked_cash += 1
                continue

            cash -= ORDER_BUDGET
            qty = ORDER_BUDGET / modeled_buy_fill(
                float(item["entry_price"]),
                scenario,
            )
            positions[ticker] = {
                "trajectory_id": tid,
                "qty": qty,
                "cost": ORDER_BUDGET,
                "entry_t": int(item["entry_t"]),
                "day": str(item["trading_day"]),
            }
            accepted += 1
            max_open_positions = max(max_open_positions, len(positions))
            ledger_status[tid] = (
                "open_unresolved"
                if item.get("status") != "completed"
                else "open"
            )

        marked_value = 0.0
        for ticker, pos in positions.items():
            item = by_id[int(pos["trajectory_id"])]
            group = state_groups.get(
                (str(item["trading_day"]), str(ticker))
            )
            mark = float(item["entry_price"])
            if group is not None and not group.empty:
                after = group.loc[
                    pd.to_numeric(
                        group["t"],
                        errors="coerce",
                    ).ge(int(pos["entry_t"]))
                ].copy()
                closes = pd.to_numeric(after["c"], errors="coerce")
                valid = after.loc[closes.gt(0)]
                if not valid.empty:
                    mark = float(valid.iloc[-1]["c"])
            marked_value += float(pos["qty"]) * mark

        ending_marked = cash + marked_value
        unresolved = len(positions)
        complete = unresolved == 0 and missing_entries == 0
        output.append(
            {
                "month": month,
                "policy": policy,
                "scenario": scenario.name,
                "attempts": int(len(rows)),
                "accepted": accepted,
                "closed": closed,
                "unresolved": unresolved,
                "missing_entry_references": missing_entries,
                "blocked_cash": blocked_cash,
                "blocked_position": blocked_position,
                "max_open_positions": max_open_positions,
                "initial_cash": INITIAL_CASH,
                "ending_cash": cash,
                "committed_capital": unresolved * ORDER_BUDGET,
                "realized_pnl": realized_pnl,
                "ending_stale_marked_equity": ending_marked,
                "marked_return_pct": (
                    ending_marked / INITIAL_CASH - 1.0
                )
                * 100.0,
                "complete_accounting": complete,
            }
        )
    return pd.DataFrame(output)


def run_fold(
    anchor_paths: dict[str, Path],
    state_paths: dict[str, Path],
    evaluation_month: str,
):
    fit, calibration, evaluation, provenance = load_fold(
        anchor_paths,
        evaluation_month,
    )
    opportunity = train_opportunity_model(fit, calibration)

    partitions = []
    for name, frame in (
        ("fit", fit),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        item = frame.copy()
        item["_partition"] = name
        item["opportunity_probability"] = predict_opportunity(
            item,
            opportunity,
        )
        partitions.append(item)
    anchors = pd.concat(partitions, ignore_index=True)

    sequence_parts = []
    state_cache: dict[str, pd.DataFrame] = {}
    for month, group in anchors.groupby(_month_of(anchors), sort=True):
        if month not in state_paths:
            raise ValueError(f"missing state panel for {month}")
        state = pd.read_parquet(state_paths[month])
        state_cache[month] = state
        sequence_parts.append(build_continuation_rows(state, group))

    sequence = pd.concat(sequence_parts, ignore_index=True)
    fit_rows = sequence.loc[
        sequence["_partition"].eq("fit")
    ].reset_index(drop=True)
    cal_rows = sequence.loc[
        sequence["_partition"].eq("calibration")
    ].reset_index(drop=True)
    eval_rows = sequence.loc[
        sequence["_partition"].eq("evaluation")
    ].reset_index(drop=True)

    model = train_transition_model(fit_rows, cal_rows)
    eval_rows["hold_probability"] = predict_transition_continuation(
        eval_rows,
        model,
    )

    eval_anchors = anchors.loc[
        anchors["_partition"].eq("evaluation")
    ].reset_index(drop=True)
    eval_state = state_cache.get(evaluation_month)
    if eval_state is None:
        eval_state = pd.read_parquet(state_paths[evaluation_month])

    recurrent, decisions = build_recurrent_trajectories(
        eval_anchors,
        eval_rows,
        eval_state,
        evaluation_month,
    )
    comparator = build_always_hold_comparator(
        recurrent,
        eval_state,
        evaluation_month,
    )
    summary, bootstrap = trade_summary(
        recurrent,
        comparator,
        evaluation_month,
    )

    recurrent_accounts = account_replay(
        recurrent,
        eval_state,
        "recurrent_v38",
        evaluation_month,
    )
    comparator_accounts = account_replay(
        comparator,
        eval_state,
        "always_hold_30m",
        evaluation_month,
    )
    accounts = pd.concat(
        [recurrent_accounts, comparator_accounts],
        ignore_index=True,
    )

    primary = summary.iloc[0]
    boot = bootstrap.iloc[0]
    base_account = accounts.loc[
        accounts["policy"].eq("recurrent_v38")
        & accounts["scenario"].eq("base")
    ]
    if len(base_account) != 1:
        raise AssertionError("expected one recurrent BASE account row")
    account = base_account.iloc[0]

    success = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "at_least_100_completed": bool(
                    primary["completed_trades"] >= 100
                ),
                "completion_at_least_90pct": bool(
                    primary["completion_coverage"] >= 0.90
                ),
                "base_mean_positive": bool(
                    primary["recurrent_base_net_return_pct_mean"] > 0
                ),
                "day_base_mean_positive": bool(
                    primary["recurrent_day_balanced_base_mean_pct"] > 0
                ),
                "base_bootstrap_low_positive": bool(
                    boot["recurrent_ci_low_pct"] > 0
                ),
                "beats_30m_day_mean": bool(
                    primary["recurrent_day_balanced_base_mean_pct"]
                    > primary[
                        "comparator_day_balanced_base_mean_pct"
                    ]
                ),
                "difference_bootstrap_low_positive": bool(
                    boot["difference_ci_low_pct"] > 0
                ),
                "base_account_positive_complete": bool(
                    account["marked_return_pct"] > 0
                    and bool(account["complete_accounting"])
                ),
            }
        ]
    )
    success["all_frozen_checks_pass"] = success.iloc[:, 1:].all(axis=1)

    exit_reasons = (
        recurrent.groupby(
            ["month", "exit_reason"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    return (
        recurrent,
        decisions,
        comparator,
        summary,
        bootstrap,
        exit_reasons,
        accounts,
        provenance,
        success,
    )


def render_report(outputs: tuple[pd.DataFrame, ...]) -> str:
    (
        _recurrent,
        _decisions,
        _comparator,
        summary,
        bootstrap,
        exit_reasons,
        accounts,
        provenance,
        success,
    ) = outputs
    lines = [
        "=== MoneyMaker Recurrent Path Policy v3.8 ===",
        "frozen v3.7 classifier composed into reached ENTRY->HOLD/EXIT paths",
        "full round-trip light/base/stress friction; hard maximum hold=30m",
        "missing decision/open => first subsequent observed open exit; no gap compression",
        "primary comparator=always HOLD same gated entry to 30m",
        "April 2026+ sealed",
    ]
    for title, frame in (
        ("Trade summary", summary),
        ("Day bootstrap", bootstrap),
        ("Exit reasons", exit_reasons),
        ("Account replay", accounts),
        ("Provenance", provenance),
        ("Frozen checks", success),
    ):
        lines.extend(("", f"=== {title} ===", frame.to_string(index=False)))
    return "\n".join(lines)


def _parse(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-month", required=True)
    parser.add_argument("--anchor", action="append", type=_parse, required=True)
    parser.add_argument("--state", action="append", type=_parse, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trajectories-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--comparator-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-csv", type=Path, required=True)
    parser.add_argument("--exit-reasons-csv", type=Path, required=True)
    parser.add_argument("--accounts-csv", type=Path, required=True)
    parser.add_argument("--provenance-csv", type=Path, required=True)
    parser.add_argument("--success-csv", type=Path, required=True)
    args = parser.parse_args()

    outputs = run_fold(
        dict(args.anchor),
        dict(args.state),
        args.evaluation_month,
    )
    report = render_report(outputs)
    print(report, flush=True)

    paths = (
        args.trajectories_csv,
        args.decisions_csv,
        args.comparator_csv,
        args.summary_csv,
        args.bootstrap_csv,
        args.exit_reasons_csv,
        args.accounts_csv,
        args.provenance_csv,
        args.success_csv,
    )
    for path, frame in zip(paths, outputs, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
