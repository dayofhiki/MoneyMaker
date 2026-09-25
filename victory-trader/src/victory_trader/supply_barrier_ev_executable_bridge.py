"""Request 202: supply barrier-EV executable bridge.

FIT-only training, chronological calibration action evaluation. Request-178
remains sealed unless the frozen calibration bridge passes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .causal_first_passage_directional_edge import (
    add_first_passage,
    prepare_directional_states,
)
from .causal_second_risk_compatible_entry import (
    RULES,
    SecondStore,
    _replay_state,
    _summarize,
    first_admitted_baseline,
    matched_difference,
)
from .future_cost_cover_state_observability import (
    _day_weights,
    _safe_spearman,
)
from .supply_first_passage_directional_edge import (
    BASE_COLUMNS,
    RegularVolumeIndex,
    SUPPLY_FEATURES,
    SupplyStore,
    add_supply_features,
    coverage,
)

REQUEST_ID = 202
RULE_NAME = "runner"
MODEL_SEED = 20261203
MIN_SELECTION_RATE = 0.05
MAX_SELECTION_RATE = 0.50
MIN_ADMITTED = 30
MIN_BARRIER_COVERAGE = 0.95
MIN_BARRIER_SPEARMAN = 0.10
MIN_RISK_COVERAGE = 0.85
MIN_POSITIVE_DAYS = 6


@dataclass(frozen=True)
class EVModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]


def _x(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def fit_ev(
    frame: pd.DataFrame,
    *,
    include_supply: bool,
    seed: int,
) -> EVModel:
    target = pd.to_numeric(
        frame["barrier_proxy_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = frame.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 500:
        raise ValueError("request 202 insufficient barrier-EV FIT support")
    columns = tuple(
        column
        for column in (
            [*BASE_COLUMNS, *SUPPLY_FEATURES]
            if include_supply
            else list(BASE_COLUMNS)
        )
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _x(train, columns),
        y,
        sample_weight=_day_weights(train),
    )
    return EVModel(model=model, columns=columns)


def predict(frame: pd.DataFrame, model: EVModel) -> np.ndarray:
    return model.model.predict(_x(frame, model.columns))


def _by_day_spearman(
    frame: pd.DataFrame,
    score_column: str,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for day, group in frame.groupby(
        frame["trading_day"].astype(str),
        sort=True,
    ):
        out[str(day)] = _safe_spearman(
            group["barrier_proxy_pct"],
            group[score_column],
        )
    return out


def _by_day_spearman(
    frame: pd.DataFrame,
    score_column: str,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for day, group in frame.groupby(
        frame["trading_day"].astype(str),
        sort=True,
    ):
        out[str(day)] = _safe_spearman(
            group["barrier_proxy_pct"],
            group[score_column],
        )
    return out


def select_positive_ev(
    frame: pd.DataFrame,
    score_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = frame.copy()
    work["_score"] = pd.to_numeric(
        work[score_column],
        errors="coerce",
    )
    work = work.loc[work["_score"].notna()].sort_values(
        ["trading_day", "ticker", "hot_t", "state_t"],
        kind="stable",
    )
    chosen: list[pd.Series] = []
    signaled_episodes = 0
    expired_attempts = 0
    for _, group in work.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        positive = group.loc[group["_score"].gt(0)]
        if positive.empty:
            continue
        signaled_episodes += 1
        for _, row in positive.iterrows():
            if str(row["first_passage_status"]) == "entry_unavailable":
                expired_attempts += 1
                continue
            chosen.append(row)
            break
    selected = (
        pd.DataFrame(chosen).drop(columns=["_score"], errors="ignore")
        if chosen
        else work.iloc[0:0].drop(columns=["_score"], errors="ignore").copy()
    )
    return selected, {
        "signaled_episodes": int(signaled_episodes),
        "expired_entry_attempts": int(expired_attempts),
        "admitted_episodes": int(len(selected)),
    }


def _selection_rate(
    selected: pd.DataFrame,
    all_states: pd.DataFrame,
) -> float:
    total = int(
        all_states[
            ["trading_day", "ticker", "hot_t"]
        ].drop_duplicates().shape[0]
    )
    return float(len(selected) / total) if total else 0.0


def barrier_summary(frame: pd.DataFrame) -> dict[str, object]:
    total = int(len(frame))
    proxy = pd.to_numeric(
        frame["barrier_proxy_pct"],
        errors="coerce",
    )
    valid = proxy.notna()
    work = frame.loc[valid].copy()
    work["_proxy"] = proxy.loc[valid]
    daily = (
        work.groupby(
            work["trading_day"].astype(str),
            sort=True,
        )["_proxy"].mean()
        if not work.empty
        else pd.Series(dtype=float)
    )
    status = frame["first_passage_status"].astype(str)
    return {
        "episodes": total,
        "evaluable": int(valid.sum()),
        "evaluable_coverage": (
            float(valid.mean()) if total else None
        ),
        "mean_proxy_pct": (
            float(work["_proxy"].mean())
            if not work.empty
            else None
        ),
        "day_balanced_proxy_pct": (
            float(daily.mean()) if len(daily) else None
        ),
        "positive_days": int(daily.gt(0).sum()),
        "by_day_mean_pct": {
            str(k): float(v)
            for k, v in daily.items()
        },
        "take_first_rate": (
            float(status.eq("take_first").mean())
            if total else None
        ),
        "stop_first_rate": (
            float(status.eq("stop_first").mean())
            if total else None
        ),
        "neither_rate": (
            float(status.eq("neither").mean())
            if total else None
        ),
        "ambiguous_rate": (
            float(status.eq("ambiguous").mean())
            if total else None
        ),
        "predicted_ev_mean_pct": (
            float(
                pd.to_numeric(
                    frame["supply_barrier_ev"],
                    errors="coerce",
                ).mean()
            )
            if total else None
        ),
    }


def replay_selected(
    selected: pd.DataFrame,
    store: SecondStore,
) -> pd.DataFrame:
    if selected.empty:
        out = selected.copy()
        out["replay_status"] = pd.Series(dtype=str)
        out["policy_net_return_pct"] = pd.Series(dtype=float)
        return out
    records: list[dict[str, object]] = []
    for row in selected.to_dict("records"):
        replay = _replay_state(
            row,
            store,
            RULES[RULE_NAME],
        )
        records.append({**row, **replay})
    return pd.DataFrame(records)


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    cal_positions = pd.read_parquet(calibration_path)
    scan = pd.read_parquet(history_scan_path)

    fit_days = set(fit_positions["trading_day"].astype(str))
    cal_days = set(cal_positions["trading_day"].astype(str))
    fit_scan = scan.loc[
        scan["trading_day"].astype(str).isin(fit_days)
    ].copy()
    cal_scan = scan.loc[
        scan["trading_day"].astype(str).isin(cal_days)
    ].copy()

    second_store = SecondStore()
    fit = prepare_directional_states(
        fit_positions,
        fit_scan,
        second_store,
    )
    cal = prepare_directional_states(
        cal_positions,
        cal_scan,
        second_store,
    )

    supply_store = SupplyStore()
    volume_index = RegularVolumeIndex(second_store)
    fit = add_supply_features(
        fit,
        supply_store,
        volume_index,
    )
    cal = add_supply_features(
        cal,
        supply_store,
        volume_index,
    )

    take_pct = RULES[RULE_NAME].take_pct
    fit = add_first_passage(
        fit,
        second_store,
        take_pct=take_pct,
    )
    cal = add_first_passage(
        cal,
        second_store,
        take_pct=take_pct,
    )

    base_model = fit_ev(
        fit,
        include_supply=False,
        seed=MODEL_SEED,
    )
    supply_model = fit_ev(
        fit,
        include_supply=True,
        seed=MODEL_SEED + 1,
    )
    cal["base_barrier_ev"] = predict(
        cal,
        base_model,
    )
    cal["supply_barrier_ev"] = predict(
        cal,
        supply_model,
    )

    base_spearman = _safe_spearman(
        cal["barrier_proxy_pct"],
        cal["base_barrier_ev"],
    )
    supply_spearman = _safe_spearman(
        cal["barrier_proxy_pct"],
        cal["supply_barrier_ev"],
    )

    selected, action_audit = select_positive_ev(
        cal,
        "supply_barrier_ev",
    )
    selected_rate = _selection_rate(selected, cal)
    barrier = barrier_summary(selected)

    selected_risk = replay_selected(
        selected,
        second_store,
    )
    risk_summary = _summarize(selected_risk)

    baseline_states, baseline_audit = first_admitted_baseline(
        cal
    )
    baseline_risk = replay_selected(
        baseline_states,
        second_store,
    )
    baseline_summary = _summarize(baseline_risk)
    matched = matched_difference(
        selected_risk,
        baseline_risk,
    )

    matched_day = matched["day_balanced_difference_pct"]
    gate = bool(
        len(selected) >= MIN_ADMITTED
        and MIN_SELECTION_RATE <= selected_rate <= MAX_SELECTION_RATE
        and barrier["evaluable_coverage"] is not None
        and float(barrier["evaluable_coverage"])
        >= MIN_BARRIER_COVERAGE
        and supply_spearman is not None
        and float(supply_spearman) >= MIN_BARRIER_SPEARMAN
        and barrier["day_balanced_proxy_pct"] is not None
        and float(barrier["day_balanced_proxy_pct"]) > 0
        and int(barrier["positive_days"]) >= MIN_POSITIVE_DAYS
        and risk_summary["closed_coverage"] is not None
        and float(risk_summary["closed_coverage"]) >= MIN_RISK_COVERAGE
        and risk_summary["day_balanced_net_return_pct"] is not None
        and float(risk_summary["day_balanced_net_return_pct"]) > 0
        and int(risk_summary["positive_days"]) >= MIN_POSITIVE_DAYS
        and matched_day is not None
        and float(matched_day) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fresh_evaluated": False,
        "promotion_eligible": False,
        "runner_rule": {
            "stop_pct": RULES[RULE_NAME].stop_pct,
            "take_pct": RULES[RULE_NAME].take_pct,
            "trail_retrace_pct": RULES[RULE_NAME].trail_retrace_pct,
        },
        "training_partition": "request171_fit_only",
        "evaluation_partition": "request171_calibration_only",
        "fit_supply_coverage": coverage(fit),
        "calibration_supply_coverage": coverage(cal),
        "base_barrier_ev_spearman": base_spearman,
        "supply_barrier_ev_spearman": supply_spearman,
        "supply_minus_base_spearman": (
            float(supply_spearman) - float(base_spearman)
            if supply_spearman is not None
            and base_spearman is not None
            else None
        ),
        "base_by_day_spearman": _by_day_spearman(
            cal,
            "base_barrier_ev",
        ),
        "supply_by_day_spearman": _by_day_spearman(
            cal,
            "supply_barrier_ev",
        ),
        "selection_rate": selected_rate,
        "selected_action_audit": action_audit,
        "selected_barrier": barrier,
        "selected_runner_risk": risk_summary,
        "earliest_executable_baseline_audit": baseline_audit,
        "earliest_executable_runner_risk": baseline_summary,
        "matched_selected_minus_earliest": matched,
        "calibration_bridge_pass": gate,
        "second_client_stats": second_store.client.stats.to_dict(),
        "supply_client_stats": supply_store.client.stats.to_dict(),
        "halt_feed_by_day": dict(
            sorted(second_store.halt_status.items())
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
