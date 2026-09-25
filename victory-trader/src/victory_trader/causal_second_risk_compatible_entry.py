"""Request 197: causal one-second risk-compatible entry bridge.

Development-only. Trains on Request-171 FIT, chooses rule/threshold on its
chronological calibration block, and evaluates on already-open Request-178
development dates. The target is the realized outcome of the same causal
one-second risk policy that is evaluated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

from .causal_second_execution import RiskRule, SecondBar, replay_long
from .config import load_settings
from .decomposed_cost_aware_entry_surplus import FRESH_DAYS
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, ExecutionScenario
from .future_cost_cover_state_observability import _day_weights
from .halts import fetch_nasdaq_halts
from .market_regime_position_value import attach_market_regime, build_market_regime
from .massive_client import MassiveClient
from .rich_second_position_value import RICH_SECOND_FEATURES, rich_second_features
from .second_path_attention_probe import SECOND_CACHE_DIR, _second_frame
from .shadow_entry_repricing_decomposition import (
    EPISODE_KEYS,
    FEATURES,
    SHADOW_MAX_MINUTE,
    SHADOW_MIN_MINUTE,
    add_shadow_features,
)

REQUEST_ID = 197
BASE_SCENARIO: ExecutionScenario = next(
    item for item in DEFAULT_EXECUTION_SCENARIOS if item.name == "base"
)
LATENCY_MS = 1_000
ENTRY_EXPIRY_MS = 5_000
MINUTES_TO_CLOSE_FLOOR = 31.0
QUANTILES = (0.80, 0.90, 0.95)
MIN_CAL_TRADES = 30
MIN_SELECTION_RATE = 0.05
MAX_SELECTION_RATE = 0.50
MIN_CLOSED_COVERAGE = 0.90
MODEL_SEED = 20261107
BOOTSTRAP_SEED = 20261108
BOOTSTRAP_SAMPLES = 10_000

RULES = {
    "tight": RiskRule(
        stop_pct=3.0,
        take_pct=5.0,
        trail_retrace_pct=5.0,
        partial_fraction=0.5,
        max_hold_ms=30 * 60_000,
    ),
    "runner": RiskRule(
        stop_pct=3.0,
        take_pct=10.0,
        trail_retrace_pct=7.0,
        partial_fraction=0.5,
        max_hold_ms=30 * 60_000,
    ),
}


@dataclass(frozen=True)
class FittedRuleModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    low: float
    high: float


@dataclass(frozen=True)
class SelectedPolicy:
    rule_name: str
    score_threshold: float
    quantile: float
    calibration_day_balanced_mean_pct: float


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    status = frame["replay_status"].fillna("missing").astype(str)
    return {str(k): int(v) for k, v in status.value_counts().sort_index().items()}


def _day_balanced_mean(frame: pd.DataFrame, column: str) -> float | None:
    work = frame.loc[frame["replay_status"].eq("closed")].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[column])
    if work.empty:
        return None
    daily = work.groupby(work["trading_day"].astype(str), sort=True)[column].mean()
    return float(daily.mean()) if len(daily) else None


def _daily_bootstrap(frame: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    work = frame.loc[frame["replay_status"].eq("closed")].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[column])
    daily = (
        work.groupby(work["trading_day"].astype(str), sort=True)[column]
        .mean()
        .dropna()
    )
    if daily.empty:
        return {"days": 0, "mean_pct": None, "ci_low_pct": None, "ci_high_pct": None}
    values = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    means = samples.mean(axis=1)
    return {
        "days": int(len(values)),
        "mean_pct": float(values.mean()),
        "ci_low_pct": float(np.quantile(means, 0.025)),
        "ci_high_pct": float(np.quantile(means, 0.975)),
    }


def _summarize(frame: pd.DataFrame) -> dict[str, object]:
    total = int(len(frame))
    closed = frame.loc[frame["replay_status"].eq("closed")].copy()
    values = pd.to_numeric(closed["policy_net_return_pct"], errors="coerce").dropna()
    closed = closed.loc[values.index].copy()
    wins = values.loc[values.gt(0)]
    losses = values.loc[values.lt(0)]
    by_day = (
        closed.assign(_value=values)
        .groupby(closed["trading_day"].astype(str), sort=True)["_value"]
        .mean()
    )
    return {
        "episodes": total,
        "closed": int(len(closed)),
        "closed_coverage": float(len(closed) / total) if total else None,
        "status_counts": _status_counts(frame) if total else {},
        "mean_net_return_pct": float(values.mean()) if len(values) else None,
        "day_balanced_net_return_pct": _day_balanced_mean(
            frame, "policy_net_return_pct"
        ),
        "daily_bootstrap": _daily_bootstrap(frame, "policy_net_return_pct"),
        "positive_rate": float(values.gt(0).mean()) if len(values) else None,
        "mean_winner_pct": float(wins.mean()) if len(wins) else None,
        "mean_loser_pct": float(losses.mean()) if len(losses) else None,
        "payoff_ratio": (
            float(wins.mean() / abs(losses.mean()))
            if len(wins) and len(losses) and float(losses.mean()) != 0
            else None
        ),
        "severe_loss_rate": float(values.le(-5.0).mean()) if len(values) else None,
        "positive_days": int(by_day.gt(0).sum()) if len(by_day) else 0,
        "by_day_mean_pct": {str(k): float(v) for k, v in by_day.items()},
        "entry_minute": {
            str(int(k)): int(v)
            for k, v in pd.to_numeric(
                frame.get("minutes_held"), errors="coerce"
            ).dropna().astype(int).value_counts().sort_index().items()
        },
    }


def _feature_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def _state_filter(frame: pd.DataFrame) -> pd.DataFrame:
    held = pd.to_numeric(frame["minutes_held"], errors="coerce")
    to_close = pd.to_numeric(frame["minutes_to_close"], errors="coerce")
    return frame.loc[
        held.notna()
        & held.between(SHADOW_MIN_MINUTE, SHADOW_MAX_MINUTE)
        & to_close.ge(MINUTES_TO_CLOSE_FLOOR)
    ].copy()


def _halt_intervals(day: str) -> dict[str, list[tuple[int, int]]]:
    records = fetch_nasdaq_halts(date.fromisoformat(day))
    out: dict[str, list[tuple[int, int]]] = {}
    far_future = 2**63 - 1
    for record in records:
        out.setdefault(record.symbol.upper(), []).append(
            (
                int(record.halt_at_ms),
                int(record.resume_at_ms) if record.resume_at_ms is not None else far_future,
            )
        )
    return out


def _bars_from_seconds(
    seconds: pd.DataFrame,
    *,
    ticker: str,
    intervals: dict[str, list[tuple[int, int]]],
) -> list[SecondBar]:
    windows = intervals.get(ticker.upper(), [])
    bars: list[SecondBar] = []
    for row in seconds.to_dict("records"):
        t = int(row["t"])
        halted = any(start <= t < end for start, end in windows)
        bars.append(
            SecondBar(
                t=t,
                o=float(row["o"]),
                h=float(row["h"]),
                low=float(row["l"]),
                c=float(row["c"]),
                halted=halted,
            )
        )
    return bars


class SecondStore:
    def __init__(self) -> None:
        settings = load_settings()
        self.client = MassiveClient(
            settings.massive_api_key,
            cache_dir=SECOND_CACHE_DIR,
            request_interval_seconds=0.0,
        )
        self._seconds: dict[tuple[str, str], pd.DataFrame] = {}
        self._halts: dict[str, dict[str, list[tuple[int, int]]]] = {}
        self.halt_status: dict[str, str] = {}

    def seconds(self, day: str, ticker: str) -> pd.DataFrame:
        key = (day, ticker.upper())
        if key not in self._seconds:
            payload = self.client.second_bars_range(
                ticker,
                date.fromisoformat(day),
                date.fromisoformat(day),
                adjusted=False,
            )
            self._seconds[key] = _second_frame(payload)
        return self._seconds[key]

    def halts(self, day: str) -> dict[str, list[tuple[int, int]]]:
        if day not in self._halts:
            try:
                self._halts[day] = _halt_intervals(day)
                self.halt_status[day] = "available"
            except (
                requests.RequestException,
                ElementTree.ParseError,
                ValueError,
            ) as exc:
                # Unknown halt coverage is explicit. Missing second bars still
                # cannot fill orders, but this development bridge must not
                # pretend that an unavailable official feed proves no halts.
                self._halts[day] = {}
                self.halt_status[day] = (
                    f"unavailable:{type(exc).__name__}"
                )
        return self._halts[day]


def _enrich_rich_second(frame: pd.DataFrame, store: SecondStore) -> pd.DataFrame:
    work = frame.copy()
    records: list[dict[str, float]] = []
    for row in work.loc[:, ["trading_day", "ticker", "state_t"]].to_dict("records"):
        seconds = store.seconds(str(row["trading_day"]), str(row["ticker"]))
        records.append(rich_second_features(seconds, int(row["state_t"])))
    rich = pd.DataFrame(records, index=work.index)
    for column in RICH_SECOND_FEATURES:
        if column not in work.columns:
            work[column] = rich[column]
        else:
            existing = pd.to_numeric(work[column], errors="coerce")
            work[column] = existing.where(existing.notna(), rich[column])
    return work


def _replay_state(
    row: dict[str, object],
    store: SecondStore,
    rule: RiskRule,
) -> dict[str, object]:
    day = str(row["trading_day"])
    ticker = str(row["ticker"]).upper()
    state_t = int(row["state_t"])
    seconds = store.seconds(day, ticker)
    end_t = state_t + rule.max_hold_ms + LATENCY_MS + 10_000
    path = seconds.loc[
        pd.to_numeric(seconds["t"], errors="coerce").between(
            state_t,
            end_t,
            inclusive="both",
        )
    ].copy()
    bars = _bars_from_seconds(
        path,
        ticker=ticker,
        intervals=store.halts(day),
    )
    result = replay_long(
        bars,
        decision_t=state_t,
        rule=rule,
        scenario=BASE_SCENARIO,
        latency_ms=LATENCY_MS,
        entry_expiry_ms=ENTRY_EXPIRY_MS,
        quantity=1.0,
    )
    entry_fill_t = result.entry.fill_t if result.entry is not None else None
    exit_fill_t = result.exits[-1].fill_t if result.exits else None
    exit_reasons = [fill.reason for fill in result.exits]
    return {
        "replay_status": result.status,
        "replay_reason": result.reason,
        "first_exit_reason": exit_reasons[0] if exit_reasons else None,
        "exit_reason_sequence": ",".join(exit_reasons),
        "partial_take_reached": "partial_take" in exit_reasons,
        "policy_net_return_pct": result.net_return_pct,
        "entry_fill_t": entry_fill_t,
        "exit_fill_t": exit_fill_t,
        "entry_reference_price": (
            result.entry.reference_price if result.entry is not None else np.nan
        ),
        "entry_modeled_price": (
            result.entry.modeled_price if result.entry is not None else np.nan
        ),
    }


def prepare_states(
    positions: pd.DataFrame,
    scan: pd.DataFrame,
    store: SecondStore,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = _state_filter(positions)
    base = add_shadow_features(base)
    regime = build_market_regime(scan)
    base = attach_market_regime(base, regime)
    base = _enrich_rich_second(base, store)

    by_rule: dict[str, pd.DataFrame] = {}
    for rule_name, rule in RULES.items():
        records = [
            _replay_state(row, store, rule)
            for row in base.to_dict("records")
        ]
        replay = pd.DataFrame(records, index=base.index)
        by_rule[rule_name] = pd.concat(
            [base.reset_index(drop=True), replay.reset_index(drop=True)],
            axis=1,
        )
    return base, by_rule


def train_model(frame: pd.DataFrame, *, seed: int) -> FittedRuleModel:
    target = pd.to_numeric(frame["policy_net_return_pct"], errors="coerce")
    valid = frame["replay_status"].eq("closed") & target.notna()
    train = frame.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 500:
        raise ValueError(f"request 197 insufficient closed FIT rows: {len(train)}")
    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    columns = tuple(
        column
        for column in FEATURES
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
        _feature_frame(train, columns),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(train),
    )
    return FittedRuleModel(model=model, columns=columns, low=float(low), high=float(high))


def score(frame: pd.DataFrame, fitted: FittedRuleModel) -> np.ndarray:
    return fitted.model.predict(_feature_frame(frame, fitted.columns))


def first_qualifying(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    work = frame.copy()
    work["_score"] = pd.to_numeric(work["entry_score"], errors="coerce")
    work = work.loc[work["_score"].notna()].sort_values(
        EPISODE_KEYS + ["state_t"], kind="stable"
    )
    eligible = work.loc[work["_score"].ge(float(threshold))].copy()
    if eligible.empty:
        return eligible
    return eligible.groupby(EPISODE_KEYS, sort=False, as_index=False).head(1).copy()


def first_state_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_values(EPISODE_KEYS + ["state_t"], kind="stable")
    return work.groupby(EPISODE_KEYS, sort=False, as_index=False).head(1).copy()


def first_admitted_qualifying(
    frame: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reattempt after an expired/unavailable entry, but never after admission.

    A score crossing submits an order. If no eligible open arrives before the
    entry expiry, no position exists and the trader may keep observing. Closed,
    unresolved and ambiguous paths all imply an admitted position and terminate
    that episode's entry search.
    """
    work = frame.copy()
    work["_score"] = pd.to_numeric(work["entry_score"], errors="coerce")
    work = work.loc[work["_score"].notna()].sort_values(
        EPISODE_KEYS + ["state_t"], kind="stable"
    )
    chosen: list[pd.Series] = []
    signaled_episodes = 0
    expired_attempts = 0
    for _, group in work.groupby(EPISODE_KEYS, sort=False):
        qualified = group.loc[group["_score"].ge(float(threshold))]
        if qualified.empty:
            continue
        signaled_episodes += 1
        for _, row in qualified.iterrows():
            if str(row["replay_status"]) == "entry_unavailable":
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


def first_admitted_baseline(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = frame.sort_values(EPISODE_KEYS + ["state_t"], kind="stable")
    chosen: list[pd.Series] = []
    expired_attempts = 0
    episodes = 0
    for _, group in work.groupby(EPISODE_KEYS, sort=False):
        episodes += 1
        for _, row in group.iterrows():
            if str(row["replay_status"]) == "entry_unavailable":
                expired_attempts += 1
                continue
            chosen.append(row)
            break
    selected = (
        pd.DataFrame(chosen)
        if chosen
        else work.iloc[0:0].copy()
    )
    return selected, {
        "episodes": int(episodes),
        "expired_entry_attempts": int(expired_attempts),
        "admitted_episodes": int(len(selected)),
    }


def _selection_rate(selected: pd.DataFrame, all_states: pd.DataFrame) -> float:
    total = int(all_states.loc[:, EPISODE_KEYS].drop_duplicates().shape[0])
    return float(len(selected) / total) if total else 0.0


def choose_policy(
    calibration_by_rule: dict[str, pd.DataFrame],
) -> tuple[SelectedPolicy | None, dict[str, object]]:
    table: dict[str, object] = {}
    eligible: list[tuple[float, float, float, str, float, float]] = []
    for rule_name, frame in calibration_by_rule.items():
        scores = pd.to_numeric(frame["entry_score"], errors="coerce").dropna()
        for quantile in QUANTILES:
            threshold = float(scores.quantile(quantile))
            selected, action_audit = first_admitted_qualifying(
                frame, threshold
            )
            summary = _summarize(selected)
            rate = _selection_rate(selected, frame)
            key = f"{rule_name}_p{int(round(100 * quantile))}"
            table[key] = {
                "rule": rule_name,
                "quantile": quantile,
                "threshold": threshold,
                "selection_rate": rate,
                "action_audit": action_audit,
                "summary": summary,
            }
            day_mean = summary["day_balanced_net_return_pct"]
            severe = summary["severe_loss_rate"]
            coverage = summary["closed_coverage"]
            if (
                len(selected) >= MIN_CAL_TRADES
                and MIN_SELECTION_RATE <= rate <= MAX_SELECTION_RATE
                and coverage is not None
                and float(coverage) >= MIN_CLOSED_COVERAGE
                and day_mean is not None
                and severe is not None
            ):
                eligible.append(
                    (
                        -float(day_mean),
                        float(severe),
                        float(rate),
                        rule_name,
                        float(threshold),
                        float(quantile),
                    )
                )
    if not eligible:
        return None, table
    eligible.sort()
    best = eligible[0]
    chosen = SelectedPolicy(
        rule_name=best[3],
        score_threshold=best[4],
        quantile=best[5],
        calibration_day_balanced_mean_pct=-best[0],
    )
    return chosen, table


def matched_difference(
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, object]:
    left = selected.loc[
        selected["replay_status"].eq("closed"),
        [*EPISODE_KEYS, "policy_net_return_pct"],
    ].rename(columns={"policy_net_return_pct": "selected_return"})
    right = baseline.loc[
        baseline["replay_status"].eq("closed"),
        [*EPISODE_KEYS, "policy_net_return_pct"],
    ].rename(columns={"policy_net_return_pct": "baseline_return"})
    merged = left.merge(right, on=EPISODE_KEYS, how="inner", validate="one_to_one")
    merged["difference_pct"] = (
        pd.to_numeric(merged["selected_return"], errors="coerce")
        - pd.to_numeric(merged["baseline_return"], errors="coerce")
    )
    merged = merged.dropna(subset=["difference_pct"])
    if merged.empty:
        return {
            "matched_episodes": 0,
            "mean_difference_pct": None,
            "day_balanced_difference_pct": None,
            "bootstrap": {"days": 0, "mean_pct": None, "ci_low_pct": None, "ci_high_pct": None},
        }
    temp = merged.rename(columns={"difference_pct": "policy_net_return_pct"}).copy()
    temp["replay_status"] = "closed"
    return {
        "matched_episodes": int(len(merged)),
        "mean_difference_pct": float(merged["difference_pct"].mean()),
        "day_balanced_difference_pct": _day_balanced_mean(
            temp, "policy_net_return_pct"
        ),
        "bootstrap": _daily_bootstrap(temp, "policy_net_return_pct"),
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    calibration_positions = pd.read_parquet(calibration_path)
    history_scan = pd.read_parquet(history_scan_path)

    fit_days = set(fit_positions["trading_day"].astype(str))
    cal_days = set(calibration_positions["trading_day"].astype(str))
    fit_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(fit_days)
    ].copy()
    cal_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(cal_days)
    ].copy()

    fresh_position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    fresh_scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if (
        len(fresh_position_paths) != len(FRESH_DAYS)
        or len(fresh_scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError("request 197 requires complete Request-178 fresh shards")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in fresh_position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in fresh_scan_paths],
        ignore_index=True,
    )

    store = SecondStore()
    _, fit_by_rule = prepare_states(fit_positions, fit_scan, store)
    models: dict[str, FittedRuleModel] = {}
    for i, (rule_name, frame) in enumerate(fit_by_rule.items()):
        models[rule_name] = train_model(frame, seed=MODEL_SEED + i)
        frame["entry_score"] = score(frame, models[rule_name])

    _, cal_by_rule = prepare_states(calibration_positions, cal_scan, store)
    for rule_name, frame in cal_by_rule.items():
        frame["entry_score"] = score(frame, models[rule_name])

    chosen, calibration_table = choose_policy(cal_by_rule)

    if chosen is None:
        result = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "opens_new_dates": False,
            "promotion_eligible": False,
            "base_scenario": asdict(BASE_SCENARIO),
            "latency_ms": LATENCY_MS,
            "entry_expiry_ms": ENTRY_EXPIRY_MS,
            "rules": {name: asdict(rule) for name, rule in RULES.items()},
            "training_partition": "request171_fit_only",
            "calibration_partition": "request171_calibration_only",
            "development_days": list(FRESH_DAYS),
            "selected_policy": None,
            "calibration_table": calibration_table,
            "calibration_bridge_pass": False,
            "bridge_failure_reason": (
                "no calibration rule/threshold combination met frozen "
                "trade-count, selection-rate, and closed-coverage floors"
            ),
            "fresh_evaluated": False,
            "development_gate_pass": False,
            "second_client_stats": store.client.stats.to_dict(),
            "halt_feed_by_day": dict(sorted(store.halt_status.items())),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    _, fresh_by_rule = prepare_states(fresh_positions, fresh_scan, store)
    selected_frame = fresh_by_rule[chosen.rule_name]
    selected_frame["entry_score"] = score(
        selected_frame, models[chosen.rule_name]
    )
    selected, selected_action_audit = first_admitted_qualifying(
        selected_frame, chosen.score_threshold
    )
    baseline, baseline_action_audit = first_admitted_baseline(
        selected_frame
    )

    selected_summary = _summarize(selected)
    baseline_summary = _summarize(baseline)
    matched = matched_difference(selected, baseline)
    selected_rate = _selection_rate(selected, selected_frame)

    bootstrap_low = selected_summary["daily_bootstrap"]["ci_low_pct"]
    matched_day = matched["day_balanced_difference_pct"]
    selected_severe = selected_summary["severe_loss_rate"]
    baseline_severe = baseline_summary["severe_loss_rate"]
    gate = bool(
        MIN_SELECTION_RATE <= selected_rate <= MAX_SELECTION_RATE
        and selected_summary["closed_coverage"] is not None
        and float(selected_summary["closed_coverage"]) >= MIN_CLOSED_COVERAGE
        and selected_summary["day_balanced_net_return_pct"] is not None
        and float(selected_summary["day_balanced_net_return_pct"]) > 0
        and bootstrap_low is not None
        and float(bootstrap_low) > 0
        and int(selected_summary["positive_days"]) >= 4
        and matched_day is not None
        and float(matched_day) > 0
        and selected_severe is not None
        and baseline_severe is not None
        and float(selected_severe) < float(baseline_severe)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "base_scenario": asdict(BASE_SCENARIO),
        "latency_ms": LATENCY_MS,
        "entry_expiry_ms": ENTRY_EXPIRY_MS,
        "rules": {name: asdict(rule) for name, rule in RULES.items()},
        "training_partition": "request171_fit_only",
        "calibration_partition": "request171_calibration_only",
        "development_days": list(FRESH_DAYS),
        "selected_policy": asdict(chosen),
        "calibration_table": calibration_table,
        "fresh_selection_rate": selected_rate,
        "fresh_selected": selected_summary,
        "fresh_selected_action_audit": selected_action_audit,
        "fresh_earliest_executable_baseline": baseline_summary,
        "fresh_baseline_action_audit": baseline_action_audit,
        "matched_selected_minus_minute1": matched,
        "development_gate_pass": gate,
        "second_client_stats": store.client.stats.to_dict(),
        "halt_feed_by_day": dict(sorted(store.halt_status.items())),
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
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
