"""R305: future-knowledge ceilings and separate causal observability audits.

No trading policy is fitted, selected or changed by this diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .execution_costs import DEFAULT_EXECUTION_SCENARIOS
from .frozen_entry_second_hold_exit import KEYS, net
from .hierarchical_crack_entry_controller import _episode_day_weights, _predict
from .lagged_minute_context import CROSSFIT_DAYS
from .pending_exit_integrity import regular_bars, session_limits
from .preentry_context_hold_value import ALL_FEATURES
from .risk_reachable_hold_value import (
    TARGET_PI0, model_fit, replay_scored, signal, validate_fit_days, verify_reproduction,
)
from .weighted_hold_offset import weighted_offset_model

REQUEST_ID = 305
CEILING = "reachable_max_base_net_pct"
ADVANTAGE = "reachable_max_exit_advantage_pct"
LEVELS = (0, 5, 10, 20)


def validate_source(source: pd.DataFrame, raw: pd.DataFrame) -> None:
    if (set(source.trading_day.astype(str)) != set(CROSSFIT_DAYS)
            or source.duplicated([*KEYS, "decision_t"]).any()
            or len(source[KEYS].drop_duplicates()) != 86):
        raise ValueError("R305 requires exact unique86 May5-8 positions")
    if raw.empty or not set(raw.trading_day.astype(str)).issubset(CROSSFIT_DAYS):
        raise ValueError("new raw dates forbidden; June is sealed")


def schedule(group: pd.DataFrame, bars: pd.DataFrame, terminal: int) -> pd.DataFrame:
    """Allowed submissions, not the hindsight maximum of all second highs."""
    times = np.unique(np.r_[group.loc[group.decision_t.lt(terminal), "decision_t"].to_numpy(np.int64), terminal])
    positions = np.searchsorted(bars.t.to_numpy(np.int64), times)
    valid = positions < len(bars)
    prices = np.full(len(times), np.nan)
    fills = np.full(len(times), np.nan)
    prices[valid] = bars.o.to_numpy(float)[positions[valid]]
    fills[valid] = bars.t.to_numpy(np.int64)[positions[valid]]
    price = float(group.entry_price.iloc[0])
    result = pd.DataFrame({"submission_t": times, "fill_t": fills, "gross_pct": (prices/price-1)*100})
    for scenario in DEFAULT_EXECUTION_SCENARIOS:
        result[scenario.name] = [net(price, p, scenario) if np.isfinite(p) else np.nan for p in prices]
    return result


def build_ceiling_labels(source: pd.DataFrame, contexts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = source.copy()
    frame[CEILING], frame[ADVANTAGE] = np.nan, np.nan
    episodes = []
    for key, group in frame.groupby(KEYS, sort=False):
        g = group.sort_values("decision_t")
        bars, _, closing = contexts[key]
        cap = min(int(g.hot_t.iloc[0])+3600000, closing)
        stop = float(g.forced_stop_t.iloc[0])
        terminal = min(int(stop), cap) if np.isfinite(stop) else cap
        feasible = schedule(g, bars, terminal)
        whole = schedule(g, bars, cap)
        # If the terminal market order cannot fill, the ceiling is censored.
        # An earlier finite print does not prove the unobserved maximum failed.
        complete = bool(np.isfinite(feasible.base.iloc[-1]))
        result = {**dict(zip(KEYS, key)), "entry_t": int(g.entry_t.iloc[0]),
                  "entry_price": float(g.entry_price.iloc[0]), "terminal_submission_t": terminal,
                  "terminal_fill_t": float(feasible.fill_t.iloc[-1]), "reachable_complete": complete,
                  "has_eligible_hold": bool(g.risk_reachable_hold.any()),
                  "whole_path_complete": bool(np.isfinite(whole.base.iloc[-1]))}
        for name in ("gross_pct", "light", "base", "stress"):
            result[f"reachable_max_{name}"] = float(feasible[name].max()) if complete else np.nan
            result[f"whole_path_max_{name}"] = float(whole[name].max()) if result["whole_path_complete"] else np.nan
        if complete:
            suffix = np.maximum.accumulate(feasible.base.to_numpy(float)[::-1])[::-1]
            indices = np.searchsorted(feasible.submission_t, g.decision_t.to_numpy(np.int64))
            eligible = g.risk_reachable_hold.to_numpy(bool) & (g.decision_t.to_numpy() < terminal)
            selected = g.index[eligible]
            frame.loc[selected, CEILING] = suffix[indices[eligible]]
            frame.loc[selected, ADVANTAGE] = suffix[indices[eligible]]-g.loc[selected, "exit_now_pct"].to_numpy(float)
        episodes.append(result)
    # Future diagnostic construction must not rewrite the causal representation.
    pd.testing.assert_frame_equal(frame[list(ALL_FEATURES)], source[list(ALL_FEATURES)])
    return frame, pd.DataFrame(episodes)


def truth(values: pd.Series, level: int) -> np.ndarray:
    return (values.gt(0) if level == 0 else values.ge(level)).to_numpy(bool)


def first_states(frame: pd.DataFrame) -> pd.DataFrame:
    # Never skip an ineligible first state to choose a more convenient later one.
    return frame.sort_values("decision_t").groupby(KEYS, sort=False).head(1)


def fit_classifier(train: pd.DataFrame, level: int, kind: str, seed: int):
    y, weights = truth(train[CEILING], level), _episode_day_weights(train)
    prevalence = float(np.average(y, weights=weights))
    if len(np.unique(y)) == 1:
        return None, prevalence
    if kind == "snapshot":
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(), LogisticRegression(C=.1, max_iter=2000, random_state=seed),
        )
        model.fit(train[list(ALL_FEATURES)], y, logisticregression__sample_weight=weights)
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=.04, max_iter=220, max_leaf_nodes=15, min_samples_leaf=75,
            l2_regularization=3., early_stopping=False, random_state=seed,
        )
        model.fit(train[list(ALL_FEATURES)], y, sample_weight=weights)
    return model, prevalence


def crossfit_diagnostics(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eligible = source.loc[source.risk_reachable_hold & source[CEILING].notna()].copy()
    first = first_states(source)
    first = first.loc[first.risk_reachable_hold & first[CEILING].notna()].copy()
    parts, snapshots, provenance = [], [], {}
    for fold, day in enumerate(CROSSFIT_DAYS):
        train = eligible.loc[eligible.trading_day.ne(day)]
        held = eligible.loc[eligible.trading_day.eq(day)].copy()
        first_train = first.loc[first.trading_day.ne(day)]
        first_held = first.loc[first.trading_day.eq(day)].copy()
        fit_days = sorted(train.trading_day.unique())
        validate_fit_days(fit_days, excluded=[day])
        if len(fit_days) != 3 or first_train.empty:
            raise ValueError("R305 requires all three training days")
        seed = 20264050+fold*100
        info = {"fit_days": fit_days, "held_day": day, "seed": seed,
                "training_states": len(train), "training_episodes": len(train[KEYS].drop_duplicates()),
                "training_snapshots": len(first_train), "features": list(ALL_FEATURES), "models": {}}
        for kind, training, scoring in (("snapshot", first_train, first_held), ("states", train, held)):
            for level in LEVELS:
                model, prior = fit_classifier(training, level, kind, seed+level)
                scoring[f"p_net_{level}"] = model.predict_proba(scoring[list(ALL_FEATURES)])[:, 1] if model is not None else prior
                scoring[f"prior_net_{level}"] = prior
                info["models"][f"{kind}_net_{level}"] = {"prior": prior, "single_class_fallback": model is None,
                    "positive_episodes": len(training.loc[truth(training[CEILING], level), KEYS].drop_duplicates())}
        reg = model_fit(train, ADVANTAGE, seed, ALL_FEATURES)
        reg, offset_audit = weighted_offset_model(train.assign(**{TARGET_PI0: train[ADVANTAGE]}), reg)
        held["predicted_max_exit_advantage_pct"] = _predict(held, reg)
        info["regression_offset"] = offset_audit
        parts.append(held)
        snapshots.append(first_held)
        provenance[day] = info
        print(f"R305 observability fold complete {day}", flush=True)
    return pd.concat(parts, ignore_index=True), pd.concat(snapshots, ignore_index=True), provenance


def classification_metrics(frame: pd.DataFrame, level: int) -> dict:
    if frame.empty:
        return {"states": 0, "episodes": 0, "auc": None, "ap": None}
    y = truth(frame[CEILING], level)
    p = frame[f"p_net_{level}"].to_numpy(float)
    prior = frame[f"prior_net_{level}"].to_numpy(float)
    w = _episode_day_weights(frame)
    both = len(np.unique(y)) == 2
    return {"states": len(frame), "episodes": len(frame[KEYS].drop_duplicates()),
            "positive_episodes": len(frame.loc[y, KEYS].drop_duplicates()),
            "weighted_prevalence": float(np.average(y, weights=w)),
            "auc": float(roc_auc_score(y, p, sample_weight=w)) if both else None,
            "ap": float(average_precision_score(y, p, sample_weight=w)) if both else None,
            "prior_ap": float(average_precision_score(y, prior, sample_weight=w)) if both else None,
            "brier": float(np.average((y-p)**2, weights=w)),
            "prior_brier": float(np.average((y-prior)**2, weights=w))}


def clustered_intervals(frame: pd.DataFrame, level: int) -> dict:
    """Whole first-snapshot clusters; never treat seconds as independent draws."""
    output = {}
    original_weights = _episode_day_weights(frame) if len(frame) else np.array([])
    for index, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        clusters = list(frame.groupby(columns, sort=False).indices.values())
        rng = np.random.default_rng(20264060+level+index)
        aucs, aps = [], []
        if not clusters:
            continue
        for _ in range(1000):
            choices = rng.integers(0, len(clusters), len(clusters))
            positions = np.concatenate([clusters[i] for i in choices])
            sample = frame.iloc[positions].copy()
            # Duplicate sampled groups must remain separate for day weighting.
            # Fixed first-snapshot original weights follow each sampled row.
            y = truth(sample[CEILING], level)
            if len(np.unique(y)) != 2:
                continue
            w = original_weights[positions]
            p = sample[f"p_net_{level}"].to_numpy(float)
            aucs.append(roc_auc_score(y, p, sample_weight=w))
            aps.append(average_precision_score(y, p, sample_weight=w))
        output["day" if index == 0 else "ticker_day"] = {
            "clusters": len(clusters), "draws": 1000, "two_class_draws": len(aucs),
            "auc_ci95": np.quantile(aucs, [.025, .975]).tolist() if aucs else None,
            "ap_ci95": np.quantile(aps, [.025, .975]).tolist() if aps else None,
        }
    return output


def observability_report(frame: pd.DataFrame) -> dict:
    result = {}
    for level in LEVELS:
        metrics = classification_metrics(frame, level)
        metrics["per_day"] = {str(day): classification_metrics(g, level) for day, g in frame.groupby("trading_day")}
        # Fixed score quartiles are descriptions, never selected admission rules.
        ordered = frame.sort_values([f"p_net_{level}", *KEYS, "decision_t"])
        metrics["score_quartiles"] = [classification_metrics(ordered.iloc[idx], level) for idx in np.array_split(np.arange(len(ordered)), 4)]
        below = frame.loc[frame.observed_net_return_pct.lt(level)]
        metrics["not_yet_reached_observed_target"] = classification_metrics(below, level)
        result[f"net_{level}"] = metrics
    return result


def economics_report(episodes: pd.DataFrame, decisions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    pair = episodes.merge(decisions, on=KEYS, suffixes=("", "_r304"), validate="one_to_one")
    complete = pair.reachable_complete & pair.resolved
    pair["diagnostic_category"] = "censored"
    positive_possible = pair.reachable_max_base.gt(0)
    pair.loc[complete & ~positive_possible, "diagnostic_category"] = "no_feasible_net_profit"
    pair.loc[complete & positive_possible & pair.base_net_return_pct.le(0), "diagnostic_category"] = "feasible_profit_missed"
    pair.loc[complete & positive_possible & pair.base_net_return_pct.gt(0), "diagnostic_category"] = "positive_profit_captured"
    pair["oracle_model_gap_pp"] = pair.reachable_max_base-pair.base_net_return_pct
    if (pair.loc[complete, "oracle_model_gap_pp"] < -1e-9).any():
        raise ValueError("reachable ceiling below reproduced actual return")
    def summarize(g):
        def mean(column):
            value = g[column].mean()
            return float(value) if np.isfinite(value) else None
        result = {"entries": len(g), "complete_reachable": int(g.reachable_complete.sum()),
                  "no_eligible_hold": int((~g.has_eligible_hold).sum()),
                  "categories": g.diagnostic_category.value_counts().to_dict(),
                  "actual_mean_net_pct": mean("base_net_return_pct"),
                  "oracle_mean_net_pct": mean("reachable_max_base"),
                  "mean_oracle_model_gap_pp": mean("oracle_model_gap_pp"),
                  "cost_only_stop_breaches": int(g.stop_cost_only_breach.sum()), "levels": {}}
        for path in ("reachable", "whole_path"):
            result["levels"][path] = {f"plus{level}": {
                "gross": int(g[f"{path}_max_gross_pct"].ge(level).sum()),
                "net": int(g[f"{path}_max_base"].ge(level).sum()),
            } for level in (5, 10, 20)}
        result["cost_scenario_ceiling_means"] = {name: mean(f"reachable_max_{name}") for name in ("light", "base", "stress")}
        return result
    result = summarize(pair)
    result["by_day"] = {str(day): summarize(g) for day, g in pair.groupby("trading_day")}
    result["by_category"] = {str(cat): summarize(g) for cat, g in pair.groupby("diagnostic_category")}
    return result, pair


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("states", "decisions", "raw-seconds", "output", "states-output", "episodes-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_parquet(args.states)
    source = source.loc[source.arm.eq("B")].copy().reset_index(drop=True)
    raw = pd.read_parquet(args.raw_seconds)
    validate_source(source, raw)
    contexts = {}
    for key, _ in source.groupby(KEYS, sort=False):
        opening, closing = session_limits(str(key[0]))
        bars = raw.loc[raw.trading_day.astype(str).eq(str(key[0])) & raw.ticker.astype(str).eq(str(key[1]))]
        contexts[key] = (regular_bars(bars, opening, closing), opening, closing)
    decisions = pd.read_parquet(args.decisions)
    decisions = decisions.loc[decisions.arm.eq("B_common_cap_reference")].copy()
    reproduced = replay_scored(source, contexts, "B_common_cap_reference", next_event=True)
    reproduction = verify_reproduction(reproduced, decisions)
    print("R305 exact R304 reproduction passed", flush=True)
    labeled, episodes = build_ceiling_labels(source, contexts)
    states, snapshots, folds = crossfit_diagnostics(labeled)
    economics, pair = economics_report(episodes, decisions)
    first = first_states(states)
    # Policy visitation is descriptive; fitting never depends on model visits.
    visited = states.merge(decisions[KEYS+["submission_t"]], on=KEYS, validate="many_to_one")
    visited = visited.loc[visited.decision_t.le(visited.submission_t)]
    reports = {"primary_first_snapshot_logistic": observability_report(snapshots),
               "secondary_first_state_tree": observability_report(first),
               "secondary_all_reachable_tree": observability_report(states),
               "secondary_R304_visited_tree": observability_report(visited)}
    for level in LEVELS:
        reports["primary_first_snapshot_logistic"][f"net_{level}"]["cluster_intervals"] = clustered_intervals(snapshots, level)
    checks = {"exact_R304_replay": reproduction["maximum_return_error"] <= 1e-9,
              "all86_entries_retained": len(pair) == 86,
              "all_training_days_exclude_scored_day": all(day not in f["fit_days"] for day, f in folds.items()),
              "causal_feature_columns_unchanged": True,
              "weighted_regression_residual_zero": all(abs(f["regression_offset"]["weighted_residual_after_pp"]) <= 1e-9 for f in folds.values()),
              "no_missing_terminal_labeled_negative": labeled.loc[labeled[KEYS].apply(tuple, axis=1).isin(set(map(tuple, episodes.loc[~episodes.reachable_complete, KEYS].to_numpy()))), CEILING].isna().all()}
    result = {"request_id": REQUEST_ID, "diagnostic_only": True, "promotion_eligible": False,
              "June_HOLD_opened": False, "network_requests": 0, "R304_reproduction": reproduction,
              "economics_decomposition": economics, "observability": reports, "outer_folds": folds,
              "regression_diagnostics": {name: signal(g, ADVANTAGE, "predicted_max_exit_advantage_pct") for name, g in (("all", states), ("first", first), ("R304_visited", visited))},
              "first_snapshot_support": {"entries": len(first_states(source)), "eligible_snapshots": len(snapshots), "excluded_from_learning_only": 86-len(snapshots)},
              "checks": {k: bool(v) for k, v in checks.items()}, "integrity_pass": bool(all(checks.values())),
              "limits": "Future maxima are diagnostic bounds, not executable optimal policies or learned HOLD value. First HOLD snapshots are post-entry, not an entry veto. Four reused days; upstream entry OOF not nested; sparse +10/+20 paths and correlated seconds. Next-print fills are assumptions; event mean is not account return. No threshold, feature, seed or policy tuning after results."}
    for path in (args.output, args.states_output, args.episodes_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([states.assign(diagnostic_model="states"), snapshots.assign(diagnostic_model="snapshot")], ignore_index=True).to_parquet(args.states_output, index=False)
    pair.to_parquet(args.episodes_output, index=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({"checks": result["checks"], "economics": economics["categories"], "primary_net5": reports["primary_first_snapshot_logistic"]["net_5"]}, indent=2), flush=True)
    return 0 if result["integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
