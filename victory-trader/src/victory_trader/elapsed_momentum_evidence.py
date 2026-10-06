"""R312: fixed temporal-support intervention with chronological first-state fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .clock_momentum_increment import CONTROL, MOMENTUM, SIGNAL, within_day_auc
from .compact_momentum_representation import fit_linear, predict_linear
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS, first_observation, identity_hash
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .pending_exit_integrity import regular_bars, session_limits
from .preentry_momentum_observability import CLOCK_FEATURES, clock_features, entry_labels, metrics, snapshots, validate_states

REQUEST_ID = 312
SEED = 20264705
WINDOWS = (10, 30, 120)
SUPPORT = ("evidence_previous_interprint_s",) + tuple(
    f"evidence_{name}_{window}s" for window in WINDOWS
    for name in ("baseline_age_s", "internal_span_s", "maximum_interprint_s")
)
DIRECTION = tuple(f"evidence_{name}_{window}s" for window in WINDOWS
                  for name in ("baseline_log_rate", "internal_log_rate")) + ("evidence_acceleration_10_30",)
PACKAGES = {"M": MOMENTUM, "K": CONTROL + SUPPORT, "T": CONTROL + SUPPORT + SIGNAL + DIRECTION}
ARMS = tuple(PACKAGES)
CONTRASTS = (("T", "M"), ("T", "K"), ("K", "M"))


def contexts_for(raw: pd.DataFrame, allowed_days: tuple[str, ...], allowed_pairs: set) -> dict:
    if not set(raw.trading_day.astype(str)).issubset(allowed_days):
        raise ValueError("raw data crossed fixed May dates")
    pairs = set(map(tuple, raw[["trading_day", "ticker"]].drop_duplicates().to_numpy()))
    if not pairs.issubset(allowed_pairs):
        raise ValueError("raw data crossed declared identity scope")
    result = {}
    for key, bars in raw.groupby(["trading_day", "ticker"], sort=False):
        opening, closing = session_limits(str(key[0]))
        ordered = regular_bars(bars, opening, closing)
        if len(ordered) != len(bars):
            raise ValueError("raw checkpoint contains nonregular bars")
        result[key] = (ordered, opening, closing)
    return result


def temporal_features(source: pd.DataFrame, contexts: dict) -> pd.DataFrame:
    """No labels or future prints enter timestamp support or elapsed direction."""
    result = source.copy()
    names = SUPPORT + DIRECTION
    result[list(names)] = np.nan
    for key, group in result.groupby(["trading_day", "ticker"], sort=False):
        if key not in contexts:
            raise ValueError("missing raw history for an available observation")
        bars = contexts[key][0]
        ends = bars.t.to_numpy(np.int64) + 1000
        closes = bars.c.to_numpy(float)
        if np.any(np.diff(ends) <= 0) or not np.isfinite(closes).all() or (closes <= 0).any():
            raise ValueError("sorted unique seconds and positive finite prices required")
        added = []
        for row in group.itertuples():
            decision = int(row.decision_t)
            right = int(np.searchsorted(ends, decision, side="right"))
            if right == 0:
                raise ValueError("available observation has no completed price")
            out = dict.fromkeys(names, np.nan)
            current_end, current = ends[right-1], closes[right-1]
            if right >= 2:
                out[SUPPORT[0]] = float((current_end-ends[right-2])/1000)
            for window in WINDOWS:
                left = int(np.searchsorted(ends, decision-window*1000, side="right"))
                count = right-left
                if count:
                    span = float((current_end-ends[left])/1000)
                    out[f"evidence_internal_span_s_{window}s"] = span
                    if count >= 2:
                        out[f"evidence_internal_log_rate_{window}s"] = float(100*np.log(current/closes[left])/span)
                if left:
                    age = float((decision-ends[left-1])/1000)
                    elapsed = float((current_end-ends[left-1])/1000)
                    out[f"evidence_baseline_age_s_{window}s"] = age
                    if elapsed > 0:
                        out[f"evidence_baseline_log_rate_{window}s"] = float(100*np.log(current/closes[left-1])/elapsed)
                    gaps = np.diff(ends[left-1:right])
                    if len(gaps):
                        out[f"evidence_maximum_interprint_s_{window}s"] = float(gaps.max()/1000)
            short, long = (out[f"evidence_baseline_log_rate_{w}s"] for w in (10, 30))
            if np.isfinite(short) and np.isfinite(long):
                out[DIRECTION[-1]] = short-long
            added.append(out)
        result.loc[group.index, list(names)] = pd.DataFrame(added, index=group.index)[list(names)]
    return result


def replay_inputs(train: pd.DataFrame, evaluation: pd.DataFrame, cohort: pd.DataFrame,
                  train_contexts: dict, eval_contexts: dict, official: dict, previous: dict) -> dict:
    validate_states(train)
    if previous.get("request_id") != 310 or official.get("request_id") != 311:
        raise ValueError("official R310/R311 provenance required")
    if tuple(previous["packages"]["M"]) != MOMENTUM or tuple(official["packages"]["M"]) != MOMENTUM:
        raise ValueError("original momentum package changed")
    if not train.label_complete.all() or not np.isfinite(train[CEILING]).all():
        raise ValueError("complete original training labels required")
    if set(evaluation.trading_day.astype(str)) != set(EVAL_DAYS) or evaluation.duplicated(KEYS).any():
        raise ValueError("unique original eight-day cohort required")
    pd.testing.assert_frame_equal(evaluation[list(cohort)].reset_index(drop=True), cohort.reset_index(drop=True),
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    hashes = [identity_hash(str(r.trading_day), str(r.ticker), int(r.hot_t)) for r in cohort.itertuples()]
    if not np.array_equal(hashes, cohort.sample_hash_byte.to_numpy()) or np.any(np.asarray(hashes) >= 64):
        raise ValueError("original outcome-independent cohort changed")
    for frame, contexts in ((train, train_contexts), (evaluation.loc[evaluation.observation_available], eval_contexts)):
        replay = clock_features(frame.drop(columns=list(CLOCK_FEATURES)), contexts)
        pd.testing.assert_frame_equal(replay[list(CLOCK_FEATURES)], frame[list(CLOCK_FEATURES)],
                                      check_dtype=False, atol=1e-9, rtol=1e-12)
    for row in evaluation.itertuples():
        key = (str(row.trading_day), str(row.ticker))
        bars, _, closing = eval_contexts.get(key, (pd.DataFrame(columns=["t"]), 0, session_limits(key[0])[1]))
        decision = first_observation(bars, int(row.hot_t), closing)
        if (decision is not None) != bool(row.observation_available):
            raise ValueError("canonical first-observation availability changed")
        if decision is not None and decision != row.decision_t:
            raise ValueError("canonical first-observation clock changed")
        if decision is not None:
            label = entry_labels(bars, decision, min(int(row.hot_t)+3600000, closing))
            if label["label_complete"] != row.label_complete or not np.isclose(
                    label[CEILING], getattr(row, CEILING), atol=1e-9, rtol=0, equal_nan=True):
                raise ValueError("original risk/cost label changed")
    counts = {"sampled_episodes": len(evaluation), "observed_episodes": int(evaluation.observation_available.sum()),
              "labeled_episodes": int(complete_mask(evaluation).sum())}
    if any(counts[k] != official["transport"][k] for k in counts):
        raise ValueError("original coverage changed")
    return {**counts, "cohort_clock_and_label_replay": True, "all_clock_inputs_replay": True}


def complete_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.observation_available.astype(bool) & frame.label_complete.astype(bool) & np.isfinite(frame[CEILING])


def score_fit(train: pd.DataFrame, held: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if train.empty:
        raise ValueError("nonempty complete training support required")
    scored, audit = held.copy(), {}
    available = scored.observation_available.astype(bool)
    for arm, features in PACKAGES.items():
        model, transform, prior, fit_audit = fit_linear(train, features, SEED)
        scored[f"{arm}_p_net_5"] = np.nan
        if available.any():
            scored.loc[available, f"{arm}_p_net_5"] = predict_linear(scored.loc[available], model, transform, prior)
        scored[f"{arm}_prior_net_5"] = prior
        audit[arm] = fit_audit
    first = snapshots(train, 0)
    scored["first_training_prior"] = float(np.average(truth(first[CEILING], 5), weights=_episode_day_weights(first)))
    return scored, audit


def expanding_score(original_first: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if set(original_first.trading_day.astype(str)) != set(CROSSFIT_DAYS) or original_first.duplicated(KEYS).any():
        raise ValueError("unique original May5-8 first support required")
    if set(evaluation.trading_day.astype(str)) != set(EVAL_DAYS) or evaluation.duplicated(KEYS).any():
        raise ValueError("unique fixed May evaluation support required")
    pieces, folds = [], {}
    for day in EVAL_DAYS:
        history = evaluation.loc[evaluation.trading_day.lt(day) & complete_mask(evaluation)]
        fitting = pd.concat([original_first, history], ignore_index=True)
        fit_days = sorted(fitting.trading_day.astype(str).unique())
        if any(d >= day for d in fit_days) or not set(fit_days).issubset((*CROSSFIT_DAYS, *EVAL_DAYS)):
            raise ValueError("chronological fitting crossed current/later/sealed day")
        held = evaluation.loc[evaluation.trading_day.eq(day)].copy()
        scored, audit = score_fit(fitting, held)
        folds[day] = {"fit_days": fit_days, "original_first_episodes": len(original_first),
                      "earlier_broad_first_episodes": len(history), "models": audit}
        pieces.append(scored)
        print(f"R312 expanding-first completed {day}, training episodes={len(fitting)}", flush=True)
    return pd.concat(pieces, ignore_index=True), folds


def paired_intervals(frame: pd.DataFrame, draws: int = 1000) -> dict:
    if frame.empty:
        return {}
    y, weights = truth(frame[CEILING], 5), _episode_day_weights(frame)
    probabilities = {a: frame[f"{a}_p_net_5"].to_numpy(float) for a in ARMS}
    output = {}
    for i, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        clusters = list(frame.groupby(columns, sort=False).indices.values())
        values = {f"{a}_minus_{b}": {"auc": [], "within_day_auc": [], "brier": []} for a, b in CONTRASTS}
        rng = np.random.default_rng(20264750+i)
        for _ in range(draws):
            indices = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
            sample, w = frame.iloc[indices], weights[indices]
            both = len(np.unique(y[indices])) == 2
            auc = {a: roc_auc_score(y[indices], probabilities[a][indices], sample_weight=w) for a in ARMS} if both else {}
            within = {a: within_day_auc(sample, a, w) for a in ARMS}
            for a, b in CONTRASTS:
                v = values[f"{a}_minus_{b}"]
                if both:
                    v["auc"].append(float(auc[a]-auc[b]))
                if within[a] is not None:
                    v["within_day_auc"].append(float(within[a]-within[b]))
                v["brier"].append(float(np.average((y[indices]-probabilities[a][indices])**2 -
                                                 (y[indices]-probabilities[b][indices])**2, weights=w)))
        output["day" if i == 0 else "ticker_day"] = {"clusters": len(clusters), "draws": draws,
            "comparisons": {name: {f"{metric}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None
                                    for metric, v in report.items()} |
                                   {f"{metric}_valid_draws": len(v) for metric, v in report.items()}
                            for name, report in values.items()}}
    return output


def report(frame: pd.DataFrame) -> dict:
    labeled = frame.loc[complete_mask(frame)].copy()
    def rank(group, arm):
        return metrics(group, arm, 5) | {"within_day_auc": within_day_auc(group, arm)}
    per_day = {day: {"sampled": int(frame.trading_day.eq(day).sum()),
                     "labeled": int(labeled.trading_day.eq(day).sum()),
                     "positive_episodes": int(truth(labeled.loc[labeled.trading_day.eq(day), CEILING], 5).sum()),
                     "arms": {a: rank(labeled.loc[labeled.trading_day.eq(day)], a) for a in ARMS}}
               for day in EVAL_DAYS}
    return {"sampled_episodes": len(frame), "observed_episodes": int(frame.observation_available.sum()),
            "labeled_episodes": len(labeled), "complete_coverage": len(labeled)/len(frame) if len(frame) else 0.,
            "missing_reasons": frame.missing_reason.fillna("complete").value_counts().to_dict(),
            "arms": {a: rank(labeled, a) for a in ARMS}, "per_day": per_day,
            "first_training_prior_brier": float(np.average((truth(labeled[CEILING], 5)-labeled.first_training_prior.to_numpy())**2,
                                                          weights=_episode_day_weights(labeled))) if len(labeled) else None,
            "paired_intervals": paired_intervals(labeled)}


def gate(result: dict) -> dict:
    t, m, k = (result["arms"][a] for a in ("T", "M", "K"))
    def greater(a, b):
        return a is not None and b is not None and a > b
    comparisons = result.get("paired_intervals", {}).get("ticker_day", {}).get("comparisons", {})
    days = [r for r in result["per_day"].values() if r["arms"]["M"]["auc"] is not None]
    checks = {"complete_coverage_at_least90pct": result["complete_coverage"] >= .9,
              "at_least20_positive_episodes": t.get("positive_episodes", 0) >= 20,
              "at_least4_positive_days": sum(r["positive_episodes"] > 0 for r in result["per_day"].values()) >= 4,
              "T_within_day_auc_beats_M_and_K": all(greater(t["within_day_auc"], r["within_day_auc"]) for r in (m, k)),
              "T_ap_beats_M_and_K": all(greater(t["ap"], r["ap"]) for r in (m, k)),
              "T_brier_beats_M_K_and_prior": all(greater(v, t.get("brier")) for v in
                                                 (m.get("brier"), k.get("brier"), result["first_training_prior_brier"])),
              "T_M_auc_improves_majority_days": bool(days) and sum(greater(r["arms"]["T"]["auc"], r["arms"]["M"]["auc"]+1e-9)
                                                                    for r in days) > len(days)/2}
    for name in ("T_minus_M", "T_minus_K"):
        ci = comparisons.get(name, {}).get("within_day_auc_ci95")
        checks[f"ticker_day_{name}_within_day_gain_ci_positive"] = ci is not None and ci[0] > 0
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("training", "training-raw", "evaluation", "output", "states-output", "training-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    train_path = args.training/"request310-states.parquet"
    eval_path = args.evaluation/"request311-states.parquet"
    raw_path = args.evaluation/"request311-raw-seconds.parquet"
    cohort_path = args.evaluation/"request311-cohort.parquet"
    previous_path, official_path = args.training/"request310.json", args.evaluation/"request311.json"
    train, evaluation, cohort = (pd.read_parquet(p) for p in (train_path, eval_path, cohort_path))
    previous, official = (json.loads(p.read_text()) for p in (previous_path, official_path))
    train_contexts = contexts_for(pd.read_parquet(args.training_raw), CROSSFIT_DAYS,
                                 set(map(tuple, train[["trading_day", "ticker"]].drop_duplicates().to_numpy())))
    eval_contexts = contexts_for(pd.read_parquet(raw_path), EVAL_DAYS,
                                set(map(tuple, cohort[["trading_day", "ticker"]].to_numpy())))
    integrity = replay_inputs(train, evaluation, cohort, train_contexts, eval_contexts, official, previous)
    train = temporal_features(train, train_contexts)
    observed = temporal_features(evaluation.loc[evaluation.observation_available].copy(), eval_contexts)
    expanded = evaluation.copy()
    expanded[list(SUPPORT+DIRECTION)] = np.nan
    expanded.loc[observed.index, list(SUPPORT+DIRECTION)] = observed[list(SUPPORT+DIRECTION)]
    frozen, frozen_audit = score_fit(train, expanded)
    available = expanded.observation_available
    error = float(np.max(np.abs(frozen.loc[available, "M_p_net_5"]-evaluation.loc[available, "M_p_net_5"])))
    if error > 1e-9:
        raise ValueError("frozen M failed official R311 score replay")
    chronological, folds = expanding_score(snapshots(train, 0).assign(observation_available=True), expanded)
    reports = {"frozen": report(frozen), "expanding_first": report(chronological)}
    checks = gate(reports["expanding_first"])
    scored = expanded.copy()
    for track, frame in (("frozen", frozen), ("expanding_first", chronological)):
        frame = frame.set_index(KEYS).reindex(scored.set_index(KEYS).index)
        for column in [*[f"{a}_{kind}_net_5" for a in ARMS for kind in ("p", "prior")], "first_training_prior"]:
            scored[f"{track}_{column}"] = frame[column].to_numpy()
    inputs = (train_path, args.training_raw, eval_path, raw_path, cohort_path, previous_path, official_path)
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
              "June_HOLD_opened": False, "final_July_August_opened": False, "network_market_requests": 0,
              "entries_retuned": False, "seed": SEED, "packages": {a: list(v) for a, v in PACKAGES.items()},
              "primary_contrast": "expanding_first T minus M same-day first-observation AUROC",
              "references": {"R310_run": 37218791436, "R306_raw_run": 37211254516, "R311_run": 37223064559},
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              "integrity": integrity | {"maximum_frozen_M_replay_error": error},
              "frozen_training": frozen_audit, "expanding_folds": folds, "reports": reports,
              "checks": checks, "research_gate_pass": all(checks.values()),
              "limitations": "Reused May development with incomplete coverage. Frozen input intervention isolates package change; expanding support also changes first/all-state weighting and population size. Chronological fits exclude current/later dates, but prior episodes overlap ticker identities. Bootstrap is conditional on fitted predictions, not a refit uncertainty estimate. Direction contrast includes missingness and nominal signals. Reachable net5 is hindsight under unchanged costs/stops/fills, not realized profits. No policy/threshold selection or sealed data access."}
    for path in (args.output, args.states_output, args.training_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.training_output, index=False, compression="zstd")
    scored.to_parquet(args.states_output, index=False, compression="zstd")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"checks": checks, "arms": {k: r["arms"] for k, r in reports.items()}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
