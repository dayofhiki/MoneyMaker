"""R313: timestamp-only evidence arrival with a frozen preentry score."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clock_momentum_increment import MOMENTUM, within_day_auc
from .compact_momentum_representation import fit_linear, predict_linear
from .elapsed_momentum_evidence import complete_mask, contexts_for, replay_inputs
from .feasible_upside_observability import CEILING, truth
from .frozen_momentum_may_transport import EVAL_DAYS
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .pending_exit_integrity import session_limits
from .preentry_momentum_observability import CLOCK_FEATURES, clock_features, entry_labels, metrics

REQUEST_ID = 313


def evidence_observation(bars: pd.DataFrame, hot_t: int, closing: int) -> int | None:
    ends = bars.t.to_numpy(np.int64)+1000
    # Price/volume/labels deliberately not read by this selection function.
    for index in np.flatnonzero((bars.t.to_numpy() >= hot_t) & (bars.t.to_numpy() < hot_t+300000)):
        decision = int(ends[index])
        if decision >= min(hot_t+3600000, closing):
            break
        left = int(np.searchsorted(ends, decision-10000, side="right"))
        if index+1-left >= 2:
            return decision
    return None


def later_states(cohort: pd.DataFrame, contexts: dict) -> pd.DataFrame:
    if set(cohort.trading_day.astype(str)) != set(EVAL_DAYS) or cohort.duplicated(["trading_day", "ticker", "hot_t"]).any():
        raise ValueError("fixed original May cohort required")
    rows = []
    for raw in cohort.to_dict("records"):
        key = (str(raw["trading_day"]), str(raw["ticker"]))
        closing = session_limits(key[0])[1]
        bars = contexts.get(key, (pd.DataFrame(columns=["t"]), 0, closing))[0]
        decision = evidence_observation(bars, int(raw["hot_t"]), closing)
        row = {**raw, "decision_t": np.nan, "observation_available": False,
               "label_complete": False, CEILING: np.nan, "missing_reason": "no_two_print_evidence_before5min"}
        if decision is not None:
            row.update(decision_t=decision, observation_available=True, missing_reason=None)
            row.update(entry_labels(bars, decision, min(int(raw["hot_t"])+3600000, closing)))
            if not row["label_complete"]:
                row["missing_reason"] = "incomplete_entry_or_terminal_fill_label"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame[list(CLOCK_FEATURES)] = np.nan
    available = frame.observation_available
    if available.any():
        rebuilt = clock_features(frame.loc[available], contexts)
        frame.loc[available, list(CLOCK_FEATURES)] = rebuilt[list(CLOCK_FEATURES)]
    return frame


def paired_report(original: pd.DataFrame, later: pd.DataFrame, draws: int = 1000) -> dict:
    keys = ["trading_day", "ticker", "hot_t"]
    fields = ["decision_t", "observation_available", "label_complete", CEILING, "M_p_net_5", "M_prior_net_5"]
    joined = original[keys+fields].merge(later[keys+fields], on=keys, suffixes=("_original", "_evidence"), validate="one_to_one")
    original_ok, later_ok = complete_mask(original).to_numpy(), complete_mask(later).to_numpy()
    # Cohort order is required; paired identity matching protects against silent reorder.
    pd.testing.assert_frame_equal(original[keys].reset_index(drop=True), later[keys].reset_index(drop=True))
    both = original_ok & later_ok
    reports = {}
    for name, frame in (("original", original.loc[both].copy()), ("evidence", later.loc[both].copy())):
        reports[name] = metrics(frame, "M", 5) | {"within_day_auc": within_day_auc(frame, "M")}
    first_y = truth(original[CEILING], 5) & original_ok
    later_y = truth(later[CEILING], 5) & later_ok
    delay = (joined.decision_t_evidence-joined.decision_t_original)/1000
    observed_both = joined.observation_available_original & joined.observation_available_evidence
    if (delay.loc[observed_both] < 0).any():
        raise ValueError("evidence observation preceded canonical first print")
    table = {f"original_{int(a)}_evidence_{int(b)}": int((both & (first_y == a) & (later_y == b)).sum())
             for a in (False, True) for b in (False, True)}
    per_day = {}
    for day in EVAL_DAYS:
        mask = both & original.trading_day.eq(day).to_numpy()
        per_day[day] = {"paired_labeled": int(mask.sum()), "original_positives": int((mask & first_y).sum()),
                       "evidence_positives": int((mask & later_y).sum()),
                       "arms": {name: metrics(frame.loc[mask], "M", 5) for name, frame in (("original", original), ("evidence", later))}}
    return {"sampled_episodes": len(original), "original_complete": int(original_ok.sum()),
            "evidence_observed": int(later.observation_available.sum()), "evidence_complete": int(later_ok.sum()),
            "paired_complete": int(both.sum()), "same_clock_observed": int((observed_both & delay.eq(0)).sum()),
            "later_clock_observed": int((observed_both & delay.gt(0)).sum()),
            "additional_wait_s_quantiles": {str(q): float(delay.loc[observed_both].quantile(q)) if observed_both.any() else None
                                            for q in (.1, .5, .9)},
            "missing_reasons": later.missing_reason.fillna("complete").value_counts().to_dict(),
            "paired_positive_table": table, "original_positive_total": int(first_y.sum()),
            "original_positive_no_later_observation": int((first_y & ~later.observation_available.to_numpy()).sum()),
            "original_positive_later_incomplete": int((first_y & later.observation_available.to_numpy() & ~later_ok).sum()),
            "paired_arms": reports, "per_day": per_day,
            "paired_intervals": observation_intervals(original.loc[both].copy(), later.loc[both].copy(), draws)}


def observation_intervals(original: pd.DataFrame, later: pd.DataFrame, draws: int = 1000) -> dict:
    """Paired identity draws with changed entry-time outcomes kept on each side."""
    if original.empty:
        return {}
    from sklearn.metrics import roc_auc_score
    weights = _episode_day_weights(original)
    outputs = {}
    for i, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        groups = list(original.groupby(columns, sort=False).indices.values())
        rng = np.random.default_rng(20264850+i)
        values = {"auc": [], "within_day_auc": [], "brier": []}
        for _ in range(draws):
            rows = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            w, first, evidence = weights[rows], original.iloc[rows], later.iloc[rows]
            y0, y1 = truth(first[CEILING], 5), truth(evidence[CEILING], 5)
            p0, p1 = first.M_p_net_5.to_numpy(), evidence.M_p_net_5.to_numpy()
            if len(np.unique(y0)) == len(np.unique(y1)) == 2:
                values["auc"].append(float(roc_auc_score(y1, p1, sample_weight=w)-roc_auc_score(y0, p0, sample_weight=w)))
            a0, a1 = within_day_auc(first, "M", w), within_day_auc(evidence, "M", w)
            if a0 is not None and a1 is not None:
                values["within_day_auc"].append(a1-a0)
            values["brier"].append(float(np.average((y1-p1)**2-(y0-p0)**2, weights=w)))
        outputs["day" if i == 0 else "ticker_day"] = {"clusters": len(groups), "draws": draws,
            **{f"{name}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None for name, v in values.items()},
            **{f"{name}_valid_draws": len(v) for name, v in values.items()}}
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("training", "training-raw", "evaluation", "output", "states-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    paths = [args.training/"request310-states.parquet", args.training_raw,
             args.evaluation/"request311-states.parquet", args.evaluation/"request311-cohort.parquet",
             args.evaluation/"request311-raw-seconds.parquet", args.training/"request310.json", args.evaluation/"request311.json"]
    train, original, cohort = (pd.read_parquet(paths[i]) for i in (0, 2, 3))
    def pair(frame):
        return set(map(tuple, frame[["trading_day", "ticker"]].drop_duplicates().to_numpy()))
    train_contexts = contexts_for(pd.read_parquet(paths[1]), CROSSFIT_DAYS, pair(train))
    eval_contexts = contexts_for(pd.read_parquet(paths[4]), EVAL_DAYS, pair(cohort))
    previous, official = (json.loads(paths[i].read_text()) for i in (5, 6))
    integrity = replay_inputs(train, original, cohort, train_contexts, eval_contexts, official, previous)
    model, transform, prior, fit_audit = fit_linear(train, MOMENTUM, 20264605)
    available = original.observation_available
    replay = predict_linear(original.loc[available], model, transform, prior)
    error = float(np.max(np.abs(replay-original.loc[available, "M_p_net_5"].to_numpy())))
    if error > 1e-9:
        raise ValueError("frozen R311 M probabilities failed replay")
    evidence = later_states(cohort, eval_contexts)
    evidence["M_p_net_5"] = np.nan
    available = evidence.observation_available
    if available.any():
        evidence.loc[available, "M_p_net_5"] = predict_linear(evidence.loc[available], model, transform, prior)
    evidence["M_prior_net_5"] = prior
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
              "June_HOLD_opened": False, "final_July_August_opened": False, "network_market_requests": 0,
              "entries_retuned": False, "original_M_fit": fit_audit, "packages": {"M": list(MOMENTUM)},
              "integrity": integrity | {"maximum_frozen_M_replay_error": error},
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "comparison": paired_report(original, evidence),
              "limitations": "Timestamp-only two-print10s evidence contract, not a learned or forced wait policy. Reused May development. Paired metrics condition on both resolved labels; entry-time labels differ, so gains are not pure same-target effects. Missing identities and lost first opportunities remain visible. No model/package/delay/threshold search, live controller or realized profit claim."}
    saved = evidence.copy()
    for name in ("decision_t", "observation_available", "label_complete", CEILING, "M_p_net_5"):
        saved["original_"+name] = original[name].to_numpy()
    for path in (args.output, args.states_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    saved.to_parquet(args.states_output, index=False, compression="zstd")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["comparison"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
