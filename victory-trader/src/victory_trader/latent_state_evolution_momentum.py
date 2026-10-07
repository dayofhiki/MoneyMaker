"""R324: prediction-only recursion over latent remaining-opportunity labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .canonical_population_momentum import verify_checkpoint
from .compact_momentum_representation import fit_transform, independent_weights
from .elapsed_momentum_evidence import complete_mask
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS
from .lagged_minute_context import CROSSFIT_DAYS
from .nearby_state_momentum import LocalCache, local_manifest
from .state_evolution_momentum import HISTORY, PACKAGES as PREVIOUS_PACKAGES, RankCache, metrics
from .within_episode_momentum import frozen_transform, metrics as raw_metrics

REQUEST_ID = 324
GAP = "state_gap_log_s"
PACKAGES = {"F": PREVIOUS_PACKAGES["T"]+(GAP,), "H": PREVIOUS_PACKAGES["G"]+(GAP,), "A": HISTORY+(GAP,)}
ARMS = ("F", "H", "A", "K", "B", "B0")
METRICS = ("between_episode_same_day_auc", "auc", "ap", "brier", "within_episode_auc", "local_pair_rank")
CONTRASTS = (("F", "H"), ("F", "A"), ("F", "K"), ("F", "B"), ("F", "B0"), ("H", "A"))
PARAMETERS = dict(C=.1, l1_ratio=0., solver="lbfgs", fit_intercept=True, max_iter=2000, tol=1e-8, random_state=20265805)


def causal_states(states: pd.DataFrame) -> pd.DataFrame:
    if states.empty or not states.index.is_unique or states.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("nonempty unique observed states required")
    if states.decision_t.isna().any() or not set(states.trading_day).issubset((*CROSSFIT_DAYS, *EVAL_DAYS)):
        raise ValueError("fixed causal clocks/dates required")
    result = states.copy()
    result["state_previous_t"] = np.nan
    result[GAP] = np.nan
    for _, group in result.groupby(KEYS, sort=False):
        ordered = group.sort_values("decision_t")
        times = ordered.decision_t.to_numpy(np.int64)
        if (np.diff(times) <= 0).any():
            raise ValueError("strictly increasing episode clocks required")
        result.loc[ordered.index[1:], "state_previous_t"] = times[:-1]
        result.loc[ordered.index[1:], GAP] = np.log1p(np.diff(times)/1000)
    return result


def transition_manifest(training: pd.DataFrame):
    if training.empty or not set(training.trading_day).issubset(CROSSFIT_DAYS):
        raise ValueError("fixed original training dates required")
    causal_states(training)  # Validate original identities before adjacency is enumerated.
    rows, counts = [], []
    for key, group in training.groupby(KEYS, sort=False):
        ordered = group.sort_values("decision_t")
        available = complete_mask(ordered).to_numpy()
        labels = truth(ordered[CEILING], 5)
        indices = ordered.index.to_numpy()
        count = 0
        for j in range(1, len(ordered)):
            if available[j-1] and available[j]:
                rows.append(dict(zip(KEYS, key)) | {"previous_index": int(indices[j-1]), "current_index": int(indices[j]),
                    "previous_label": int(labels[j-1]), "current_label": int(labels[j]),
                    "gap_ms": int(ordered.decision_t.iloc[j]-ordered.decision_t.iloc[j-1])})
                count += 1
        counts.append(dict(zip(KEYS, key)) | {"observed_states": len(ordered), "complete_states": int(available.sum()), "transitions": count})
    manifest = pd.DataFrame(rows, columns=[*KEYS, "previous_index", "current_index", "previous_label", "current_label", "gap_ms"])
    manifest["transform_weight"] = pd.Series(dtype=float)
    manifest["conditional_weight"] = pd.Series(dtype=float)
    if not manifest.empty:
        currents = training.loc[manifest.current_index]
        manifest["transform_weight"] = independent_weights(currents)
        for previous in (0, 1):
            mask = manifest.previous_label.eq(previous).to_numpy()
            if mask.any():
                manifest.loc[mask, "conditional_weight"] = independent_weights(currents.loc[mask])
    return manifest, pd.DataFrame(counts)


def transition_support(manifest: pd.DataFrame):
    return {"rows": len(manifest), "episodes": len(manifest[KEYS].drop_duplicates()),
        "cells": {f"{p}_to_{c}": {"rows": len(g), "episodes": len(g[KEYS].drop_duplicates()), "dates": sorted(g.trading_day.unique())}
                  for (p, c), g in manifest.groupby(["previous_label", "current_label"])},
        "conditional": {str(p): {"rows": len(g), "episodes": len(g[KEYS].drop_duplicates())} for p, g in manifest.groupby("previous_label")}}


def fit_transitions(training: pd.DataFrame, manifest: pd.DataFrame):
    if manifest.empty or not set(training.trading_day).issubset(CROSSFIT_DAYS):
        raise ValueError("nonempty preceding transition support required")
    reproduced, _ = transition_manifest(training)
    verify_checkpoint(manifest.reset_index(drop=True), reproduced)
    currents = training.loc[manifest.current_index]
    transform_weights = independent_weights(currents)
    bundles, audits = {}, {}
    for arm, features in PACKAGES.items():
        transform = fit_transform(currents, features, transform_weights)
        bundles[arm], conditional = {}, {}
        for previous in (0, 1):
            mask = manifest.previous_label.eq(previous).to_numpy()
            selected = currents.loc[mask]
            y = manifest.current_label.to_numpy()[mask]
            if selected.empty:
                bundles[arm][previous] = (None, transform, None)
                conditional[str(previous)] = {"no_support": True, "rows": 0, "prior": None, "coefficients": None}
                continue
            weights = independent_weights(selected)
            prior = float(np.average(y, weights=weights))
            model = None
            if len(np.unique(y)) == 2:
                model = LogisticRegression(**PARAMETERS)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(transform.apply(selected), y, sample_weight=weights)
            bundles[arm][previous] = (model, transform, prior)
            conditional[str(previous)] = {"no_support": False, "single_class_fallback": model is None, "rows": len(selected),
                "episodes": len(selected[KEYS].drop_duplicates()), "fit_days": sorted(selected.trading_day.unique()),
                "weight_sum": float(weights.sum()), "positive_rows": int(y.sum()), "prior": prior,
                "coefficients": model.coef_[0].tolist() if model is not None else None,
                "intercept": float(model.intercept_[0]) if model is not None else None,
                "iterations": int(model.n_iter_.max()) if model is not None else None}
        audits[arm] = {"fit_days": sorted(currents.trading_day.unique()), "parameters": PARAMETERS.copy(),
            "preprocessing": transform.audit(), "transform_states": len(currents),
            "transform_episodes": len(currents[KEYS].drop_duplicates()), "transform_weight_sum": float(transform_weights.sum()),
            "conditional": conditional}
    return bundles, audits


def frozen_initial(frame: pd.DataFrame, audit: dict):
    if not audit["fit_days"] or max(audit["fit_days"]) >= frame.trading_day.min():
        raise ValueError("initial model must use strictly preceding dates")
    transform = frozen_transform(audit["preprocessing"])
    values = transform.apply(frame)
    if audit["single_class_fallback"]:
        p = np.full(len(frame), audit["prior"])
    else:
        p = expit(values@np.asarray(audit["coefficients"])+audit["intercept"])
    error = float(np.max(np.abs(p-frame.G_p_net_5.to_numpy(float)))) if "G_p_net_5" in frame else None
    if error is not None and error > 1e-9:
        raise ValueError("frozen initial/current G score replay changed")
    return p, error


def infer(states: pd.DataFrame, bundles: dict, initial_audit: dict):
    """Requires causal features only: no labels, complete mask or outcomes are read."""
    result = causal_states(states)
    initial, replay_error = frozen_initial(result, initial_audit)
    result["R324_B_p_net_5"] = initial
    first = result.assign(_initial=initial).sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    first_map = first.set_index(KEYS)._initial.to_dict()
    result["R324_B0_p_net_5"] = [first_map[k] for k in result[KEYS].itertuples(index=False, name=None)]
    for arm in (*PACKAGES, "K"):
        q = {}
        for previous in (0, 1):
            model, transform, prior = bundles["F" if arm == "K" else arm][previous]
            if prior is None:
                q[previous] = np.full(len(result), np.nan)
            elif arm == "K" or model is None:
                q[previous] = np.full(len(result), prior)
            else:
                q[previous] = model.predict_proba(transform.apply(result))[:, 1]
            result[f"R324_{arm}_q{previous}1"] = q[previous]
        probabilities = np.full(len(result), np.nan)
        for _, group in result.groupby(KEYS, sort=False):
            ordered = group.sort_values("decision_t")
            positions = result.index.get_indexer(ordered.index)
            probabilities[positions[0]] = initial[positions[0]]
            for j in range(1, len(positions)):
                current, previous = positions[j], positions[j-1]
                p = probabilities[previous]
                probabilities[current] = (1-p)*q[0][current]+p*q[1][current]
        result[f"R324_{arm}_p_net_5"] = probabilities
    for arm in ARMS:
        result[f"R324_{arm}_prior_net_5"] = initial_audit["prior"]
    return result, {"maximum_saved_G_score_replay_error": replay_error,
        "unscored_observations": {a: int(result[f"R324_{a}_p_net_5"].isna().sum()) for a in ARMS}}


def probability_frame(frame: pd.DataFrame):
    columns = [f"R324_{a}_{s}" for a in ARMS for s in ("p_net_5", "prior_net_5")]
    return frame[[*KEYS, "decision_t", "observation_available", "label_complete", CEILING, *columns]].rename(columns={c: c.removeprefix("R324_") for c in columns})


def raw_frame(frame: pd.DataFrame):
    result = frame[[*KEYS, "decision_t", "observation_available", "label_complete", CEILING]].copy()
    for a in ARMS:
        result[f"{a}_score"] = frame[f"R324_{a}_p_net_5"]
    return result


def report_metrics(frame: pd.DataFrame, pairs: pd.DataFrame):
    finite = np.isfinite(frame[[f"R324_{a}_p_net_5" for a in ARMS]].to_numpy(float)).all(axis=1)
    ranked = frame.loc[finite]
    usable_pairs = pairs.loc[pairs.positive_index.isin(ranked.index) & pairs.negative_index.isin(ranked.index)].reset_index(drop=True)
    result = metrics(probability_frame(ranked), ARMS)
    nearby = LocalCache(raw_frame(ranked), usable_pairs, ARMS).evaluate()
    return {a: result[a] | nearby[a] for a in ARMS}


def intervals(frame: pd.DataFrame, pairs: pd.DataFrame, draws=1000):
    complete = frame.loc[complete_mask(frame)]
    cache, local = RankCache(probability_frame(complete), ARMS), LocalCache(raw_frame(complete), pairs, ARMS)
    output = {}
    for scheme, columns, seed in (("day", ["trading_day"], 20265850), ("ticker_day", ["trading_day", "ticker"], 20265851)):
        groups = list(complete.groupby(columns, sort=False).indices.values())
        ids = np.zeros(len(complete), int)
        for j, index in enumerate(groups):
            ids[index] = j
        rng = np.random.default_rng(seed)
        values = {f"{a}_minus_{b}": {m: [] for m in METRICS} for a, b in CONTRASTS}
        for _ in range(draws):
            counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            m = counts[ids]
            ranked, nearby = cache.evaluate(m), local.evaluate(m)
            for a, b in CONTRASTS:
                for metric in METRICS:
                    scored = nearby if metric == "local_pair_rank" else ranked
                    x, y = scored[a][metric], scored[b][metric]
                    if x is not None and y is not None:
                        values[f"{a}_minus_{b}"][metric].append(x-y)
        output[scheme] = {"clusters": len(groups), "draws": draws, "comparisons": {
            c: {f"{m}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None for m, v in item.items()} |
               {f"{m}_valid_draws": len(v) for m, v in item.items()} for c, item in values.items()}}
        print(f"R324 {scheme} bootstrap complete", flush=True)
    return output


def gate(report):
    a, cells, rank = report["arms"], report["training_support"]["cells"], "between_episode_same_day_auc"
    def greater(x, y):
        return x is not None and y is not None and x > y
    checks = {"at_least20_onset_training_episodes": cells.get("0_to_1", {}).get("episodes", 0) >= 20,
        "at_least20_decay_training_episodes": cells.get("1_to_0", {}).get("episodes", 0) >= 20,
        "onset_decay_at_least3_training_dates": all(len(cells.get(c, {}).get("dates", [])) >= 3 for c in ("0_to_1", "1_to_0")),
        "at_least200_transition_training_episodes": report["training_support"]["episodes"] >= 200,
        "at_least50_mixed_evaluation_episodes": a["F"]["within_episode_evaluable_episodes"] >= 50,
        "at_least30_local_evaluation_episodes": a["F"]["local_evaluable_episodes"] >= 30,
        "at_least4_positive_evaluation_dates": sum(v["positive_episodes"] > 0 for v in report["per_day"].values()) >= 4,
        "F_admission_beats_H_A_K_B_B0": all(greater(a["F"][rank], a[x][rank]) for x in ("H", "A", "K", "B", "B0")),
        "F_timing_beats_H_A_K_B_and_half": all(greater(a["F"]["within_episode_auc"], v) for v in [a[x]["within_episode_auc"] for x in ("H", "A", "K", "B")]+[.5]),
        "F_local_beats_H_A_K_B_and_half": all(greater(a["F"]["local_pair_rank"], v) for v in [a[x]["local_pair_rank"] for x in ("H", "A", "K", "B")]+[.5]),
        "F_brier_beats_H_A_K_B_B0_and_prior": all(greater(v, a["F"]["brier"]) for v in [a[x]["brier"] for x in ("H", "A", "K", "B", "B0")]+[a["F"]["prior_brier"]])}
    days = [v["arms"] for v in report["per_day"].values() if v["arms"]["B"][rank] is not None]
    checks["F_B_admission_improves_majority_dates"] = bool(days) and sum(greater(v["F"][rank], v["B"][rank]) for v in days) > len(days)/2
    for contrast, metric, positive in (("F_minus_B", rank, True), ("F_minus_B", "within_episode_auc", True),
                                      ("F_minus_B", "local_pair_rank", True), ("F_minus_H", rank, True),
                                      ("F_minus_A", rank, True), ("F_minus_B", "brier", False)):
        for scheme in ("day", "ticker_day"):
            ci = report["paired_intervals"][scheme]["comparisons"][contrast][f"{metric}_ci95"]
            checks[f"{scheme}_{contrast}_{metric}_ci_{'positive' if positive else 'negative'}"] = ci is not None and (ci[0] > 0 if positive else ci[1] < 0)
    return checks


def chronological(training, saved, prior):
    pieces, folds = [], {}
    for day in CROSSFIT_DAYS[1:]:
        preceding = training.loc[training.trading_day.lt(day)]
        manifest, _ = transition_manifest(preceding)
        bundles, fits = fit_transitions(preceding, manifest)
        held = saved.loc[saved.trading_day.eq(day)]
        scored, audit = infer(held, bundles, prior["chronological_training"]["folds"][day]["G"])
        pieces.append(scored)
        folds[day] = {"fits": fits, "inference": audit, "training_support": transition_support(manifest)}
    return pd.concat(pieces, ignore_index=True), folds


def run(previous: Path, pinned: Path, output: Path):
    hashes = json.loads(pinned.read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((previous/name).read_bytes()).hexdigest() != digest:
            raise ValueError("preregistered input hash changed: "+name)
    prior = json.loads((previous/"official319/request319.json").read_text())
    if prior["request_id"] != 319 or prior["market_requests"] != 0:
        raise ValueError("official cached R319 provenance required")
    training = causal_states(pd.read_parquet(previous/"official319/request319-training-states.parquet").reset_index(drop=True))
    evaluation = pd.read_parquet(previous/"official321/request321-states.parquet")
    ledger = pd.read_parquet(previous/"official319/request319-episode-ledger.parquet")
    manifest = pd.read_parquet(previous/"official319/request319-clock-manifest.parquet")
    verify_checkpoint(manifest, evaluation[list(manifest)])
    local_pairs = pd.read_parquet(previous/"official321/request321-local-pair-manifest.parquet")
    local_ledger = pd.read_parquet(previous/"official321/request321-local-pair-ledger.parquet")
    replay_pairs, replay_ledger = local_manifest(evaluation, ledger)
    verify_checkpoint(local_pairs, replay_pairs)
    verify_checkpoint(local_ledger, replay_ledger)
    transitions, transition_ledger = transition_manifest(training)
    support = transition_support(transitions)
    if len(training) != 9325 or len(evaluation) != 19530 or len(ledger) != 828 or len(transitions) != 8445 or support["episodes"] != 286 or len(local_pairs) != 8623 or set(evaluation.trading_day) != set(EVAL_DAYS):
        raise ValueError("registered support changed")
    if {k: v["rows"] for k, v in support["cells"].items()} != {"0_to_0": 7553, "0_to_1": 88, "1_to_0": 99, "1_to_1": 705}:
        raise ValueError("registered transition label support changed")
    output.mkdir(parents=True, exist_ok=True)
    transitions.to_parquet(output/"request324-transition-manifest.parquet", index=False, compression="zstd")
    transition_ledger.to_parquet(output/"request324-transition-ledger.parquet", index=False, compression="zstd")
    print("R324 original adjacent transitions persisted before fits", flush=True)
    bundles, fits = fit_transitions(training, transitions)
    scored, inference = infer(evaluation, bundles, prior["fits"]["G"])
    if any(inference["unscored_observations"].values()):
        raise ValueError("full evaluation contains unsupported inference")
    verify_checkpoint(evaluation, scored[list(evaluation)])
    saved_chrono = pd.read_parquet(previous/"official321/request321-chronological-states.parquet")
    chrono, folds = chronological(training, saved_chrono, prior)
    verify_checkpoint(saved_chrono, chrono[list(saved_chrono)])
    first = scored.sort_values("decision_t").groupby(KEYS, sort=False).head(1).reset_index(drop=True)
    for suffix, frame in (("training-states", training), ("states", scored), ("chronological-states", chrono),
                          ("first-states", first), ("episode-ledger", ledger)):
        frame.to_parquet(output/f"request324-{suffix}.parquet", index=False, compression="zstd")
    report = {"training_support": support, "observed_states": len(scored), "complete_states": int(complete_mask(scored).sum()),
        "complete_evaluation_episodes": len(scored.loc[complete_mask(scored), KEYS].drop_duplicates()), "ledger_episodes": len(ledger),
        "arms": report_metrics(scored, local_pairs),
        "per_day": {d: {"arms": report_metrics(scored.loc[scored.trading_day.eq(d)], local_pairs.loc[local_pairs.trading_day.eq(d)].reset_index(drop=True)),
            "positive_episodes": len(scored.loc[scored.trading_day.eq(d) & complete_mask(scored) & truth(scored[CEILING], 5), KEYS].drop_duplicates())} for d in EVAL_DAYS},
        "paired_intervals": intervals(scored, local_pairs)}
    chrono_pairs, _ = local_manifest(chrono, chrono[KEYS].drop_duplicates())
    gap = (scored.decision_t-scored.state_previous_t)/1000
    long_gap = {n: metrics(probability_frame(scored.loc[m]), ARMS) for n, m in (("at_most30s", gap.le(30)), ("over30s", gap.gt(30)))}
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "market_requests": 0, "June_HOLD_opened": False, "final_July_August_opened": False,
        "input_sha256": hashes, "packages": {a: list(f) for a, f in PACKAGES.items()}, "fits": fits,
        "inference": inference, "comparison": report, "gate_checks": gate(report),
        "first": metrics(probability_frame(first), ARMS), "long_gap_diagnostic": long_gap,
        "chronological_training": {"arms": report_metrics(chrono, chrono_pairs), "folds": folds},
        "external_saved_references": {"whole_raw_ranks": raw_metrics(scored, ("W", "U", "T")),
            "local_raw_ranks": LocalCache(scored, local_pairs, ("W", "U", "T")).evaluate()},
        "integrity": {"pinned_hashes_verified": True, "saved_inputs_labels_missingness_scores_preserved": True,
            "training_adjacency_before_censor_filter_verified": True, "inference_reads_no_outcomes": True,
            "chronological_preceding_fit_scope_verified": True},
        "limitations": "Known reused May development. Hindsight reachable net>=5 labels are latent research targets, not observable regimes or realized profit.286 transition episodes; conditional supervised support279/54, onset/decay38/48, not8445 independent examples. Approximate transition recursion can compound error and does not identify an optimal Markov/Bayesian model. Initial score is a frozen causal G classifier trained on all complete states, not optimized specifically for first clocks. All censored observations stay in inference; their labels never enter. Fixed-fit intervals omit initial/transition fitting uncertainty. No feature/initial-model/gap/threshold search, policy/profits, promotion or sealed-date access."}
    (output/"request324.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("previous", "pinned-inputs", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.previous, args.pinned_inputs, args.output_dir)
    print(json.dumps({"arms": result["comparison"]["arms"], "gate_checks": result["gate_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
