"""R321: fixed nearby-pair diagnostic and one elapsed-only timing control."""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .canonical_population_momentum import verify_checkpoint
from .compact_momentum_representation import Transform, fit_transform, independent_weights
from .elapsed_momentum_evidence import complete_mask
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS
from .lagged_minute_context import CROSSFIT_DAYS
from .state_evolution_momentum import HISTORY
from .within_episode_momentum import RankCache, frozen_transform, metrics, pair_manifest, verify_pairs

REQUEST_ID = 321
PROXIMITY_MS = 30000
ARMS = ("A", "U", "V", "W", "Z", "T", "W0", "A0")
CONTRASTS = (("W", "A"), ("W", "U"), ("W", "V"), ("W", "Z"),
             ("W", "T"), ("A", "A0"), ("W", "W0"), ("V", "A"))


def subset_transform(audit: dict) -> Transform:
    """Copy per-feature training parameters; never fit a new evaluation transform."""
    full = frozen_transform(audit)
    columns = [full.features.index(f) for f in HISTORY]
    return Transform(HISTORY, *(getattr(full, n)[columns].copy() for n in
        ("lower", "upper", "median", "mean", "scale")), tuple(f for f in full.all_missing if f in HISTORY))


def fit_elapsed(training: pd.DataFrame, pairs: pd.DataFrame, frozen: dict):
    if training.empty or not set(training.trading_day).issubset(CROSSFIT_DAYS):
        raise ValueError("fixed preceding training scope required")
    verify_pairs(training, pairs)
    complete = training.loc[complete_mask(training)]
    transform = subset_transform(frozen)
    replay = fit_transform(complete, HISTORY, independent_weights(complete))
    if transform.all_missing != replay.all_missing or any(not np.allclose(getattr(transform, n),
        getattr(replay, n), atol=1e-9, rtol=0) for n in ("lower", "upper", "median", "mean", "scale")):
        raise ValueError("elapsed transform does not replay training scope")
    parameters = dict(C=.1, l1_ratio=0., solver="lbfgs", fit_intercept=False,
                      max_iter=2000, tol=1e-8, random_state=20265405)
    audit = {"fit_days": sorted(training.trading_day.unique()), "preprocessing": transform.audit(),
        "parameters": parameters, "transform_episodes": len(complete[KEYS].drop_duplicates()),
        "mixed_training_episodes": len(pairs[KEYS].drop_duplicates()), "unordered_pairs": len(pairs),
        "effective_weight_sum": float(pairs.pair_weight.sum()), "no_pair_fit": pairs.empty}
    if pairs.empty:
        return None, transform, audit | {"coefficients": None}
    delta = transform.apply(training.loc[pairs.positive_index])-transform.apply(training.loc[pairs.negative_index])
    model = LogisticRegression(**parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(np.r_[delta, -delta], np.r_[np.ones(len(delta), int), np.zeros(len(delta), int)],
                  sample_weight=np.tile(pairs.pair_weight.to_numpy()/2, 2))
    return model, transform, audit | {"coefficients": model.coef_[0].tolist(),
        "iterations": int(model.n_iter_.max()), "intercept": 0.}


def score_elapsed(held: pd.DataFrame, model, transform: Transform) -> pd.DataFrame:
    result = held.drop(columns=["A0_score"], errors="ignore").copy()
    result["A_score"] = model.decision_function(transform.apply(held)) if model is not None else np.nan
    first = result.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    ref = first[[*KEYS, "A_score"]].rename(columns={"A_score": "A0_score"})
    return result.merge(ref, on=KEYS, how="left", validate="many_to_one", sort=False)


def local_manifest(states: pd.DataFrame, identities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not states.index.is_unique or states.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("unique causal states required")
    if identities.duplicated(KEYS).any():
        raise ValueError("unique ledger identities required")
    complete = states.loc[complete_mask(states)]
    rows, counts = [], []
    for key, group in complete.groupby(KEYS, sort=False):
        y = truth(group[CEILING], 5)
        p, n = group.index.to_numpy()[y], group.index.to_numpy()[~y]
        count = 0
        for positive in p:
            gaps = np.abs(states.loc[n, "decision_t"].to_numpy()-states.loc[positive, "decision_t"])
            for negative, gap in zip(n[gaps <= PROXIMITY_MS], gaps[gaps <= PROXIMITY_MS]):
                rows.append(dict(zip(KEYS, key)) | {"positive_index": int(positive),
                    "negative_index": int(negative), "absolute_gap_ms": int(gap)})
                count += 1
        counts.append(dict(zip(KEYS, key)) | {"complete_states": len(group),
            "positive_states": int(y.sum()), "negative_states": int((~y).sum()), "local_pairs": count})
    counts_frame = pd.DataFrame(counts, columns=[*KEYS, "complete_states", "positive_states", "negative_states", "local_pairs"])
    ledger = identities[KEYS].merge(counts_frame, on=KEYS, how="left", validate="one_to_one")
    for name in ("complete_states", "positive_states", "negative_states", "local_pairs"):
        ledger[name] = ledger[name].fillna(0).astype(int)
    ledger["local_evaluable"] = ledger.local_pairs.gt(0)
    pairs = pd.DataFrame(rows, columns=[*KEYS, "positive_index", "negative_index", "absolute_gap_ms"])
    eligible = ledger.loc[ledger.local_evaluable]
    if pairs.empty:
        pairs["pair_weight"] = pd.Series(dtype=float)
        return pairs, ledger
    days = eligible.trading_day.value_counts()
    masses = eligible.set_index(KEYS).local_pairs.to_dict()
    pairs["pair_weight"] = [1/(len(days)*days.loc[key[0]]*masses[key]) for key in pairs[KEYS].itertuples(index=False, name=None)]
    verify_pairs(states, pairs)
    if not np.isclose(pairs.pair_weight.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("local pair mass must sum to one")
    return pairs, ledger


class LocalCache:
    def __init__(self, frame: pd.DataFrame, pairs: pd.DataFrame, arms=ARMS):
        self.size, self.groups = len(frame), list(pairs.groupby(KEYS, sort=False).indices.values())
        self.ep_first = np.array([frame.index.get_indexer([pairs.iloc[i[0]].positive_index])[0] for i in self.groups], int)
        self.ep_weight = np.array([pairs.iloc[i].pair_weight.sum() for i in self.groups])
        self.episodes = list(frame.groupby(KEYS, sort=False).indices.values())
        self.values = {}
        for arm in arms:
            p = frame.loc[pairs.positive_index, f"{arm}_score"].to_numpy(float)
            n = frame.loc[pairs.negative_index, f"{arm}_score"].to_numpy(float)
            if not np.isfinite(p).all() or not np.isfinite(n).all():
                raise ValueError("local pair scores must be finite")
            correct = (p > n).astype(float)+.5*(p == n)
            self.values[arm] = np.array([correct[i].mean() for i in self.groups])

    def evaluate(self, multiplicity=None) -> dict:
        m = np.ones(self.size) if multiplicity is None else np.asarray(multiplicity, float)
        if len(m) != self.size or not np.isfinite(m).all() or (m < 0).any() or any(np.ptp(m[i]) != 0 for i in self.episodes):
            raise ValueError("nonnegative whole-episode multiplicities required")
        w = self.ep_weight*m[self.ep_first]
        return {a: {"local_pair_rank": float(np.average(v, weights=w)) if w.sum() > 0 else None,
                    "local_evaluable_episodes": int((w > 0).sum())} for a, v in self.values.items()}


def local_metrics(frame: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    return LocalCache(frame, pairs).evaluate()


def intervals(frame: pd.DataFrame, pairs: pd.DataFrame, draws=1000) -> dict:
    complete = frame.loc[complete_mask(frame)]
    whole, local = RankCache(complete, ARMS), LocalCache(complete, pairs)
    output = {}
    for scheme, columns, seed in (("day", ["trading_day"], 20265550), ("ticker_day", ["trading_day", "ticker"], 20265551)):
        groups = list(complete.groupby(columns, sort=False).indices.values())
        ids = np.zeros(len(complete), int)
        for j, index in enumerate(groups):
            ids[index] = j
        rng = np.random.default_rng(seed)
        values = {f"{a}_minus_{b}": {m: [] for m in ("within_episode_auc", "local_pair_rank")} for a, b in CONTRASTS}
        for _ in range(draws):
            multiplicity = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))[ids]
            ranked, nearby = whole.evaluate(multiplicity), local.evaluate(multiplicity)
            for a, b in CONTRASTS:
                for metric, scored in (("within_episode_auc", ranked), ("local_pair_rank", nearby)):
                    x, y = scored[a][metric], scored[b][metric]
                    if x is not None and y is not None:
                        values[f"{a}_minus_{b}"][metric].append(x-y)
        output[scheme] = {"clusters": len(groups), "draws": draws, "comparisons": {
            c: {f"{m}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None for m, v in item.items()} |
               {f"{m}_valid_draws": len(v) for m, v in item.items()} for c, item in values.items()}}
        print(f"R321 {scheme} bootstrap complete", flush=True)
    return output


def gate(report: dict) -> dict:
    local = report["local"]["arms"]
    def greater(a, b):
        return a is not None and b is not None and a > b
    checks = {"at_least30_mixed_training_episodes": report["mixed_training_episodes"] >= 30,
        "at_least30_local_evaluation_episodes": local["W"]["local_evaluable_episodes"] >= 30,
        "at_least4_local_evaluable_dates": sum(v["W"]["local_pair_rank"] is not None for v in report["local"]["per_day"].values()) >= 4,
        "W_local_beats_A_U_and_half": all(greater(local["W"]["local_pair_rank"], v) for v in
            [local[a]["local_pair_rank"] for a in ("A", "U")]+[.5])}
    days = [v for v in report["local"]["per_day"].values() if v["A"]["local_pair_rank"] is not None]
    checks["W_A_local_improves_majority_dates"] = bool(days) and sum(greater(v["W"]["local_pair_rank"], v["A"]["local_pair_rank"]) for v in days) > len(days)/2
    for contrast in ("W_minus_A", "W_minus_U"):
        for scheme in ("day", "ticker_day"):
            ci = report["paired_intervals"][scheme]["comparisons"][contrast]["local_pair_rank_ci95"]
            checks[f"{scheme}_{contrast}_local_ci_positive"] = ci is not None and ci[0] > 0
    return checks


def run(previous: Path, pinned: Path, output: Path) -> dict:
    hashes = json.loads(pinned.read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((previous/name).read_bytes()).hexdigest() != digest:
            raise ValueError("preregistered input hash changed: "+name)
    prior = json.loads((previous/"official320/request320.json").read_text())
    if prior["request_id"] != 320 or prior["market_requests"] != 0:
        raise ValueError("official cached R320 required")
    training = pd.read_parquet(previous/"official319/request319-training-states.parquet").reset_index(drop=True)
    evaluation = pd.read_parquet(previous/"official320/request320-states.parquet")
    identities = pd.read_parquet(previous/"official319/request319-episode-ledger.parquet")
    manifest = pd.read_parquet(previous/"official319/request319-clock-manifest.parquet")
    verify_checkpoint(manifest, evaluation[list(manifest)])
    pairs = pd.read_parquet(previous/"official320/request320-pair-manifest.parquet")
    pair_ledger = pd.read_parquet(previous/"official320/request320-pair-ledger.parquet")
    reproduced, reproduced_ledger = pair_manifest(training)
    verify_checkpoint(pairs, reproduced)
    verify_checkpoint(pair_ledger, reproduced_ledger)
    if len(training) != 9325 or len(evaluation) != 19530 or len(identities) != 828 or len(pairs) != 47825 or set(evaluation.trading_day) != set(EVAL_DAYS):
        raise ValueError("frozen training/evaluation support changed")
    model, transform, fit = fit_elapsed(training, pairs, prior["fits"]["W"]["preprocessing"])
    scored = score_elapsed(evaluation, model, transform)
    verify_checkpoint(evaluation, scored[list(evaluation)])
    local_pairs, local_ledger = local_manifest(scored, identities)
    output.mkdir(parents=True, exist_ok=True)
    for suffix, frame in (("states", scored), ("local-pair-manifest", local_pairs), ("local-pair-ledger", local_ledger)):
        frame.to_parquet(output/f"request321-{suffix}.parquet", index=False, compression="zstd")
    whole = metrics(scored, ARMS)
    for arm in ("U", "V", "W", "Z", "T", "W0"):
        for metric in ("within_episode_auc", "between_episode_same_day_auc", "auc", "ap"):
            if abs(whole[arm][metric]-prior["comparison"]["arms"][arm][metric]) > 1e-9:
                raise ValueError("frozen R320 rank changed")
    saved = pd.read_parquet(previous/"official320/request320-chronological-states.parquet")
    pieces, folds = [], {}
    for day in CROSSFIT_DAYS[1:]:
        preceding = training.loc[training.trading_day.lt(day)].reset_index(drop=True)
        fold_pairs, _ = pair_manifest(preceding)
        held = saved.loc[saved.trading_day.eq(day)]
        model, transform, audit = fit_elapsed(preceding, fold_pairs, prior["chronological_training"]["folds"][day]["fits"]["W"]["preprocessing"])
        pieces.append(score_elapsed(held, model, transform))
        folds[day] = audit
    chrono = pd.concat(pieces, ignore_index=True)
    verify_checkpoint(saved, chrono[list(saved)])
    chrono.to_parquet(output/"request321-chronological-states.parquet", index=False, compression="zstd")
    print(f"R321 one elapsed head fitted; {len(local_pairs)} local evaluation pairs persisted", flush=True)
    report = {"mixed_training_episodes": fit["mixed_training_episodes"], "whole": {"arms": whole},
        "local": {"pairs": len(local_pairs), "ledger_episodes": len(local_ledger), "arms": local_metrics(scored, local_pairs),
            "per_day": {day: local_metrics(scored.loc[scored.trading_day.eq(day)], local_pairs.loc[local_pairs.trading_day.eq(day)].reset_index(drop=True)) for day in EVAL_DAYS}},
        "paired_intervals": intervals(scored, local_pairs)}
    positive_t = scored.loc[local_pairs.positive_index, "decision_t"].to_numpy()
    negative_t = scored.loc[local_pairs.negative_index, "decision_t"].to_numpy()
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "market_requests": 0, "June_HOLD_opened": False, "final_July_August_opened": False,
        "input_sha256": hashes, "proximity_ms": PROXIMITY_MS, "fit": fit, "comparison": report,
        "gate_checks": gate(report), "chronological_training": {"arms": metrics(chrono, ARMS), "folds": folds},
        "descriptive_pair_order": {"positive_earlier_pairs": int((positive_t < negative_t).sum()),
            "positive_later_pairs": int((positive_t > negative_t).sum()), "same_clock_pairs": int((positive_t == negative_t).sum())},
        "integrity": {"pinned_hashes_verified": True, "saved_inputs_labels_missingness_scores_preserved": True,
            "frozen_R320_ranks_replayed": True, "frozen_transform_subset_replayed": True,
            "local_pair_manifest_sha256": hashlib.sha256((output/"request321-local-pair-manifest.parquet").read_bytes()).hexdigest()},
        "limitations": "Reused May development.50 mixed training episodes; local evaluation selects label-opposed nearby pairs only for diagnostics, never online admission. Elapsed-only control and30s proximity do not eliminate all temporal/quality confounding. Equal eligible day/episode/pair mass differs from whole-episode anchor weights. Raw scores are relative, not calibrated probabilities or realized returns. Fixed-fit cluster intervals omit fitting uncertainty. No window/parameter/arm search, promotion or sealed-date access."}
    (output/"request321.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("previous", "pinned-inputs", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.previous, args.pinned_inputs, args.output_dir)
    print(json.dumps({"whole": result["comparison"]["whole"]["arms"], "local": result["comparison"]["local"], "gate_checks": result["gate_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
