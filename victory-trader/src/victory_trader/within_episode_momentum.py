"""R320: fixed within-episode objective, scoring each causal state separately."""
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
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .state_evolution_momentum import PACKAGES as PREVIOUS_PACKAGES, binned_mass, metrics as previous_metrics, pair_mass

REQUEST_ID = 320
SEED = 20265405
PACKAGES = {"U": PREVIOUS_PACKAGES["G"], "V": PREVIOUS_PACKAGES["S"], "W": PREVIOUS_PACKAGES["T"]}
ARMS = ("U", "V", "W", "Z", "S", "T", "U0", "V0", "W0")
METRICS = ("within_episode_auc", "between_episode_same_day_auc", "auc", "ap")
CONTRASTS = (("W", "V"), ("W", "U"), ("W", "T"), ("W", "Z"), ("V", "S"),
             ("W", "W0"), ("V", "V0"), ("U", "U0"))


def pair_manifest(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use all complete training endpoints; never choose strongest/future-best pairs."""
    if states.empty or not set(states.trading_day).issubset(CROSSFIT_DAYS):
        raise ValueError("nonempty fixed training dates required")
    if not states.index.is_unique or states.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("duplicate causal state")
    complete = states.loc[complete_mask(states)]
    ledger_rows = []
    for key, group in states.groupby(KEYS, sort=False):
        fitting = group.loc[complete_mask(group)]
        positive = int(truth(fitting[CEILING], 5).sum())
        negative = len(fitting)-positive
        ledger_rows.append(dict(zip(KEYS, key)) | {"observed_states": len(group), "complete_states": len(fitting),
            "positive_states": positive, "negative_states": negative, "pairs": positive*negative,
            "mixed_label": positive > 0 and negative > 0})
    ledger = pd.DataFrame(ledger_rows)
    mixed = ledger.loc[ledger.mixed_label]
    if mixed.empty:
        return pd.DataFrame(columns=[*KEYS, "positive_index", "negative_index", "pair_weight"]), ledger
    counts = mixed.trading_day.value_counts()
    rows = []
    for key, group in complete.groupby(KEYS, sort=False):
        y = truth(group[CEILING], 5)
        positive, negative = group.index.to_numpy()[y], group.index.to_numpy()[~y]
        if not len(positive) or not len(negative):
            continue
        mass = len(mixed)/(len(counts)*counts.loc[key[0]])
        # One positive-first pair; fitting later adds the opposite orientation
        # with half this weight, preserving the episode's total fitting mass.
        weight = float(mass/(len(positive)*len(negative)))
        rows.extend(dict(zip(KEYS, key)) | {"positive_index": int(p), "negative_index": int(n), "pair_weight": weight}
                    for p in positive for n in negative)
    return pd.DataFrame(rows), ledger


def verify_pairs(states: pd.DataFrame, pairs: pd.DataFrame) -> None:
    if pairs.empty:
        return
    if pairs.duplicated(["positive_index", "negative_index"]).any() or not np.isfinite(pairs.pair_weight).all() or pairs.pair_weight.le(0).any():
        raise ValueError("unique pairs and positive finite weights required")
    positive = states.loc[pairs.positive_index].reset_index(drop=True)
    negative = states.loc[pairs.negative_index].reset_index(drop=True)
    if not complete_mask(positive).all() or not complete_mask(negative).all():
        raise ValueError("censored pair endpoint")
    if not truth(positive[CEILING], 5).all() or truth(negative[CEILING], 5).any():
        raise ValueError("incorrect pair orientation")
    for col in KEYS:
        if not positive[col].equals(negative[col]) or not positive[col].equals(pairs[col].reset_index(drop=True)):
            raise ValueError("pair crossed episode boundary")


def frozen_transform(audit: dict) -> Transform:
    return Transform(tuple(audit["raw_features"]),
        *(np.asarray(audit[n], dtype=float) for n in ("lower", "upper", "median", "mean", "scale")),
        tuple(audit["all_training_missing"]))


def fit_pairwise(states: pd.DataFrame, pairs: pd.DataFrame, features: tuple[str, ...], frozen: dict | None = None):
    if not set(states.trading_day).issubset(CROSSFIT_DAYS) or features not in tuple(PACKAGES.values()):
        raise ValueError("declared training scope and feature package required")
    train = states.loc[complete_mask(states)]
    if train.empty:
        raise ValueError("complete preceding training support required")
    verify_pairs(states, pairs)
    replay = fit_transform(train, features, independent_weights(train))
    transform = frozen_transform(frozen) if frozen is not None else replay
    if transform.features != features or transform.all_missing != replay.all_missing:
        raise ValueError("frozen training transform package changed")
    for name in ("lower", "upper", "median", "mean", "scale"):
        if not np.allclose(getattr(transform, name), getattr(replay, name), atol=1e-9, rtol=0):
            raise ValueError("frozen transform does not replay its training scope")
    audit = {"fit_days": sorted(train.trading_day.unique()), "preprocessing": transform.audit(),
        "transform_episodes": len(train[KEYS].drop_duplicates()), "complete_training_states": len(train),
        "mixed_training_episodes": len(pairs[KEYS].drop_duplicates()), "unordered_pairs": len(pairs),
        "effective_weight_sum": float(pairs.pair_weight.sum()), "no_pair_fit": pairs.empty,
        "parameters": {"C": .1, "l1_ratio": 0., "solver": "lbfgs", "fit_intercept": False,
                       "max_iter": 2000, "tol": 1e-8, "random_state": SEED}}
    if pairs.empty:
        audit["coefficients"] = None
        return None, transform, audit
    positive, negative = states.loc[pairs.positive_index], states.loc[pairs.negative_index]
    delta = transform.apply(positive)-transform.apply(negative)
    x = np.r_[delta, -delta]
    y = np.r_[np.ones(len(delta), dtype=int), np.zeros(len(delta), dtype=int)]
    weights = np.r_[pairs.pair_weight.to_numpy()/2, pairs.pair_weight.to_numpy()/2]
    if not np.isclose(weights.sum(), len(pairs[KEYS].drop_duplicates()), rtol=0, atol=1e-9):
        raise ValueError("pair fitting mass must equal mixed episode count")
    model = LogisticRegression(**audit["parameters"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, sample_weight=weights)
    audit.update(coefficients=model.coef_[0].tolist(), iterations=int(model.n_iter_.max()),
                 intercept=float(model.intercept_[0]) if len(model.intercept_) else 0.)
    return model, transform, audit


def fit_matched_classifier(states: pd.DataFrame, pairs: pd.DataFrame, transform: Transform):
    """Same mixed episodes, preprocessing, parameter/weight mass and class balance."""
    verify_pairs(states, pairs)
    if pairs.empty:
        return None, {"no_pair_fit": True, "effective_weight_sum": 0., "train_rows": 0, "coefficients": None}
    # Pair endpoint marginals give half each episode's mass to positives and
    # half to negatives, exactly matching the two-class pair loss balance.
    weights = pd.concat([pairs.groupby("positive_index").pair_weight.sum()/2,
                         pairs.groupby("negative_index").pair_weight.sum()/2]).sort_index()
    training = states.loc[weights.index]
    parameters = dict(C=.1, l1_ratio=0., solver="lbfgs", fit_intercept=False,
                      max_iter=2000, tol=1e-8, random_state=SEED)
    model = LogisticRegression(**parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(transform.apply(training), truth(training[CEILING], 5), sample_weight=weights.to_numpy())
    return model, {"parameters": parameters, "no_pair_fit": False, "train_rows": len(training),
        "mixed_training_episodes": len(training[KEYS].drop_duplicates()), "effective_weight_sum": float(weights.sum()),
        "positive_weight": float(weights.to_numpy()[truth(training[CEILING], 5)].sum()),
        "negative_weight": float(weights.to_numpy()[~truth(training[CEILING], 5)].sum()),
        "preprocessing": transform.audit(), "fit_days": sorted(training.trading_day.unique()),
        "coefficients": model.coef_[0].tolist(), "iterations": int(model.n_iter_.max())}


def score(states: pd.DataFrame, pairs: pd.DataFrame, held: pd.DataFrame, frozen: dict | None = None) -> tuple[pd.DataFrame, dict]:
    result, audits = held.copy(), {}
    for arm, features in PACKAGES.items():
        model, transform, audit = fit_pairwise(states, pairs, features, frozen[arm] if frozen is not None else None)
        result[f"{arm}_score"] = model.decision_function(transform.apply(held)) if model is not None else np.nan
        audits[arm] = audit
        if arm == "W":
            classifier, control_audit = fit_matched_classifier(states, pairs, transform)
            result["Z_score"] = classifier.decision_function(transform.apply(held)) if classifier is not None else np.nan
            audits["Z"] = control_audit
    # These are saved classification scores used only for ranks, not refitted.
    for arm in ("S", "T"):
        result[f"{arm}_score"] = result[f"{arm}_p_net_5"]
    return result, audits


def broadcast_first(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    first = result.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    for arm in PACKAGES:
        ref = first[[*KEYS, f"{arm}_score"]].rename(columns={f"{arm}_score": f"{arm}0_score"})
        result = result.merge(ref, on=KEYS, how="left", validate="many_to_one", sort=False)
    return result


class RankCache:
    """Unbounded relative scores; deliberately no probability Brier metric."""
    def __init__(self, frame: pd.DataFrame, arms=ARMS):
        self.y, self.w = truth(frame[CEILING], 5), _episode_day_weights(frame)
        self.days = list(frame.groupby("trading_day", sort=False).indices.values())
        self.episodes = list(frame.groupby(KEYS, sort=False).indices.values())
        self.ep_weights = np.array([self.w[i].sum() for i in self.episodes])
        self.ep_first = np.array([i[0] for i in self.episodes])
        self.bins, self.within = {}, {}
        for arm in arms:
            p = frame[f"{arm}_score"].to_numpy(float)
            if not np.isfinite(p).all():
                raise ValueError("all ranked relative scores must be finite")
            self.bins[arm] = np.unique(p, return_inverse=True)[1]
            self.within[arm] = np.array([pair_mass(self.y[i], p[i], self.w[i]) for i in self.episodes])

    def evaluate(self, multiplicity=None) -> dict:
        m = np.ones(len(self.y)) if multiplicity is None else np.asarray(multiplicity, float)
        if len(m) != len(self.y) or (m < 0).any() or not np.isfinite(m).all() or not (m > 0).any():
            raise ValueError("finite nonnegative whole-cluster multiplicities required")
        if any(np.ptp(m[i]) != 0 for i in self.episodes):
            raise ValueError("whole episodes must remain together")
        weights, ep_m = self.w*m, m[self.ep_first]
        output = {}
        for arm, bins in self.bins.items():
            numerator, denominator = binned_mass(self.y, bins, weights)
            pos = np.bincount(bins, weights=weights*self.y)
            neg = np.bincount(bins, weights=weights*(~self.y), minlength=len(pos))
            tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
            precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=tp+fp > 0)
            between_n = between_d = 0.
            for i in self.days:
                n, d = binned_mass(self.y[i], bins[i], weights[i])
                between_n += n
                between_d += d
            within = self.within[arm]
            between_n -= float(np.sum(within[:, 0]*ep_m**2))
            between_d -= float(np.sum(within[:, 1]*ep_m**2))
            valid = (within[:, 1] > 0) & (ep_m > 0)
            output[arm] = {"auc": numerator/denominator if denominator > 0 else None,
                "ap": float(np.sum(pos[::-1]*precision)/pos.sum()) if denominator > 0 else None,
                "between_episode_same_day_auc": between_n/between_d if between_d > 1e-14 else None,
                "within_episode_auc": float(np.average(within[valid, 0]/within[valid, 1], weights=self.ep_weights[valid]*ep_m[valid])) if valid.any() else None,
                "within_episode_evaluable_episodes": int(valid.sum())}
        return output


def metrics(frame: pd.DataFrame, arms=ARMS) -> dict:
    complete = frame.loc[complete_mask(frame)].copy()
    finite = np.isfinite(complete[[f"{a}_score" for a in arms]].to_numpy(float)).all(axis=1)
    ranked = complete.loc[finite].reset_index(drop=True)
    if ranked.empty:
        return {a: {**dict.fromkeys(METRICS, None), "within_episode_evaluable_episodes": 0} for a in arms}
    return RankCache(ranked, arms).evaluate()


def intervals(frame: pd.DataFrame, draws=1000) -> dict:
    complete = frame.loc[complete_mask(frame)].reset_index(drop=True)
    cache = RankCache(complete)
    output = {}
    for scheme, columns, seed in (("day", ["trading_day"], 20265450), ("ticker_day", ["trading_day", "ticker"], 20265451)):
        groups = list(complete.groupby(columns, sort=False).indices.values())
        ids = np.zeros(len(complete), dtype=int)
        for j, index in enumerate(groups):
            ids[index] = j
        rng = np.random.default_rng(seed)
        values = {f"{a}_minus_{b}": {m: [] for m in METRICS} for a, b in CONTRASTS}
        for _ in range(draws):
            counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            scored = cache.evaluate(counts[ids])
            for a, b in CONTRASTS:
                for metric in METRICS:
                    x, y = scored[a][metric], scored[b][metric]
                    if x is not None and y is not None:
                        values[f"{a}_minus_{b}"][metric].append(x-y)
        output[scheme] = {"clusters": len(groups), "draws": draws, "comparisons": {
            n: {f"{m}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None for m, v in item.items()} |
               {f"{m}_valid_draws": len(v) for m, v in item.items()} for n, item in values.items()}}
        print(f"R320 {scheme} bootstrap complete", flush=True)
    return output


def chronological(training: pd.DataFrame, saved: pd.DataFrame, frozen: dict | None = None) -> tuple[pd.DataFrame, dict]:
    pieces, folds = [], {}
    for day in CROSSFIT_DAYS[1:]:
        preceding = training.loc[training.trading_day.lt(day)].reset_index(drop=True)
        pairs, ledger = pair_manifest(preceding)
        held = saved.loc[saved.trading_day.eq(day)]
        if held.empty or preceding.empty or preceding.trading_day.ge(day).any():
            raise ValueError("nonempty preceding chronological scope required")
        if frozen is None:
            scored, audit = score(preceding, pairs, held)
        else:
            scored, audit = score(preceding, pairs, held, frozen[day])
        pieces.append(broadcast_first(scored))
        folds[day] = {"fits": audit, "pair_ledger_mixed_episodes": int(ledger.mixed_label.sum()), "unscored_no_pair_fold": pairs.empty}
    return pd.concat(pieces, ignore_index=True), folds


def gate(report: dict) -> dict:
    arms = report["arms"]
    def greater(a, b):
        return a is not None and b is not None and a > b
    checks = {"at_least30_mixed_training_episodes": report["mixed_training_episodes"] >= 30,
        "at_least50_mixed_evaluation_episodes": arms["W"]["within_episode_evaluable_episodes"] >= 50,
        "at_least4_evaluable_dates": sum(v["W"]["within_episode_auc"] is not None for v in report["per_day"].values()) >= 4,
        "W_timing_beats_V_U_T_Z_and_half": all(greater(arms["W"]["within_episode_auc"], v) for v in [arms[a]["within_episode_auc"] for a in ("V", "U", "T", "Z")]+[.5])}
    days = [v for v in report["per_day"].values() if v["V"]["within_episode_auc"] is not None]
    checks["W_V_timing_improves_majority_dates"] = bool(days) and sum(greater(v["W"]["within_episode_auc"], v["V"]["within_episode_auc"]) for v in days) > len(days)/2
    for contrast in ("W_minus_V", "W_minus_T", "W_minus_Z", "W_minus_W0"):
        for scheme in ("day", "ticker_day"):
            ci = report["paired_intervals"][scheme]["comparisons"][contrast]["within_episode_auc_ci95"]
            checks[f"{scheme}_{contrast}_timing_ci_positive"] = ci is not None and ci[0] > 0
    return checks


def run(previous: Path, pinned: Path, output: Path) -> dict:
    hashes = json.loads(pinned.read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((previous/name).read_bytes()).hexdigest() != digest:
            raise ValueError("preregistered R319 input hash changed: "+name)
    prior = json.loads((previous/"request319.json").read_text())
    if prior.get("request_id") != 319 or prior.get("market_requests") != 0:
        raise ValueError("official cached R319 provenance required")
    training = pd.read_parquet(previous/"request319-training-states.parquet").reset_index(drop=True)
    evaluation = pd.read_parquet(previous/"request319-states.parquet")
    ledger = pd.read_parquet(previous/"request319-episode-ledger.parquet")
    manifest = pd.read_parquet(previous/"request319-clock-manifest.parquet")
    verify_checkpoint(manifest, evaluation[list(manifest)])
    if len(training) != 9325 or len(evaluation) != 19530 or len(ledger) != 828 or set(evaluation.trading_day) != set(EVAL_DAYS):
        raise ValueError("fixed R319 support changed")
    pairs, pair_ledger = pair_manifest(training)
    if len(pairs) != 47825 or int(pair_ledger.mixed_label.sum()) != 50:
        raise ValueError("declared mixed-label training support changed")
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output/"request320-pair-manifest.parquet", index=False, compression="zstd")
    pair_ledger.to_parquet(output/"request320-pair-ledger.parquet", index=False, compression="zstd")
    print(f"R320 froze {len(pairs)} within-episode pairs from50 episodes before fits", flush=True)
    transforms = {a: prior["fits"][r]["preprocessing"] for a, r in (("U", "G"), ("V", "S"), ("W", "T"))}
    scored, fits = score(training, pairs, evaluation, transforms)
    scored = broadcast_first(scored)
    verify_checkpoint(evaluation, scored[list(evaluation)])
    # The state transforms are EXACT R319 transforms, fitted on all336 complete
    # episodes; supervised pair fitting alone has50 independent mixed episodes.
    for arm, ref in (("U", "G"), ("V", "S"), ("W", "T")):
        if fits[arm]["preprocessing"] != prior["fits"][ref]["preprocessing"]:
            raise ValueError("frozen R319 preprocessing changed")
    baseline = previous_metrics(evaluation, ("S", "T"))
    current = metrics(scored)
    baseline_errors = {}
    for arm in ("S", "T"):
        errors = [abs(current[arm][m]-baseline[arm][m]) for m in METRICS if baseline[arm][m] is not None]
        baseline_errors[arm] = max(errors, default=0.)
        if baseline_errors[arm] > 1e-9:
            raise ValueError("R319 classification rank replay changed")
    scored.to_parquet(output/"request320-states.parquet", index=False, compression="zstd")
    saved_chrono = pd.read_parquet(previous/"request319-chronological-states.parquet")
    fold_transforms = {day: {a: prior["chronological_training"]["folds"][day][r]["preprocessing"]
        for a, r in (("U", "G"), ("V", "S"), ("W", "T"))} for day in CROSSFIT_DAYS[1:]}
    chrono, folds = chronological(training, saved_chrono, fold_transforms)
    verify_checkpoint(saved_chrono, chrono[list(saved_chrono)])
    chrono.to_parquet(output/"request320-chronological-states.parquet", index=False, compression="zstd")
    print("R320 all fixed/chronological fits complete", flush=True)
    report = {"mixed_training_episodes": 50, "observed_states": len(scored), "complete_states": int(complete_mask(scored).sum()),
        "complete_evaluation_episodes": len(scored.loc[complete_mask(scored), KEYS].drop_duplicates()),
        "ledger_episodes": len(ledger), "arms": current,
        "per_day": {d: metrics(scored.loc[scored.trading_day.eq(d)]) for d in EVAL_DAYS},
        "paired_intervals": intervals(scored)}
    elapsed = (scored.decision_t-scored.hot_t)/1000
    phases = {f"hot_elapsed_{lo}_{hi}s": metrics(scored.loc[elapsed.gt(lo) & elapsed.le(hi)]) for lo, hi in ((0, 30), (30, 120), (120, 300))}
    phases["lag_available"] = metrics(scored.loc[scored.trajectory_lag_age_s.notna()])
    phases["lag_unavailable"] = metrics(scored.loc[scored.trajectory_lag_age_s.isna()])
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "market_requests": 0, "June_HOLD_opened": False, "final_July_August_opened": False,
        "input_sha256": hashes, "packages": {a: list(f) for a, f in PACKAGES.items()},
        "fits": fits, "comparison": report, "fixed_phases": phases, "gate_checks": gate(report),
        "chronological_training": {"arms": metrics(chrono), "folds": folds},
        "integrity": {"pinned_hashes_verified": True, "saved_R319_scores_inputs_labels_missingness_preserved": True,
            "R319_rank_replay_max_errors": baseline_errors, "frozen_training_transforms_verified": True,
            "pair_manifest_sha256": hashlib.sha256((output/"request320-pair-manifest.parquet").read_bytes()).hexdigest()},
        "limitations": "Known reused May development. Only50 mixed training episodes, not47825 independent pairs. Training pairs condition on complete positive/negative labels, while all evaluation observations remain scored. Raw scores are relative ordering, not absolute probabilities; no Brier, probability calibration or realized returns. W versus saved R319 T also changes supervised support, mass and intercept; matched Z shares these contracts and endpoint class balance but compares state-classification loss with pairwise relative loss. Rank intervals condition on fitted models. No feature/parameter/window search, policy, promotion or sealed-date access."}
    (output/"request320.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("previous", "pinned-inputs", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.previous, args.pinned_inputs, args.output_dir)
    print(json.dumps({"arms": result["comparison"]["arms"], "gate_checks": result["gate_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
