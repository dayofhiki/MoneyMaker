"""R318: fixed nonlinear/interaction-capable fits on official R317 states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .canonical_population_momentum import verify_checkpoint
from .clock_momentum_increment import CONTROL, MOMENTUM, within_day_auc
from .compact_momentum_representation import fit_linear, fit_transform, independent_weights, predict_linear
from .continuous_population_momentum import fitting_rows, state_support
from .elapsed_momentum_evidence import complete_mask
from .feasible_upside_observability import CEILING, classification_metrics, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS

REQUEST_ID = 318
SEED = 20265205
ARMS = ("R", "C", "Q", "X", "K", "A", "D")
NEW_HEADS = {"X": (MOMENTUM, 2), "K": (CONTROL, 2), "A": (MOMENTUM, 1), "D": (CONTROL, 1)}
CONTRASTS = (("X", "C"), ("X", "K"), ("X", "R"), ("X", "A"), ("A", "D"), ("K", "Q"))
DIFFERENCES = {"joint_minus_linear_increment": (("X", "K"), ("C", "Q")),
               "joint_minus_additive_increment": (("X", "K"), ("A", "D"))}


def ranks(frame: pd.DataFrame, arms: tuple[str, ...]) -> dict:
    # Legacy preentry.metrics reserves A for an unrelated nonprobability model.
    # R318 A is the preregistered additive probability head; all arms use the
    # same generic classification evaluator here.
    labeled = frame.loc[complete_mask(frame)]
    return {a: classification_metrics(labeled.rename(columns={f"{a}_p_net_5": "p_net_5",
                         f"{a}_prior_net_5": "prior_net_5"}), 5) |
                    {"within_day_auc": within_day_auc(labeled, a)} for a in arms}


def tree_audit(model, transform, training: pd.DataFrame, weights: np.ndarray) -> dict:
    x = transform.apply(training)
    episode_ids = pd.factorize(pd.MultiIndex.from_frame(training[KEYS]), sort=False)[0]
    types = ["control" if f in CONTROL else "signal" for f in transform.features]*2
    leaf_weights, leaf_episodes, depths = [], [], []
    paths = {"control_only": 0, "signal_only": 0, "mixed_control_signal": 0, "no_split": 0}
    for estimator in model.estimators_[:, 0]:
        tree = estimator.tree_
        depths.append(int(tree.max_depth))
        assignments = estimator.apply(x)
        for leaf in np.unique(assignments):
            mask = assignments == leaf
            weight = float(weights[mask].sum())
            if not np.isclose(weight, tree.weighted_n_node_samples[leaf], atol=1e-9, rtol=0):
                raise ValueError("weighted leaf support mismatch")
            if weight+1e-9 < .025*weights.sum():
                raise ValueError("minimum fitting-weight leaf support violated")
            leaf_weights.append(weight)
            leaf_episodes.append(int(len(np.unique(episode_ids[mask]))))
        stack = [(0, frozenset())]
        while stack:
            node, used = stack.pop()
            if tree.children_left[node] == tree.children_right[node]:
                name = "mixed_control_signal" if len(used) == 2 else next(iter(used))+"_only" if used else "no_split"
                paths[name] += 1
            else:
                added = used | {types[int(tree.feature[node])]}
                stack.extend((int(child), added) for child in (tree.children_left[node], tree.children_right[node]))
    return {"trees": len(depths), "actual_max_depth": max(depths), "leaf_paths": paths,
            "minimum_leaf_weight": min(leaf_weights), "minimum_leaf_weight_fraction": min(leaf_weights)/weights.sum(),
            "minimum_leaf_distinct_episodes": min(leaf_episodes),
            "median_leaf_distinct_episodes": float(np.median(leaf_episodes)),
            "maximum_leaf_distinct_episodes": max(leaf_episodes)}


def fit_shallow(train: pd.DataFrame, features: tuple[str, ...], depth: int):
    if depth not in (1, 2) or train.empty or not set(train.trading_day).issubset(CROSSFIT_DAYS):
        raise ValueError("nonempty fixed-date support and declared depth required")
    weights, y = independent_weights(train), truth(train[CEILING], 5)
    transform = fit_transform(train, features, weights)
    prior = float(np.average(y, weights=weights))
    parameters = dict(loss="log_loss", learning_rate=.04, n_estimators=100, subsample=1., criterion="friedman_mse",
                      min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=.025, max_depth=depth,
                      max_leaf_nodes=2**depth, max_features=None, random_state=SEED, n_iter_no_change=None)
    audit = state_support(train) | {"fit_days": sorted(train.trading_day.unique()), "preprocessing": transform.audit(),
        "parameters": parameters, "single_class_fallback": len(np.unique(y)) != 2}
    if len(np.unique(y)) != 2:
        audit["trees"] = None
        return None, transform, prior, audit
    model = GradientBoostingClassifier(**parameters)
    model.fit(transform.apply(train), y, sample_weight=weights)
    audit["trees"] = tree_audit(model, transform, train, weights)
    return model, transform, prior, audit


def score(train: pd.DataFrame, held: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if train.empty or not complete_mask(train).all() or not set(train.trading_day).issubset(CROSSFIT_DAYS):
        raise ValueError("complete fixed-date training support required")
    if train.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("duplicate continuous training observation")
    result, audits = held.copy(), {}
    for arm in ("C", "Q", *NEW_HEADS):
        features = CONTROL if arm == "Q" else MOMENTUM
        if arm in NEW_HEADS:
            features, depth = NEW_HEADS[arm]
            model, transform, prior, audit = fit_shallow(train, features, depth)
        else:
            model, transform, prior, audit = fit_linear(train, features, 20265105)
            audit["fit_days"] = sorted(train.trading_day.unique())
        available = result.observation_available.astype(bool)
        result[f"{arm}_p_net_5"] = np.nan
        if available.any():
            result.loc[available, f"{arm}_p_net_5"] = predict_linear(result.loc[available], model, transform, prior)
        result[f"{arm}_prior_net_5"] = prior
        audits[arm] = audit
    return result, audits


def chronological(first: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if not set(first.trading_day).issubset(CROSSFIT_DAYS) or not set(train.trading_day).issubset(CROSSFIT_DAYS):
        raise ValueError("chronological inputs crossed fixed dates")
    pieces, folds = [], {}
    for day in CROSSFIT_DAYS[1:]:
        fitting = train.loc[train.trading_day.lt(day)]
        held = first.loc[first.in_broad & first.trading_day.eq(day)]
        if fitting.empty or held.empty or fitting.trading_day.ge(day).any():
            raise ValueError("nonempty preceding chronological support required")
        scored, audit = score(fitting, held)
        pieces.append(scored)
        folds[day] = audit
    return pd.concat(pieces, ignore_index=True), folds


def replay_error(current: pd.DataFrame, saved: pd.DataFrame, arms=("C", "Q")) -> dict:
    protected = [c for c in saved if c not in [f"{a}_{suffix}" for a in NEW_HEADS for suffix in ("p_net_5", "prior_net_5")]]
    # Preserve prior scores within numerical replay tolerance, all inputs/labels too.
    verify_checkpoint(saved[protected].reset_index(drop=True), current[protected].reset_index(drop=True))
    errors = {}
    for arm in arms:
        available = current.observation_available.astype(bool)
        a, b = (f.loc[available, f"{arm}_p_net_5"].to_numpy(float) for f in (current, saved))
        errors[arm] = float(np.max(np.abs(a-b))) if len(a) else 0.
        if errors[arm] > 1e-9:
            raise ValueError("original R317 probability replay changed")
    return errors


def contrasts(values: dict, metric: str) -> dict:
    output = {f"{a}_minus_{b}": values[a][metric]-values[b][metric] for a, b in CONTRASTS}
    for name, ((a, b), (c, d)) in DIFFERENCES.items():
        output[name] = (values[a][metric]-values[b][metric])-(values[c][metric]-values[d][metric])
    return output


def intervals(frame: pd.DataFrame, draws: int = 1000) -> dict:
    y, weights = truth(frame[CEILING], 5), _episode_day_weights(frame)
    p = {a: frame[f"{a}_p_net_5"].to_numpy(float) for a in ARMS}
    names = [*(f"{a}_minus_{b}" for a, b in CONTRASTS), *DIFFERENCES]
    output = {}
    for i, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        groups = list(frame.groupby(columns, sort=False).indices.values())
        values = {name: {m: [] for m in ("auc", "within_day_auc", "ap", "brier")} for name in names}
        rng = np.random.default_rng(20265250+i)
        for _ in range(draws):
            index = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            sample, w = frame.iloc[index], weights[index]
            both = len(np.unique(y[index])) == 2
            scores = {a: {"brier": float(np.average((y[index]-p[a][index])**2, weights=w)),
                "within_day_auc": within_day_auc(sample, a, w)} for a in ARMS}
            if both:
                for a in ARMS:
                    scores[a].update(auc=float(roc_auc_score(y[index], p[a][index], sample_weight=w)),
                                     ap=float(average_precision_score(y[index], p[a][index], sample_weight=w)))
            for metric in ("auc", "within_day_auc", "ap", "brier"):
                if any(v.get(metric) is None for v in scores.values()):
                    continue
                for name, difference in contrasts(scores, metric).items():
                    values[name][metric].append(difference)
        output["day" if i == 0 else "ticker_day"] = {"clusters": len(groups), "draws": draws,
            "comparisons": {name: {f"{m}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None for m, v in report.items()} |
                                   {f"{m}_valid_draws": len(v) for m, v in report.items()} for name, report in values.items()}}
    return output


def gate(report: dict) -> dict:
    x, others = report["arms"]["X"], [report["arms"][a] for a in ("C", "K", "R")]
    def greater(a, b):
        return a is not None and b is not None and a > b
    days = [v for v in report["per_day"].values() if v["C"]["auc"] is not None]
    checks = {"evaluation_coverage_at_least90pct": report["complete_coverage"] >= .9,
              "at_least20_positive_episodes": x["positive_episodes"] >= 20,
              "at_least4_positive_days": sum(v["X"]["positive_episodes"] > 0 for v in report["per_day"].values()) >= 4,
              "X_same_day_auc_beats_C_K_R": all(greater(x["within_day_auc"], v["within_day_auc"]) for v in others),
              "X_ap_beats_C_K_R": all(greater(x["ap"], v["ap"]) for v in others),
              "X_brier_beats_C_K_R_and_prior": all(greater(v, x["brier"]) for v in [*(o["brier"] for o in others), x["prior_brier"]]),
              "X_C_auc_improves_majority_days": bool(days) and sum(greater(v["X"]["auc"], v["C"]["auc"]) for v in days) > len(days)/2}
    for name in ("X_minus_C", "X_minus_K", "joint_minus_linear_increment"):
        ci = report["paired_intervals"]["ticker_day"]["comparisons"][name]["within_day_auc_ci95"]
        checks[f"ticker_day_{name}_gain_ci_positive"] = ci is not None and ci[0] > 0
    return checks


def run(previous: Path, canonical: Path, pinned: Path, output: Path) -> dict:
    hashes = json.loads(pinned.read_text())
    for name, digest in hashes.items():
        path = canonical/name if name.startswith("request315") else previous/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("preregistered input hash changed: "+name)
    original = json.loads((previous/"request317.json").read_text())
    if original.get("request_id") != 317 or original.get("market_requests") != 0:
        raise ValueError("original R317 provenance required")
    states = pd.read_parquet(previous/"request317-training-states.parquet")
    manifest = pd.read_parquet(previous/"request317-clock-manifest.parquet")
    ledger = pd.read_parquet(previous/"request317-episode-ledger.parquet")
    verify_checkpoint(manifest, states[list(manifest)])
    first = pd.read_parquet(canonical/"request315-training-states.parquet")
    rows = fitting_rows(first, states)
    train = rows["C"]
    if len(states) != 9325 or len(ledger) != 407 or len(train) != 8792 or len(train[KEYS].drop_duplicates()) != 336:
        raise ValueError("original R317 training support changed")
    saved = pd.read_parquet(previous/"request317-states.parquet")
    if set(saved.trading_day) != set(EVAL_DAYS) or len(saved) != 828 or int(complete_mask(saved).sum()) != 650:
        raise ValueError("original evaluation support changed")
    output.mkdir(parents=True, exist_ok=True)
    scored, fits = score(train, saved)
    integrity = replay_error(scored, saved)
    scored.to_parquet(output/"request318-states.parquet", index=False, compression="zstd")
    chrono, folds = chronological(first, train)
    saved_chrono = pd.read_parquet(previous/"request317-chronological-states.parquet")
    common = [c for c in saved_chrono if c in chrono]
    chronological_errors = replay_error(chrono[common], saved_chrono[common])
    chrono_report = ranks(chrono, ("C", "Q", *NEW_HEADS))
    for arm in ("C", "Q"):
        for metric, value in original["chronological_training"]["arms"][arm].items():
            if value is None:
                if chrono_report[arm][metric] is not None:
                    raise ValueError("chronological metric availability changed")
            elif abs(chrono_report[arm][metric]-value) > 1e-9:
                raise ValueError("R317 chronological baseline metric replay changed")
    chrono.to_parquet(output/"request318-chronological-states.parquet", index=False, compression="zstd")
    print("R318 frozen/chronological fits complete; computing paired intervals", flush=True)
    arm_report = ranks(scored, ARMS)
    comparison = {"sampled_episodes": len(scored), "observed_episodes": int(scored.observation_available.sum()),
                  "labeled_episodes": int(complete_mask(scored).sum()), "complete_coverage": float(complete_mask(scored).mean()),
                  "arms": arm_report, "per_day": {d: ranks(scored.loc[scored.trading_day.eq(d)], ARMS) for d in EVAL_DAYS},
                  "point_contrasts": {m: contrasts(arm_report, m) for m in ("auc", "within_day_auc", "ap", "brier")},
                  "paired_intervals": intervals(scored.loc[complete_mask(scored)])}
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False, "market_requests": 0,
              "June_HOLD_opened": False, "final_July_August_opened": False, "labels_clocks_or_thresholds_changed": False,
              "input_sha256": hashes, "pinned_manifest_sha256": hashlib.sha256(pinned.read_bytes()).hexdigest(),
              "training": state_support(train) | {"all_clock_states": len(states), "ledger_episodes": len(ledger)},
              "fits": fits, "integrity": {"original_score_replay_errors": integrity,
                  "chronological_score_replay_errors": chronological_errors,
                  "protected_inputs_labels_missingness_preserved": True, "pinned_input_hashes_verified": True},
              "chronological_training": {"arms": chrono_report, "folds": folds},
              "comparison": comparison, "gate_checks": gate(comparison),
              "limitations": "Reused known May development. Fixed shallow learners, no hyperparameter or threshold search. X-A also changes tree complexity; difference-in-increments is diagnostic, not a causal interaction estimate. Mixed paths include signal missing flags. Leaf weight/episode support does not establish date generalization. Scores are future-opportunity probabilities, not executable returns. Fixed-fit intervals omit fitting uncertainty."}
    (output/"request318.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("previous", "canonical", "pinned-inputs", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.previous, args.canonical, args.pinned_inputs, args.output_dir)
    print(json.dumps({"arms": result["comparison"]["arms"], "gate_checks": result["gate_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
