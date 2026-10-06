"""R314: frozen gross opportunity and conditional net conversion experiment."""
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
from sklearn.metrics import average_precision_score, roc_auc_score

from .clock_momentum_increment import MOMENTUM, within_day_auc
from .compact_momentum_representation import fit_linear, fit_transform, independent_weights, predict_linear
from .elapsed_momentum_evidence import complete_mask, contexts_for, replay_inputs
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .preentry_momentum_observability import entry_labels, metrics, vector_net

REQUEST_ID = 314
SEED = 20264905
ARMS = ("R", "D", "P")
STAGES = ("stage_reference_gross_pct", "stage_fill_gross_pct", "stage_whole_net_pct")


def stage_labels(bars: pd.DataFrame, decision_t: int, cap: int) -> dict:
    """Same actual-entry exit opens for every counterfactual; never fill at a close."""
    out = {"stage_whole_complete": False, **dict.fromkeys(STAGES, np.nan)}
    times, opens = bars.t.to_numpy(np.int64), bars.o.to_numpy(float)
    right = int(np.searchsorted(times+1000, decision_t, side="right"))
    entry = int(np.searchsorted(times, decision_t))
    if right == 0 or entry == len(times) or times[entry] >= cap:
        return out
    observed = np.flatnonzero((times >= times[entry]) & (times+1000 <= cap))
    if not len(observed):
        return out
    submissions = np.unique(np.r_[times[observed][times[observed]+1000 < cap]+1000, cap])
    fills = np.searchsorted(times, submissions)
    if (fills >= len(times)).any():
        return out
    reference, price, exits = float(bars.c.iloc[right-1]), opens[entry], opens[fills]
    if not np.isfinite(np.r_[reference, price, exits]).all() or (np.r_[reference, price, exits] <= 0).any():
        raise ValueError("positive finite observed reference and actual fill prices required")
    return {"stage_whole_complete": True,
            "stage_reference_gross_pct": float(100*(exits.max()/reference-1)),
            "stage_fill_gross_pct": float(100*(exits.max()/price-1)),
            "stage_whole_net_pct": float(vector_net(price, exits, DEFAULT_EXECUTION_SCENARIOS[1]).max())}


def attach_stages(source: pd.DataFrame, contexts: dict) -> pd.DataFrame:
    additions = []
    for row in source.itertuples():
        available = bool(getattr(row, "observation_available", True))
        if not available:
            additions.append({"stage_whole_complete": False, **dict.fromkeys(STAGES, np.nan)})
            continue
        key = (str(row.trading_day), str(row.ticker))
        if key not in contexts:
            raise ValueError("missing declared raw context")
        bars, _, closing = contexts[key]
        cap = min(int(row.hot_t)+3600000, closing)
        old = entry_labels(bars, int(row.decision_t), cap)
        if old["label_complete"] != bool(row.label_complete) or not np.isclose(
                old[CEILING], getattr(row, CEILING), atol=1e-9, rtol=0, equal_nan=True):
            raise ValueError("original stopped label failed replay")
        new = stage_labels(bars, int(row.decision_t), cap)
        if not np.isclose(new[STAGES[2]], getattr(row, "label_whole_max_net_pct", np.nan),
                          atol=1e-9, rtol=0, equal_nan=True):
            raise ValueError("original whole-cap label failed replay")
        additions.append(new)
    result = pd.concat([source.reset_index(drop=True), pd.DataFrame(additions)], axis=1)
    common = common_mask(result)
    gross, whole, final = (truth(result.loc[common, field], 5) for field in (STAGES[1], STAGES[2], CEILING))
    if (whole & ~gross).any() or (final & ~whole).any():
        raise ValueError("cost and absorbing-stop opportunities must be nested")
    return result


def common_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.label_complete.astype(bool) & frame.stage_whole_complete.astype(bool) & np.isfinite(frame[CEILING])


def fit_conditional(common: pd.DataFrame, features: tuple[str, ...] = MOMENTUM):
    """Restrict training weights without changing the common population estimand."""
    original_weights = independent_weights(common)
    qualifying = truth(common[STAGES[1]], 5)
    conditional, weights = common.loc[qualifying], original_weights[qualifying]
    audit = {"train_rows": len(conditional), "train_episodes": len(conditional[KEYS].drop_duplicates()),
             "fit_days": sorted(conditional.trading_day.astype(str).unique()),
             "common_weight_sum": float(original_weights.sum()), "weight_sum": float(weights.sum()),
             "conditional_weights_rebalanced": False, "empty_support_fallback": conditional.empty}
    if conditional.empty:
        return None, None, 0., audit | {"prior": 0., "single_class_fallback": True}
    y = truth(conditional[CEILING], 5)
    transform = fit_transform(conditional, features, weights)
    prior = float(np.average(y, weights=weights))
    audit.update(prior=prior, preprocessing=transform.audit(), single_class_fallback=len(np.unique(y)) != 2,
                 train_positive_episodes=len(conditional.loc[y, KEYS].drop_duplicates()))
    if len(np.unique(y)) != 2:
        return None, transform, prior, audit
    model = LogisticRegression(C=.1, l1_ratio=0., solver="lbfgs", fit_intercept=True,
                               max_iter=2000, tol=1e-8, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(transform.apply(conditional), y, sample_weight=weights)
    audit.update(iterations=int(model.n_iter_.max()), intercept=float(model.intercept_[0]),
                 coefficients=model.coef_[0].tolist())
    return model, transform, prior, audit


def score_heads(train: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    common = train.loc[common_mask(train)].copy()
    if common.empty:
        raise ValueError("nonempty common training support required")
    gross = truth(common[STAGES[1]], 5)
    if (truth(common[CEILING], 5) & ~gross).any():
        raise ValueError("final event must be a subset of gross event")
    models, audit = {}, {}
    for arm, fitting, seed in (("R", train, 20264605), ("D", common, SEED),
                               ("G", common.assign(**{CEILING: common[STAGES[1]]}), SEED)):
        model, transform, prior, info = fit_linear(fitting, MOMENTUM, seed)
        models[arm] = (model, transform, prior)
        audit[arm] = info | {"fit_days": sorted(fitting.trading_day.astype(str).unique())}
    model, transform, prior, audit["H"] = fit_conditional(common, MOMENTUM)
    models["H"] = (model, transform, prior)
    prior_error = abs(models["G"][2]*models["H"][2]-models["D"][2])
    if prior_error > 1e-12:
        raise ValueError("conditional prior product lost original training weights")
    scored = evaluation.copy()
    available = scored.observation_available.astype(bool)
    for arm, (model, transform, prior) in models.items():
        scored[f"{arm}_p_net_5"] = np.nan
        if available.any():
            values = predict_linear(scored.loc[available], model, transform, prior) if transform is not None else np.full(int(available.sum()), prior)
            scored.loc[available, f"{arm}_p_net_5"] = values
        scored[f"{arm}_prior_net_5"] = prior
    scored["P_p_net_5"] = scored.G_p_net_5*scored.H_p_net_5
    scored["P_prior_net_5"] = models["G"][2]*models["H"][2]
    return scored, {"models": audit, "common_training_rows": len(common),
                    "common_training_episodes": len(common[KEYS].drop_duplicates()),
                    "excluded_whole_censored_rows": len(train)-len(common), "prior_product_error": prior_error,
                    "training_stages": stage_report(train)}


def stage_report(frame: pd.DataFrame) -> dict:
    common = frame.loc[common_mask(frame)]
    ref, gross, whole, final = (truth(common[field], 5) for field in (*STAGES, CEILING))
    return {"rows": len(frame), "final_complete_rows": int((frame.label_complete & np.isfinite(frame[CEILING])).sum()),
            "common_complete_rows": len(common), "common_complete_episodes": len(common[KEYS].drop_duplicates()),
            "stage_positive_rows": dict(zip(("observed_reference_gross5", "actual_fill_gross5", "whole_base_net5", "stopped_base_net5"),
                                            map(lambda y: int(y.sum()), (ref, gross, whole, final)), strict=True)),
            "stage_positive_episodes": {name: len(common.loc[y, KEYS].drop_duplicates()) for name, y in
                                         zip(("observed_reference_gross5", "actual_fill_gross5", "whole_base_net5", "stopped_base_net5"),
                                             (ref, gross, whole, final), strict=True)},
            "repricing_lost_rows": int((ref & ~gross).sum()), "repricing_gained_rows": int((~ref & gross).sum()),
            "cost_lost_rows": int((gross & ~whole).sum()), "stop_lost_rows": int((whole & ~final).sum()),
            "gross_to_final_conversion": float(final.sum()/gross.sum()) if gross.any() else None,
            "common_complete_per_day": common.groupby("trading_day").size().to_dict()}


def paired_intervals(frame: pd.DataFrame, draws: int = 1000) -> dict:
    if frame.empty:
        return {}
    y, weights = truth(frame[CEILING], 5), _episode_day_weights(frame)
    p = {a: frame[f"{a}_p_net_5"].to_numpy(float) for a in ARMS}
    output = {}
    for i, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        groups = list(frame.groupby(columns, sort=False).indices.values())
        rng = np.random.default_rng(20264950+i)
        values = {f"P_minus_{b}": {"auc": [], "within_day_auc": [], "ap": [], "brier": []} for b in ("D", "R")}
        for _ in range(draws):
            indices = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            sample, w = frame.iloc[indices], weights[indices]
            both = len(np.unique(y[indices])) == 2
            auc = {a: roc_auc_score(y[indices], p[a][indices], sample_weight=w) for a in ARMS} if both else {}
            ap = {a: average_precision_score(y[indices], p[a][indices], sample_weight=w) for a in ARMS} if both else {}
            within = {a: within_day_auc(sample, a, w) for a in ARMS}
            for b in ("D", "R"):
                v = values[f"P_minus_{b}"]
                if both:
                    v["auc"].append(float(auc["P"]-auc[b]))
                    v["ap"].append(float(ap["P"]-ap[b]))
                if within["P"] is not None and within[b] is not None:
                    v["within_day_auc"].append(within["P"]-within[b])
                v["brier"].append(float(np.average((y[indices]-p["P"][indices])**2-(y[indices]-p[b][indices])**2, weights=w)))
        output["day" if i == 0 else "ticker_day"] = {"clusters": len(groups), "draws": draws,
            "comparisons": {name: {f"{metric}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None for metric, v in report.items()} |
                                   {f"{metric}_valid_draws": len(v) for metric, v in report.items()} for name, report in values.items()}}
    return output


def component_diagnostics(frame: pd.DataFrame) -> dict:
    common = frame.loc[common_mask(frame)]
    if common.empty:
        return {}
    gross, final = truth(common[STAGES[1]], 5), truth(common[CEILING], 5)
    original_weights = _episode_day_weights(common)
    result = {}
    for arm, mask, target in (("G", np.ones(len(common), bool), gross), ("H", gross, final)):
        y, p, w = target[mask], common[f"{arm}_p_net_5"].to_numpy(float)[mask], original_weights[mask]
        both = len(np.unique(y)) == 2
        result[arm] = {"rows": int(mask.sum()), "positives": int(y.sum()), "original_population_weight_sum": float(w.sum()),
                       "auc": float(roc_auc_score(y, p, sample_weight=w)) if both else None,
                       "ap": float(average_precision_score(y, p, sample_weight=w)) if both else None,
                       "brier": float(np.average((y-p)**2, weights=w)) if len(y) else None}
    return result


def report(frame: pd.DataFrame, draws: int = 1000) -> dict:
    labeled = frame.loc[complete_mask(frame)]
    def rank(group, arm):
        return metrics(group, arm, 5) | {"within_day_auc": within_day_auc(group, arm)}
    per_day = {day: {"labeled": int(labeled.trading_day.eq(day).sum()),
                     "positive_episodes": int(truth(labeled.loc[labeled.trading_day.eq(day), CEILING], 5).sum()),
                     "arms": {a: rank(labeled.loc[labeled.trading_day.eq(day)], a) for a in ARMS}} for day in EVAL_DAYS}
    return {"sampled_episodes": len(frame), "observed_episodes": int(frame.observation_available.sum()),
            "labeled_episodes": len(labeled), "complete_coverage": len(labeled)/len(frame) if len(frame) else 0.,
            "missing_reasons": frame.missing_reason.fillna("complete").value_counts().to_dict(),
            "arms": {a: rank(labeled, a) for a in ARMS}, "per_day": per_day,
            "common_stages": stage_report(frame), "component_diagnostics": component_diagnostics(frame),
            "paired_intervals": paired_intervals(labeled, draws)}


def gate(result: dict) -> dict:
    p, d, r = (result["arms"][a] for a in ("P", "D", "R"))
    def greater(a, b):
        return a is not None and b is not None and a > b
    days = [v for v in result["per_day"].values() if v["arms"]["D"]["auc"] is not None]
    ci = result.get("paired_intervals", {}).get("ticker_day", {}).get("comparisons", {}).get("P_minus_D", {}).get("within_day_auc_ci95")
    return {"complete_coverage_at_least90pct": result["complete_coverage"] >= .9,
            "at_least20_positive_episodes": p.get("positive_episodes", 0) >= 20,
            "at_least4_positive_days": sum(v["positive_episodes"] > 0 for v in result["per_day"].values()) >= 4,
            "P_within_day_auc_beats_D_and_R": all(greater(p["within_day_auc"], a["within_day_auc"]) for a in (d, r)),
            "P_ap_beats_D": greater(p["ap"], d["ap"]),
            "P_brier_beats_D_and_training_prior": all(greater(v, p.get("brier")) for v in (d.get("brier"), p.get("prior_brier"))),
            "P_D_auc_improves_majority_days": bool(days) and sum(greater(v["arms"]["P"]["auc"], v["arms"]["D"]["auc"]) for v in days) > len(days)/2,
            "ticker_day_P_minus_D_within_day_gain_ci_positive": ci is not None and ci[0] > 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("training", "training-raw", "evaluation", "output", "states-output", "training-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    paths = [args.training/"request310-states.parquet", args.training_raw,
             args.evaluation/"request311-states.parquet", args.evaluation/"request311-cohort.parquet",
             args.evaluation/"request311-raw-seconds.parquet", args.training/"request310.json", args.evaluation/"request311.json"]
    train, evaluation, cohort = (pd.read_parquet(paths[i]) for i in (0, 2, 3))
    def pairs(frame):
        return set(map(tuple, frame[["trading_day", "ticker"]].drop_duplicates().to_numpy()))
    train_contexts = contexts_for(pd.read_parquet(paths[1]), CROSSFIT_DAYS, pairs(train))
    eval_contexts = contexts_for(pd.read_parquet(paths[4]), EVAL_DAYS, pairs(cohort))
    previous, official = (json.loads(paths[i].read_text()) for i in (5, 6))
    integrity = replay_inputs(train, evaluation, cohort, train_contexts, eval_contexts, official, previous)
    train, evaluation = attach_stages(train, train_contexts), attach_stages(evaluation, eval_contexts)
    scored, fitting = score_heads(train, evaluation)
    available = scored.observation_available
    error = float(np.max(np.abs(scored.loc[available, "R_p_net_5"].to_numpy()-evaluation.loc[available, "M_p_net_5"].to_numpy())))
    if error > 1e-9:
        raise ValueError("original frozen M probability replay failed")
    print("R314 all fixed heads fitted; computing paired intervals", flush=True)
    comparison = report(scored)
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
              "June_HOLD_opened": False, "final_July_August_opened": False, "network_market_requests": 0,
              "entries_retuned": False, "packages": {"all_heads": list(MOMENTUM)}, "fit": fitting,
              "integrity": integrity | {"maximum_frozen_M_replay_error": error}, "comparison": comparison,
              "gate_checks": gate(comparison),
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "limitations": "Reused May development; fixed opportunity maxima are not realized exit-policy returns. Whole-cap availability selects training and diagnostic support, but primary evaluation retains all original650 final labels. Observed-close reference is counterfactual. Component targets differ. Conditional weights preserve original common population; no evaluation target controls scoring. Paired intervals condition on fixed fits. No search, live policy or promotion."}
    for path in (args.output, args.states_output, args.training_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.training_output, index=False, compression="zstd")
    scored.to_parquet(args.states_output, index=False, compression="zstd")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"comparison": comparison, "gate_checks": result["gate_checks"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
