"""R322: past-trained timing trajectories bridged to absolute state opportunity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical_population_momentum import verify_checkpoint
from .compact_momentum_representation import fit_linear, independent_weights, predict_linear
from .elapsed_momentum_evidence import complete_mask
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS
from .lagged_minute_context import CROSSFIT_DAYS
from .state_evolution_momentum import PACKAGES as PREVIOUS_PACKAGES, RankCache, metrics
from .within_episode_momentum import frozen_transform

REQUEST_ID = 322
SEED = 20265605
META_DAYS = CROSSFIT_DAYS[1:]
REPRESENTATIONS = {a: tuple(f"bridge_{a}_{n}" for n in ("current", "from_first", "lag30_change")) for a in ("W", "A")}
PACKAGES = {"B": PREVIOUS_PACKAGES["G"]+REPRESENTATIONS["W"],
            "C": PREVIOUS_PACKAGES["G"]+REPRESENTATIONS["A"],
            "D": PREVIOUS_PACKAGES["G"], "E": PREVIOUS_PACKAGES["T"]}
ARMS = (*PACKAGES, *(a+"0" for a in PACKAGES))
METRICS = ("between_episode_same_day_auc", "auc", "ap", "brier", "within_episode_auc")
CONTRASTS = (("B", "D"), ("B", "C"), ("B", "E"), ("B", "B0"), ("C", "D"), ("E", "D"))


def normalized_head(training: pd.DataFrame, held: pd.DataFrame, audit: dict, arm: str):
    days = audit["fit_days"]
    if not days or not set(days).issubset(CROSSFIT_DAYS) or max(days) >= held.trading_day.min():
        raise ValueError("head must use strictly preceding training dates")
    train = training.loc[training.trading_day.isin(days) & complete_mask(training)]
    if set(train.trading_day) != set(days) or train.empty:
        raise ValueError("frozen head training support unavailable")
    transform = frozen_transform(audit["preprocessing"])
    coefficients = np.asarray(audit["coefficients"], float)
    if len(coefficients) != 2*len(transform.features) or not np.isfinite(coefficients).all():
        raise ValueError("frozen head coefficient dimensions changed")
    intercept = audit.get("intercept", 0.)
    train_score = transform.apply(train)@coefficients+intercept
    held_score = transform.apply(held)@coefficients+intercept
    error = float(np.max(np.abs(held_score-held[f"{arm}_score"].to_numpy(float))))
    if error > 1e-9 or not np.isfinite(held_score).all():
        raise ValueError("saved preceding-head score replay changed")
    weights = independent_weights(train)
    mean = float(np.average(train_score, weights=weights))
    std = float(np.sqrt(np.average((train_score-mean)**2, weights=weights)))
    scale = std if std > 1e-12 else 1.
    return (held_score-mean)/scale, {"fit_days": days, "complete_training_states": len(train),
        "training_episodes": len(train[KEYS].drop_duplicates()), "effective_weight_sum": float(weights.sum()),
        "weighted_mean": mean, "weighted_std": std, "scale": scale, "constant_score_fallback": std <= 1e-12,
        "maximum_saved_score_replay_error": error}


def causal_changes(frame: pd.DataFrame, normalized: dict[str, np.ndarray]) -> pd.DataFrame:
    if frame.empty or frame.duplicated([*KEYS, "decision_t"]).any() or not frame.index.is_unique:
        raise ValueError("nonempty unique causal states required")
    if set(normalized) != set(REPRESENTATIONS) or any(len(v) != len(frame) or not np.isfinite(v).all() for v in normalized.values()):
        raise ValueError("finite current scores for both fixed representations required")
    result = frame.copy()
    for columns in REPRESENTATIONS.values():
        result[list(columns)] = np.nan
    result["bridge_first_decision_t"] = np.nan
    result["bridge_lag_decision_t"] = np.nan
    for _, group in result.groupby(KEYS, sort=False):
        ordered = group.sort_values("decision_t")
        times = ordered.decision_t.to_numpy(np.int64)
        positions = result.index.get_indexer(ordered.index)
        lag = np.searchsorted(times, times-30000, side="right")-1
        available = lag >= 0
        result.loc[ordered.index, "bridge_first_decision_t"] = times[0]
        lag_clocks = np.full(len(times), np.nan)
        lag_clocks[available] = times[lag[available]]
        result.loc[ordered.index, "bridge_lag_decision_t"] = lag_clocks
        for arm, columns in REPRESENTATIONS.items():
            current = np.asarray(normalized[arm])[positions]
            change = np.full(len(times), np.nan)
            change[available] = current[available]-current[lag[available]]
            result.loc[ordered.index, list(columns)] = np.c_[current, current-current[0], change]
    return result


def enrich(training: pd.DataFrame, held: pd.DataFrame, heads: dict) -> tuple[pd.DataFrame, dict]:
    normalized, audits = {}, {}
    for arm in REPRESENTATIONS:
        normalized[arm], audits[arm] = normalized_head(training, held, heads[arm], arm)
    return causal_changes(held, normalized), audits


def fit_admission(meta: pd.DataFrame, held: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if meta.empty or not set(meta.trading_day).issubset(META_DAYS) or meta.trading_day.max() >= held.trading_day.min():
        raise ValueError("complete preceding meta-training dates required")
    if meta.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("duplicate meta state")
    train = meta.loc[complete_mask(meta)]
    if train.empty:
        raise ValueError("complete meta support required")
    scored, audits = held.copy(), {}
    for arm, features in PACKAGES.items():
        model, transform, prior, audit = fit_linear(train, features, SEED)
        scored[f"{arm}_p_net_5"] = predict_linear(held, model, transform, prior)
        scored[f"{arm}_prior_net_5"] = prior
        audits[arm] = audit | {"fit_days": sorted(train.trading_day.unique()), "seed": SEED}
    first = scored.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    for arm in PACKAGES:
        reference = first[[*KEYS, f"{arm}_p_net_5"]].rename(columns={f"{arm}_p_net_5": f"{arm}0_p_net_5"})
        scored = scored.merge(reference, on=KEYS, how="left", validate="many_to_one", sort=False)
        scored[f"{arm}0_prior_net_5"] = scored[f"{arm}_prior_net_5"]
    return scored, audits


def intervals(frame: pd.DataFrame, draws=1000) -> dict:
    complete = frame.loc[complete_mask(frame)].reset_index(drop=True)
    cache, output = RankCache(complete, ARMS), {}
    for scheme, columns, seed in (("day", ["trading_day"], 20265650), ("ticker_day", ["trading_day", "ticker"], 20265651)):
        groups = list(complete.groupby(columns, sort=False).indices.values())
        ids = np.zeros(len(complete), int)
        for j, index in enumerate(groups):
            ids[index] = j
        rng = np.random.default_rng(seed)
        values = {f"{a}_minus_{b}": {m: [] for m in METRICS} for a, b in CONTRASTS}
        for _ in range(draws):
            counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            ranked = cache.evaluate(counts[ids])
            for a, b in CONTRASTS:
                for metric in METRICS:
                    x, y = ranked[a][metric], ranked[b][metric]
                    if x is not None and y is not None:
                        values[f"{a}_minus_{b}"][metric].append(x-y)
        output[scheme] = {"clusters": len(groups), "draws": draws, "comparisons": {
            c: {f"{m}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None for m, v in item.items()} |
               {f"{m}_valid_draws": len(v) for m, v in item.items()} for c, item in values.items()}}
        print(f"R322 {scheme} bootstrap complete", flush=True)
    return output


def gate(report: dict) -> dict:
    arms, rank = report["arms"], "between_episode_same_day_auc"
    def greater(a, b):
        return a is not None and b is not None and a > b
    checks = {"at_least200_meta_training_episodes": report["training_episodes"] >= 200,
        "at_least30_positive_meta_training_episodes": report["positive_training_episodes"] >= 30,
        "at_least4_positive_evaluation_days": sum(v["positive_episodes"] > 0 for v in report["per_day"].values()) >= 4,
        "B_primary_rank_beats_C_D_E_B0": all(greater(arms["B"][rank], arms[a][rank]) for a in ("C", "D", "E", "B0")),
        "B_ap_beats_C_D_E": all(greater(arms["B"]["ap"], arms[a]["ap"]) for a in ("C", "D", "E")),
        "B_brier_beats_C_D_E_and_prior": all(greater(v, arms["B"]["brier"]) for v in [*(arms[a]["brier"] for a in ("C", "D", "E")), arms["B"]["prior_brier"]]),
        "B_timing_above_half": greater(arms["B"]["within_episode_auc"], .5)}
    days = [v["arms"] for v in report["per_day"].values() if v["arms"]["D"][rank] is not None]
    checks["B_D_rank_improves_majority_days"] = bool(days) and sum(greater(v["B"][rank], v["D"][rank]) for v in days) > len(days)/2
    for contrast in ("B_minus_D", "B_minus_C", "B_minus_E", "B_minus_B0"):
        for scheme in ("day", "ticker_day"):
            ci = report["paired_intervals"][scheme]["comparisons"][contrast][f"{rank}_ci95"]
            checks[f"{scheme}_{contrast}_rank_ci_positive"] = ci is not None and ci[0] > 0
    for contrast in ("B_minus_D", "B_minus_C"):
        for scheme in ("day", "ticker_day"):
            ci = report["paired_intervals"][scheme]["comparisons"][contrast]["brier_ci95"]
            checks[f"{scheme}_{contrast}_brier_ci_negative"] = ci is not None and ci[1] < 0
    return checks


def run(previous: Path, pinned: Path, output: Path) -> dict:
    hashes = json.loads(pinned.read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((previous/name).read_bytes()).hexdigest() != digest:
            raise ValueError("preregistered input hash changed: "+name)
    pair_audit = json.loads((previous/"official320/request320.json").read_text())
    age_audit = json.loads((previous/"official321/request321.json").read_text())
    if pair_audit["request_id"] != 320 or age_audit["request_id"] != 321 or age_audit["market_requests"] != 0:
        raise ValueError("official cached stage-one provenance required")
    training = pd.read_parquet(previous/"official319/request319-training-states.parquet")
    saved_meta = pd.read_parquet(previous/"official321/request321-chronological-states.parquet")
    evaluation = pd.read_parquet(previous/"official321/request321-states.parquet")
    ledger = pd.read_parquet(previous/"official319/request319-episode-ledger.parquet")
    manifest = pd.read_parquet(previous/"official319/request319-clock-manifest.parquet")
    verify_checkpoint(manifest, evaluation[list(manifest)])
    if len(saved_meta) != 7806 or len(evaluation) != 19530 or len(ledger) != 828 or set(saved_meta.trading_day) != set(META_DAYS) or set(evaluation.trading_day) != set(EVAL_DAYS):
        raise ValueError("frozen meta/evaluation support changed")
    pieces, normalization = [], {}
    for day in META_DAYS:
        held = saved_meta.loc[saved_meta.trading_day.eq(day)]
        heads = {"W": pair_audit["chronological_training"]["folds"][day]["fits"]["W"],
                 "A": age_audit["chronological_training"]["folds"][day]}
        enriched, normalization[day] = enrich(training, held, heads)
        pieces.append(enriched)
    meta = pd.concat(pieces, ignore_index=True)
    verify_checkpoint(saved_meta, meta[list(saved_meta)])
    complete = meta.loc[complete_mask(meta)]
    if len(complete) != 7327 or len(complete[KEYS].drop_duplicates()) != 256 or len(complete.loc[truth(complete[CEILING], 5), KEYS].drop_duplicates()) != 51:
        raise ValueError("registered meta support changed")
    enriched, normalization["evaluation"] = enrich(training, evaluation, {"W": pair_audit["fits"]["W"], "A": age_audit["fit"]})
    scored, fits = fit_admission(meta, enriched)
    verify_checkpoint(evaluation, scored[list(evaluation)])
    first = scored.sort_values("decision_t").groupby(KEYS, sort=False).head(1).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    for suffix, frame in (("meta-states", meta), ("states", scored), ("first-states", first), ("episode-ledger", ledger)):
        frame.to_parquet(output/f"request322-{suffix}.parquet", index=False, compression="zstd")
    print("R322 four matched admission heads fitted; original states retained", flush=True)
    report = {"training_episodes": 256, "positive_training_episodes": 51,
        "observed_states": len(scored), "complete_states": int(complete_mask(scored).sum()),
        "complete_evaluation_episodes": len(scored.loc[complete_mask(scored), KEYS].drop_duplicates()),
        "ledger_episodes": len(ledger), "arms": metrics(scored, ARMS),
        "per_day": {d: {"arms": metrics(scored.loc[scored.trading_day.eq(d)], ARMS),
            "positive_episodes": len(scored.loc[scored.trading_day.eq(d) & complete_mask(scored) & truth(scored[CEILING], 5), KEYS].drop_duplicates())} for d in EVAL_DAYS},
        "paired_intervals": intervals(scored)}
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "market_requests": 0, "June_HOLD_opened": False, "final_July_August_opened": False,
        "input_sha256": hashes, "packages": {a: list(f) for a, f in PACKAGES.items()},
        "normalization": normalization, "fits": fits, "comparison": report, "first": metrics(first, ARMS),
        "external_saved_references": metrics(scored, ("Q", "G", "X", "X0")), "gate_checks": gate(report),
        "integrity": {"pinned_hashes_verified": True, "preceding_day_stage_one_scope_verified": True,
            "saved_inputs_labels_missingness_scores_preserved": True, "trajectory_before_censor_filter": True,
            "meta_states_sha256": hashlib.sha256((output/"request322-meta-states.parquet").read_bytes()).hexdigest()},
        "limitations": "Known reused May development. Meta-training256 complete episodes/51 positive episodes, only3 dates; past pair heads use8/25/44 mixed episodes. Stage-one/head normalization uses preceding dates only; full train head at evaluation creates support shift. Learned score changes are causal, not an isolated causal market effect. All-state between-episode ranking differs from conditional timing. Fixed-fit intervals omit both stages' fitting uncertainty. No threshold/policy/realized returns, feature/parameter search, promotion or sealed-date access."}
    (output/"request322.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
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
