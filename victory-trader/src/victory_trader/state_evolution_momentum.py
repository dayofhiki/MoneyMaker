"""R319: preregistered causal trajectories on the fixed May cash-state cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical_population_momentum import verify_checkpoint
from .clock_momentum_increment import CONTROL, MOMENTUM
from .compact_momentum_representation import fit_linear, predict_linear
from .context_interaction_momentum import fit_shallow, ranks as first_ranks
from .continuous_population_momentum import WINDOW_MS, fitting_rows, state_support
from .elapsed_momentum_evidence import complete_mask, contexts_for
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS, first_observation, identity_hash
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .pending_exit_integrity import session_limits
from .preentry_momentum_observability import CLOCK_FEATURES, clock_features, entry_labels

REQUEST_ID = 319
SEED = 20265305
HISTORY = ("trajectory_elapsed_s", "trajectory_log_observations", "trajectory_lag_age_s")
LAG_INPUTS = tuple(f"momentum_{n}_30s" for n in (
    "return", "efficiency", "volume_rate_ratio", "transactions_rate_ratio",
    "signed_volume_proxy", "reclaim_prior_high")) + ("momentum_acceleration_10_30",)
DELTAS = tuple("trajectory_delta_"+f for f in LAG_INPUTS)
PACKAGES = {"C": MOMENTUM, "Q": CONTROL, "X": MOMENTUM, "K": CONTROL,
            "S": MOMENTUM+HISTORY, "G": CONTROL+HISTORY, "T": MOMENTUM+HISTORY+DELTAS}
ARMS = tuple(PACKAGES)
CONTRASTS = (("T", "S"), ("T", "C"), ("T", "G"), ("S", "C"),
             ("T", "T0"), ("C", "C0"), ("X", "X0"))
METRICS = ("between_episode_same_day_auc", "auc", "ap", "brier", "within_episode_auc")


def evaluation_manifest(cohort: pd.DataFrame, contexts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(cohort.trading_day) != set(EVAL_DAYS) or cohort.duplicated(KEYS).any():
        raise ValueError("original unique eight-day evaluation cohort required")
    hashes = np.array([identity_hash(str(r.trading_day), str(r.ticker), int(r.hot_t)) for r in cohort.itertuples()])
    if not np.array_equal(hashes, cohort.sample_hash_byte.to_numpy()) or (hashes >= 64).any():
        raise ValueError("original hash-selected membership changed")
    rows, ledger = [], []
    for item in cohort.sort_values(KEYS).to_dict("records"):
        day, ticker, hot = str(item["trading_day"]), str(item["ticker"]), int(item["hot_t"])
        closing = session_limits(day)[1]
        bars = contexts.get((day, ticker), (pd.DataFrame(columns=["t"]), 0, closing))[0]
        ends = bars.t.to_numpy(np.int64)+1000
        chosen = ends[(ends-1000 >= hot) & (ends <= hot+WINDOW_MS) & (ends < min(hot+3600000, closing))]
        if len(chosen) and (np.diff(chosen) <= 0).any():
            raise ValueError("unique increasing completed clocks required")
        first = first_observation(bars, hot, closing)
        if first != (int(chosen[0]) if len(chosen) else None):
            raise ValueError("canonical first clock changed")
        ledger.append({**item, "observed_states": len(chosen), "first_decision_t": first,
                       "missing_reason": None if len(chosen) else "no_eligible_completed_print"})
        rows.extend({**item, "decision_t": int(t), "observation_available": True} for t in chosen)
    return pd.DataFrame(rows, columns=[*cohort.columns, "decision_t", "observation_available"]), pd.DataFrame(ledger)


def trajectory_features(states: pd.DataFrame) -> pd.DataFrame:
    """Use completed prior states, including censored states, never their outcomes."""
    if states.duplicated([*KEYS, "decision_t"]).any() or states.decision_t.isna().any():
        raise ValueError("unique available observation identities required")
    if not set(states.trading_day).issubset((*CROSSFIT_DAYS, *EVAL_DAYS)):
        raise ValueError("trajectory crossed fixed development dates")
    if (states.decision_t.le(states.hot_t) | states.decision_t.gt(states.hot_t+WINDOW_MS)).any():
        raise ValueError("trajectory outside inherited early window")
    result = states.copy().reset_index(drop=True)
    result[list(HISTORY+DELTAS)] = np.nan
    for _, group in result.groupby(KEYS, sort=False):
        ordered = group.sort_values("decision_t")
        t = ordered.decision_t.to_numpy(np.int64)
        values = ordered[list(LAG_INPUTS)].to_numpy(float)
        if np.isinf(values).any():
            raise ValueError("infinite causal trajectory input")
        lag = np.searchsorted(t, t-30000, side="right")-1
        available = lag >= 0
        delta = np.full_like(values, np.nan)
        delta[available] = values[available]-values[lag[available]]
        age = np.full(len(t), np.nan)
        age[available] = (t[available]-t[lag[available]])/1000
        result.loc[ordered.index, list(HISTORY+DELTAS)] = np.c_[
            (t-t[0])/1000, np.log1p(np.arange(1, len(t)+1)), age, delta]
    return result


def evaluation_labels(manifest: pd.DataFrame, contexts: dict) -> pd.DataFrame:
    if not set(manifest.trading_day).issubset(EVAL_DAYS):
        raise ValueError("evaluation labels crossed fixed dates")
    features = clock_features(manifest, contexts)
    labels = []
    for row in features.itertuples():
        bars, _, closing = contexts[(str(row.trading_day), str(row.ticker))]
        labels.append(entry_labels(bars, int(row.decision_t), min(int(row.hot_t)+3600000, closing)))
    result = pd.concat([features.reset_index(drop=True), pd.DataFrame(labels)], axis=1)
    result["missing_reason"] = np.where(result.label_complete, None, "incomplete_entry_or_terminal_fill_label")
    return result


def verify_first(states: pd.DataFrame, saved: pd.DataFrame) -> None:
    current = states.sort_values("decision_t").groupby(KEYS, sort=False).head(1).sort_values(KEYS).reset_index(drop=True)
    original = saved.loc[saved.observation_available].sort_values(KEYS).reset_index(drop=True)
    columns = [*KEYS, "decision_t", "observation_available", "label_complete", CEILING, *CLOCK_FEATURES,
               *(c for c in saved if c.startswith("label_") and c != "label_complete")]
    verify_checkpoint(original[columns], current[columns])


def score(train: pd.DataFrame, held: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if train.empty or not complete_mask(train).all() or not set(train.trading_day).issubset(CROSSFIT_DAYS):
        raise ValueError("nonempty complete training support on fixed dates required")
    if train.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("duplicate training observation")
    result, audits = held.copy(), {}
    for arm, features in PACKAGES.items():
        if arm in ("X", "K"):
            model, transform, prior, audit = fit_shallow(train, features, 2)
        else:
            seed = 20265105 if arm in ("C", "Q") else SEED
            model, transform, prior, audit = fit_linear(train, features, seed)
        result[f"{arm}_p_net_5"] = predict_linear(result, model, transform, prior)
        result[f"{arm}_prior_net_5"] = prior
        audits[arm] = audit | {"fit_days": sorted(train.trading_day.unique())}
    return result, audits


def broadcast_first(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    first = result.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    for arm in ARMS:
        reference = first[[*KEYS, f"{arm}_p_net_5"]].rename(columns={f"{arm}_p_net_5": f"{arm}0_p_net_5"})
        result = result.merge(reference, on=KEYS, how="left", validate="many_to_one", sort=False)
        result[f"{arm}0_prior_net_5"] = result[f"{arm}_prior_net_5"]
    return result


def pair_mass(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted ROC numerator and denominator, with exact tie handling."""
    _, bins = np.unique(p, return_inverse=True)
    return binned_mass(y, bins, weights)


def binned_mass(y: np.ndarray, bins: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    pos = np.bincount(bins, weights=weights*y, minlength=int(bins.max())+1)
    neg = np.bincount(bins, weights=weights*(~y), minlength=len(pos))
    numerator = float(np.sum(pos*(np.cumsum(neg)-.5*neg)))
    return numerator, float(pos.sum()*neg.sum())


class RankCache:
    """Fixed score order; bootstrap whole clusters by weight multiplicity."""
    def __init__(self, frame: pd.DataFrame, arms: tuple[str, ...]):
        self.frame = frame
        self.arms = arms
        self.y = truth(frame[CEILING], 5)
        self.w = _episode_day_weights(frame)
        self.days = list(frame.groupby("trading_day", sort=False).indices.values())
        self.episodes = list(frame.groupby(KEYS, sort=False).indices.values())
        self.priors = {a: frame[f"{a}_prior_net_5"].to_numpy(float) for a in arms}
        self.scores = {a: frame[f"{a}_p_net_5"].to_numpy(float) for a in arms}
        self.bins = {}
        self.within = {}
        for arm, p in self.scores.items():
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError("finite probabilities in [0,1] required")
            self.bins[arm] = np.unique(p, return_inverse=True)[1]
            self.within[arm] = np.array([pair_mass(self.y[i], p[i], self.w[i]) for i in self.episodes])
        self.ep_weights = np.array([self.w[i].sum() for i in self.episodes])
        self.ep_first_indices = np.array([i[0] for i in self.episodes])

    def evaluate(self, multiplicity: np.ndarray | None = None) -> dict:
        m = np.ones(len(self.y)) if multiplicity is None else np.asarray(multiplicity, float)
        if len(m) != len(self.y) or (m < 0).any() or not np.isfinite(m).all() or not (m > 0).any():
            raise ValueError("nonempty finite nonnegative multiplicities required")
        # A cluster cannot resample an individual second from an episode.
        if any(np.ptp(m[i]) != 0 for i in self.episodes):
            raise ValueError("whole episodes must remain together")
        weights = self.w*m
        ep_m = m[self.ep_first_indices]
        result = {}
        for a, p in self.scores.items():
            bins = self.bins[a]
            numerator, denominator = binned_mass(self.y, bins, weights)
            pos = np.bincount(bins, weights=weights*self.y)
            neg = np.bincount(bins, weights=weights*(~self.y), minlength=len(pos))
            tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
            precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=tp+fp > 0)
            ap = float(np.sum(pos[::-1]*precision)/pos.sum()) if denominator > 0 else None
            between_num = between_den = 0.
            for index in self.days:
                n, d = binned_mass(self.y[index], bins[index], weights[index])
                between_num += n
                between_den += d
            within = self.within[a]
            between_num -= float(np.sum(within[:, 0]*ep_m**2))
            between_den -= float(np.sum(within[:, 1]*ep_m**2))
            valid = (within[:, 1] > 0) & (ep_m > 0)
            within_auc = float(np.average(within[valid, 0]/within[valid, 1], weights=self.ep_weights[valid]*ep_m[valid])) if valid.any() else None
            result[a] = {"auc": numerator/denominator if denominator > 0 else None, "ap": ap,
                "between_episode_same_day_auc": between_num/between_den if between_den > 1e-14 else None,
                "within_episode_auc": within_auc, "within_episode_evaluable_episodes": int(valid.sum()),
                "brier": float(np.average((self.y-p)**2, weights=weights)),
                "prior_brier": float(np.average((self.y-self.priors[a])**2, weights=weights))}
        return result


def metrics(frame: pd.DataFrame, arms=ARMS) -> dict:
    complete = frame.loc[complete_mask(frame)].reset_index(drop=True)
    if complete.empty:
        return {a: {**dict.fromkeys((*METRICS, "prior_brier"), None), "within_episode_evaluable_episodes": 0} for a in arms}
    return RankCache(complete, tuple(arms)).evaluate()


def intervals(frame: pd.DataFrame, draws: int = 1000) -> dict:
    complete = frame.loc[complete_mask(frame)].reset_index(drop=True)
    cache = RankCache(complete, (*ARMS, "T0", "C0", "X0"))
    output = {}
    for scheme, columns, seed in (("day", ["trading_day"], 20265350),
                                  ("ticker_day", ["trading_day", "ticker"], 20265351)):
        groups = list(complete.groupby(columns, sort=False).indices.values())
        group_ids = np.zeros(len(complete), dtype=int)
        for j, index in enumerate(groups):
            group_ids[index] = j
        values = {f"{a}_minus_{b}": {v: [] for v in METRICS} for a, b in CONTRASTS}
        rng = np.random.default_rng(seed)
        for _ in range(draws):
            counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            scores = cache.evaluate(counts[group_ids])
            for a, b in CONTRASTS:
                for metric in METRICS:
                    x, y = scores[a][metric], scores[b][metric]
                    if x is not None and y is not None:
                        values[f"{a}_minus_{b}"][metric].append(x-y)
        output[scheme] = {"clusters": len(groups), "draws": draws, "comparisons": {
            name: {f"{m}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None for m, v in report.items()} |
                  {f"{m}_valid_draws": len(v) for m, v in report.items()} for name, report in values.items()}}
        print(f"R319 {scheme} bootstrap complete", flush=True)
    return output


def support(frame: pd.DataFrame, sampled: int) -> dict:
    complete = frame.loc[complete_mask(frame)]
    return {"sampled_episodes": sampled, "observed_states": len(frame),
            "observed_episodes": len(frame[KEYS].drop_duplicates()),
            "complete_states": len(complete), "censored_states": len(frame)-len(complete),
            "complete_episode_coverage": len(complete[KEYS].drop_duplicates())/sampled,
            **state_support(complete),
            "within_episode_both_label_episodes": int(sum(g[CEILING].ge(5).nunique() == 2 for _, g in complete.groupby(KEYS))),
            "lag_available_complete_states": int(complete.trajectory_lag_age_s.notna().sum())}


def gate(report: dict) -> dict:
    arms, support_report = report["arms"], report["support"]
    t = arms["T"]
    def greater(a, b):
        return a is not None and b is not None and a > b
    checks = {"episode_coverage_at_least90pct": support_report["complete_episode_coverage"] >= .9,
        "at_least20_positive_episodes": support_report["positive_episodes"] >= 20,
        "at_least4_positive_days": sum(v["support"]["positive_episodes"] > 0 for v in report["per_day"].values()) >= 4,
        "T_primary_rank_beats_S_C_G": all(greater(t[METRICS[0]], arms[a][METRICS[0]]) for a in ("S", "C", "G")),
        "T_ap_beats_S_C_G": all(greater(t["ap"], arms[a]["ap"]) for a in ("S", "C", "G")),
        "T_brier_beats_S_C_G_and_prior": all(greater(v, t["brier"]) for v in [*(arms[a]["brier"] for a in ("S", "C", "G")), t["prior_brier"]]),
        "T_within_episode_auc_above_half": greater(t["within_episode_auc"], .5)}
    days = [v for v in report["per_day"].values() if v["arms"]["S"][METRICS[0]] is not None]
    checks["T_S_rank_improves_majority_days"] = bool(days) and sum(greater(v["arms"]["T"][METRICS[0]], v["arms"]["S"][METRICS[0]]) for v in days) > len(days)/2
    for contrast in ("T_minus_S", "T_minus_G", "T_minus_T0"):
        for scheme in ("day", "ticker_day"):
            ci = report["paired_intervals"][scheme]["comparisons"][contrast][METRICS[0]+"_ci95"]
            checks[f"{scheme}_{contrast}_rank_ci_positive"] = ci is not None and ci[0] > 0
    return checks


def chronological(states: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pieces, audits = [], {}
    for day in CROSSFIT_DAYS[1:]:
        train = states.loc[states.trading_day.lt(day) & complete_mask(states)]
        held = states.loc[states.trading_day.eq(day)]
        if train.empty or held.empty or train.trading_day.ge(day).any():
            raise ValueError("nonempty preceding-day chronological support required")
        scored, audit = score(train, held)
        pieces.append(broadcast_first(scored))
        audits[day] = audit
        print(f"R319 chronological fold {day} complete", flush=True)
    return pd.concat(pieces, ignore_index=True), audits


def run(inputs: Path, pinned: Path, output: Path) -> dict:
    hashes = json.loads(pinned.read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((inputs/name).read_bytes()).hexdigest() != digest:
            raise ValueError("preregistered input hash changed: "+name)
    original = json.loads((inputs/"official318/request318.json").read_text())
    if original.get("request_id") != 318 or original.get("market_requests") != 0:
        raise ValueError("official R318 provenance required")
    saved = pd.read_parquet(inputs/"official318/request318-states.parquet")
    cohort = pd.read_parquet(inputs/"request311/request311-cohort.parquet")
    verify_checkpoint(cohort, saved[list(cohort)])
    if len(cohort) != 828 or int(complete_mask(saved).sum()) != 650:
        raise ValueError("original first evaluation support changed")
    states = pd.read_parquet(inputs/"official317/request317-training-states.parquet")
    manifest = pd.read_parquet(inputs/"official317/request317-clock-manifest.parquet")
    verify_checkpoint(manifest, states[list(manifest)])
    first = pd.read_parquet(inputs/"official315/request315-training-states.parquet")
    fixed = fitting_rows(first, states)["C"]
    if len(states) != 9325 or len(fixed) != 8792 or len(fixed[KEYS].drop_duplicates()) != 336:
        raise ValueError("original continuous training support changed")
    training = trajectory_features(states)
    train = training.loc[complete_mask(training)]
    # No fitting or label availability can select causal history.
    verify_checkpoint(fixed.reset_index(drop=True), train[list(fixed)].reset_index(drop=True))
    raw = pd.read_parquet(inputs/"request311/request311-raw-seconds.parquet")
    pairs = set(map(tuple, cohort[["trading_day", "ticker"]].to_numpy()))
    contexts = contexts_for(raw, EVAL_DAYS, pairs)
    output.mkdir(parents=True, exist_ok=True)
    eval_manifest, ledger = evaluation_manifest(cohort, contexts)
    eval_manifest.to_parquet(output/"request319-clock-manifest.parquet", index=False, compression="zstd")
    ledger.to_parquet(output/"request319-episode-ledger.parquet", index=False, compression="zstd")
    print(f"R319 froze {len(eval_manifest)} evaluation clocks before labels", flush=True)
    evaluation = trajectory_features(evaluation_labels(eval_manifest, contexts))
    verify_first(evaluation, saved)
    scored, fits = score(train, evaluation)
    scored = broadcast_first(scored)
    first_scored = scored.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    anchor = saved.copy()
    errors = {}
    for arm in ARMS:
        reference = first_scored[[*KEYS, f"{arm}_p_net_5", f"{arm}_prior_net_5"]]
        joined = saved[KEYS].merge(reference, on=KEYS, how="left", validate="one_to_one", sort=False)
        if arm in ("C", "Q", "X", "K"):
            mask = saved.observation_available.astype(bool)
            errors[arm] = float(np.max(np.abs(saved.loc[mask, f"{arm}_p_net_5"].to_numpy()-joined.loc[mask, f"{arm}_p_net_5"].to_numpy())))
            if errors[arm] > 1e-9:
                raise ValueError("R318 first probability replay changed: "+arm)
        if arm in ("S", "G", "T"):
            for suffix in ("p_net_5", "prior_net_5"):
                anchor[f"{arm}_{suffix}"] = joined[f"{arm}_{suffix}"].to_numpy()
    verify_checkpoint(saved, anchor[list(saved)])
    for filename, frame in (("training-states", training), ("states", scored), ("first-states", anchor)):
        frame.to_parquet(output/f"request319-{filename}.parquet", index=False, compression="zstd")
    chrono, folds = chronological(training)
    chrono.to_parquet(output/"request319-chronological-states.parquet", index=False, compression="zstd")
    print("R319 fits and first-clock replay complete; longitudinal bootstrap begins", flush=True)
    compared_arms = (*ARMS, "T0", "C0", "X0")
    report = {"support": support(scored, len(cohort)), "arms": metrics(scored, compared_arms),
        "per_day": {d: {"support": support(scored.loc[scored.trading_day.eq(d)], int(cohort.trading_day.eq(d).sum())),
                         "arms": metrics(scored.loc[scored.trading_day.eq(d)], compared_arms)} for d in EVAL_DAYS},
        "paired_intervals": intervals(scored)}
    elapsed = (scored.decision_t-scored.hot_t)/1000
    phase_report = {f"hot_elapsed_{lo}_{hi}s": {"support": support(scored.loc[elapsed.gt(lo) & elapsed.le(hi)], len(cohort)),
        "arms": metrics(scored.loc[elapsed.gt(lo) & elapsed.le(hi)], compared_arms)} for lo, hi in ((0, 30), (30, 120), (120, 300))}
    for name, mask in (("lag_available", scored.trajectory_lag_age_s.notna()), ("lag_unavailable", scored.trajectory_lag_age_s.isna())):
        phase_report[name] = {"support": support(scored.loc[mask], len(cohort)), "arms": metrics(scored.loc[mask], compared_arms)}
    complete = scored.loc[complete_mask(scored)]
    later_keys = set(map(tuple, complete[KEYS].drop_duplicates().to_numpy()))
    first_keys = set(map(tuple, saved.loc[complete_mask(saved), KEYS].to_numpy()))
    first_positive = set(map(tuple, saved.loc[complete_mask(saved) & saved[CEILING].ge(5), KEYS].to_numpy()))
    ever_positive = set(map(tuple, complete.loc[complete[CEILING].ge(5), KEYS].drop_duplicates().to_numpy()))
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "market_requests": 0, "June_HOLD_opened": False, "final_July_August_opened": False,
        "input_sha256": hashes, "window_ms": WINDOW_MS, "lag_ms": 30000,
        "packages": {a: list(f) for a, f in PACKAGES.items()}, "training": state_support(train), "fits": fits,
        "integrity": {"pinned_input_hashes_verified": True, "first_inputs_labels_missingness_preserved": True,
            "first_probability_replay_max_errors": errors,
            "clock_manifest_sha256": hashlib.sha256((output/"request319-clock-manifest.parquet").read_bytes()).hexdigest()},
        "first_anchor": {"complete_episodes": int(complete_mask(anchor).sum()), "arms": first_ranks(anchor, ("R", *ARMS))},
        "comparison": report, "fixed_phases": phase_report,
        "evolution_support": {"first_complete_episodes": len(first_keys), "ever_complete_episodes": len(later_keys),
            "first_censored_later_complete_episodes": len(later_keys-first_keys),
            "first_positive_episodes": len(first_positive), "ever_positive_episodes": len(ever_positive),
            "new_later_positive_episodes": len(ever_positive-first_positive)},
        "chronological_training": {"arms": metrics(chrono, compared_arms), "folds": folds},
        "gate_checks": gate(report),
        "limitations": "Reused May development. All states are alternative cash decisions, not executed positions. T-S changes representation and dimensions, not an isolated causal order treatment. All-state and first metrics have different label support. Within-episode AUC is conditional on both labels. Missing lag flags carry history support. Fixed-fit cluster multiplicity intervals omit fitting uncertainty. No tuning, winner selection, policy change or sealed-date access."}
    (output/"request319.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("inputs", "pinned-inputs", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.inputs, args.pinned_inputs, args.output_dir)
    print(json.dumps({"support": result["comparison"]["support"], "arms": result["comparison"]["arms"],
                      "gate_checks": result["gate_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
