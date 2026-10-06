"""R317: fixed broad continuous cash-state learning, with no acquisition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .canonical_population_momentum import ranks, validate_union, verify_checkpoint
from .clock_momentum_increment import CONTROL, MOMENTUM, within_day_auc
from .compact_momentum_representation import fit_linear, independent_weights, predict_linear
from .elapsed_momentum_evidence import complete_mask, contexts_for
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS, first_observation
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .pending_exit_integrity import session_limits
from .preentry_momentum_observability import CLOCK_FEATURES, clock_features, entry_labels, validate_states

REQUEST_ID = 317
WINDOW_MS = 300000
SEED = 20265105
ARMS = ("R", "B", "C", "Q", "H")
CONTRASTS = (("C", "B"), ("H", "B"), ("C", "Q"), ("C", "R"))


def clock_manifest(cohort: pd.DataFrame, contexts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bar membership only: do not access any future outcome/price column."""
    validate_union(cohort)
    rows, ledger = [], []
    for item in cohort.loc[cohort.in_broad].sort_values(KEYS).to_dict("records"):
        key = (str(item["trading_day"]), str(item["ticker"]))
        closing = session_limits(key[0])[1]
        bars = contexts.get(key, (pd.DataFrame(columns=["t"]), 0, closing))[0]
        hot = int(item["hot_t"])
        ends = bars.t.to_numpy(np.int64)+1000
        chosen = ends[(ends-1000 >= hot) & (ends <= hot+WINDOW_MS) & (ends < min(hot+3600000, closing))]
        if len(chosen) and (np.diff(chosen) <= 0).any():
            raise ValueError("strictly increasing unique regular observations required")
        canonical = first_observation(bars, hot, closing)
        if canonical != (int(chosen[0]) if len(chosen) else None):
            raise ValueError("continuous schedule changed canonical first clock")
        ledger.append({**item, "observed_states": len(chosen), "first_decision_t": canonical,
                       "missing_reason": None if len(chosen) else "no_eligible_completed_print"})
        rows.extend({**item, "decision_t": int(t), "observation_available": True} for t in chosen)
    return pd.DataFrame(rows, columns=[*cohort.columns, "decision_t", "observation_available"]), pd.DataFrame(ledger)


def label_states(manifest: pd.DataFrame, contexts: dict) -> pd.DataFrame:
    if manifest.empty:
        return manifest.assign(label_complete=pd.Series(dtype=bool), **{CEILING: pd.Series(dtype=float)})
    if manifest.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("duplicate cash decision")
    if not set(manifest.trading_day).issubset(CROSSFIT_DAYS):
        raise ValueError("state labels outside fixed training dates")
    if (manifest.decision_t.le(manifest.hot_t) | manifest.decision_t.gt(manifest.hot_t+WINDOW_MS)).any():
        raise ValueError("state outside inherited early window")
    features = clock_features(manifest, contexts)
    labels = []
    for row in features.itertuples():
        bars, _, closing = contexts[(str(row.trading_day), str(row.ticker))]
        labels.append(entry_labels(bars, int(row.decision_t), min(int(row.hot_t)+3600000, closing)))
    result = pd.concat([features.reset_index(drop=True), pd.DataFrame(labels)], axis=1)
    result["missing_reason"] = np.where(result.label_complete, None, "incomplete_entry_or_terminal_fill_label")
    return result


def fitting_rows(first: pd.DataFrame, states: pd.DataFrame) -> dict:
    validate_union(first)
    if not set(states.trading_day).issubset(CROSSFIT_DAYS) or states.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("continuous training dates/identities changed")
    allowed = first.loc[first.in_broad, KEYS]
    joined = states[KEYS].merge(allowed, on=KEYS, how="left", indicator=True, validate="many_to_one")
    if not joined._merge.eq("both").all():
        raise ValueError("state outside frozen broad membership")
    b = first.loc[first.in_broad & complete_mask(first)].copy()
    c = states.loc[complete_mask(states)].copy()
    h = c.merge(b[KEYS], on=KEYS, how="inner", validate="many_to_one")
    if any(f.empty for f in (b, c, h)):
        raise ValueError("nonempty complete fitting support required")
    if set(map(tuple, h[KEYS].drop_duplicates().to_numpy())) != set(map(tuple, b[KEYS].to_numpy())):
        raise ValueError("matched continuous/first episode support differs")
    # First label and features on the matched set must replay the official first.
    replay = h.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    columns = [*KEYS, "decision_t", CEILING, *CLOCK_FEATURES]
    verify_checkpoint(b[columns].sort_values(KEYS).reset_index(drop=True), replay[columns].sort_values(KEYS).reset_index(drop=True))
    return {"B": b, "C": c, "Q": c, "H": h}


def score_heads(rows: dict, held: pd.DataFrame, original: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    scored, audits = held.copy(), {}
    fits = dict(rows)
    if original is not None:
        validate_states(original)
        fits["R"] = original
    for arm, fitting in fits.items():
        if fitting.empty or not set(fitting.trading_day).issubset(CROSSFIT_DAYS):
            raise ValueError("invalid fitting support")
        features = CONTROL if arm == "Q" else MOMENTUM
        seed = 20264605 if arm == "R" else 20265005 if arm == "B" else SEED
        model, transform, prior, audit = fit_linear(fitting, features, seed)
        available = scored.observation_available.astype(bool)
        scored[f"{arm}_p_net_5"] = np.nan
        if available.any():
            scored.loc[available, f"{arm}_p_net_5"] = predict_linear(scored.loc[available], model, transform, prior)
        scored[f"{arm}_prior_net_5"] = prior
        audits[arm] = audit | {"fit_days": sorted(fitting.trading_day.unique())}
    return scored, audits


def chronological(first: pd.DataFrame, states: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pieces, folds = [], {}
    for day in CROSSFIT_DAYS[1:]:
        earlier_first, earlier_states = first.loc[first.trading_day.lt(day)], states.loc[states.trading_day.lt(day)]
        rows = fitting_rows(earlier_first, earlier_states)
        if any(f.trading_day.ge(day).any() for f in rows.values()):
            raise ValueError("current/future-day chronological fit")
        held = first.loc[first.in_broad & first.trading_day.eq(day)]
        scored, audit = score_heads(rows, held)
        pieces.append(scored)
        folds[day] = audit
    return pd.concat(pieces, ignore_index=True), folds


def intervals(frame: pd.DataFrame, draws: int = 1000) -> dict:
    y, weights = truth(frame[CEILING], 5), _episode_day_weights(frame)
    p = {a: frame[f"{a}_p_net_5"].to_numpy(float) for a in ARMS}
    output = {}
    for i, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        groups = list(frame.groupby(columns, sort=False).indices.values())
        values = {f"{a}_minus_{b}": {"auc": [], "within_day_auc": [], "ap": [], "brier": []} for a, b in CONTRASTS}
        rng = np.random.default_rng(20265150+i)
        for _ in range(draws):
            index = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            sample, w = frame.iloc[index], weights[index]
            both = len(np.unique(y[index])) == 2
            auc = {a: roc_auc_score(y[index], p[a][index], sample_weight=w) for a in ARMS} if both else {}
            ap = {a: average_precision_score(y[index], p[a][index], sample_weight=w) for a in ARMS} if both else {}
            within = {a: within_day_auc(sample, a, w) for a in ARMS}
            for a, b in CONTRASTS:
                v = values[f"{a}_minus_{b}"]
                if both:
                    v["auc"].append(float(auc[a]-auc[b]))
                    v["ap"].append(float(ap[a]-ap[b]))
                if within[a] is not None and within[b] is not None:
                    v["within_day_auc"].append(within[a]-within[b])
                v["brier"].append(float(np.average((y[index]-p[a][index])**2-(y[index]-p[b][index])**2, weights=w)))
        output["day" if i == 0 else "ticker_day"] = {"clusters": len(groups), "draws": draws,
            "comparisons": {name: {f"{metric}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None for metric, v in report.items()} |
                                   {f"{metric}_valid_draws": len(v) for metric, v in report.items()} for name, report in values.items()}}
    return output


def gate(report: dict) -> dict:
    c, others = report["arms"]["C"], [report["arms"][a] for a in ("B", "R", "Q")]
    def greater(x, y):
        return x is not None and y is not None and x > y
    days = [v for v in report["per_day"].values() if v["B"]["auc"] is not None]
    checks = {"evaluation_coverage_at_least90pct": report["complete_coverage"] >= .9,
              "at_least20_positive_episodes": c["positive_episodes"] >= 20,
              "at_least4_positive_days": sum(v["C"]["positive_episodes"] > 0 for v in report["per_day"].values()) >= 4,
              "C_same_day_auc_beats_B_R_Q": all(greater(c["within_day_auc"], v["within_day_auc"]) for v in others),
              "C_ap_beats_B_R_Q": all(greater(c["ap"], v["ap"]) for v in others),
              "C_brier_beats_B_R_Q_and_prior": all(greater(v, c["brier"]) for v in [*(o["brier"] for o in others), c["prior_brier"]]),
              "C_B_auc_improves_majority_days": bool(days) and sum(greater(v["C"]["auc"], v["B"]["auc"]) for v in days) > len(days)/2}
    for contrast in ("C_minus_B", "H_minus_B", "C_minus_Q"):
        ci = report["paired_intervals"]["ticker_day"]["comparisons"][contrast]["within_day_auc_ci95"]
        checks[f"ticker_day_{contrast}_gain_ci_positive"] = ci is not None and ci[0] > 0
    return checks


def state_support(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"states": 0, "episodes": 0, "positive_episodes": 0}
    y = truth(frame[CEILING], 5)
    weights = independent_weights(frame)
    totals = frame[KEYS].assign(weight=weights).groupby(KEYS).weight.sum()
    return {"states": len(frame), "episodes": len(totals), "positive_states": int(y.sum()),
            "positive_episodes": len(frame.loc[y, KEYS].drop_duplicates()),
            "effective_weight_sum": float(weights.sum()), "weighted_prior": float(np.average(y, weights=weights)),
            "episode_weight_min": float(totals.min()), "episode_weight_max": float(totals.max())}


def verify_evaluation(held: pd.DataFrame, saved: pd.DataFrame) -> None:
    # R315 deliberately refitted Q on broad first states; its Q scores replace
    # R311's selected-state Q. Compare every other original input/label/score.
    columns = [c for c in held if c not in ("Q_p_net_5", "Q_prior_net_5")]
    verify_checkpoint(held[columns], saved[columns])


def run(training: Path, first_run: Path, evaluation: Path, output: Path) -> dict:
    summary = json.loads((first_run/"request315.json").read_text())
    if summary.get("request_id") != 315:
        raise ValueError("official R315 required")
    cohort = pd.read_parquet(first_run/"request315-training-cohort.parquet")
    first = pd.read_parquet(first_run/"request315-training-states.parquet")
    raw_path = first_run/"request315-training-raw-seconds.parquet"
    guard = json.loads((first_run/"request315-acquisition.json").read_text())
    for name, key in (("request315-training-cohort.parquet", "training_cohort_sha256"), (raw_path.name, "training_raw_sha256")):
        if hashlib.sha256((first_run/name).read_bytes()).hexdigest() != guard[key]:
            raise ValueError("official R315 input hash changed")
    validate_union(cohort)
    if len(cohort) != 478 or int(cohort.in_broad.sum()) != 407:
        raise ValueError("original frozen broad support changed")
    verify_checkpoint(cohort, first[list(cohort)])
    pairs = set(map(tuple, cohort[["trading_day", "ticker"]].to_numpy()))
    contexts = contexts_for(pd.read_parquet(raw_path), CROSSFIT_DAYS, pairs)
    output.mkdir(parents=True, exist_ok=True)
    manifest, ledger = clock_manifest(cohort, contexts)
    manifest.to_parquet(output/"request317-clock-manifest.parquet", index=False, compression="zstd")
    ledger.to_parquet(output/"request317-episode-ledger.parquet", index=False, compression="zstd")
    manifest_hash = hashlib.sha256((output/"request317-clock-manifest.parquet").read_bytes()).hexdigest()
    print(f"R317 froze {len(manifest)} completed clocks before labels", flush=True)
    states = label_states(manifest, contexts)
    states.to_parquet(output/"request317-training-states.parquet", index=False, compression="zstd")
    rows = fitting_rows(first, states)
    original = pd.read_parquet(training/"request310-states.parquet")
    saved = pd.read_parquet(first_run/"request315-states.parquet")
    held = pd.read_parquet(evaluation/"request311-states.parquet")
    eval_cohort = pd.read_parquet(evaluation/"request311-cohort.parquet")
    verify_evaluation(held, saved)
    verify_checkpoint(eval_cohort, held[list(eval_cohort)])
    if set(held.trading_day) != set(EVAL_DAYS) or len(held) != 828 or int(complete_mask(held).sum()) != 650:
        raise ValueError("original evaluation support changed")
    scored, fits = score_heads(rows, held, original)
    errors = {}
    for arm in ("R", "B"):
        a = scored.loc[scored.observation_available, f"{arm}_p_net_5"].to_numpy(float)
        b = saved.loc[saved.observation_available, f"{arm}_p_net_5"].to_numpy(float)
        errors[arm] = float(np.max(np.abs(a-b)))
        if errors[arm] > 1e-9:
            raise ValueError("frozen R315/R310 model replay changed")
    scored.to_parquet(output/"request317-states.parquet", index=False, compression="zstd")
    chrono, folds = chronological(first, states)
    chrono.to_parquet(output/"request317-chronological-states.parquet", index=False, compression="zstd")
    labeled = scored.loc[complete_mask(scored)]
    print("R317 fixed fits complete; computing paired intervals", flush=True)
    report = {"sampled_episodes": len(scored), "observed_episodes": int(scored.observation_available.sum()),
              "labeled_episodes": len(labeled), "complete_coverage": len(labeled)/len(scored),
              "arms": ranks(scored, ARMS), "per_day": {d: ranks(scored.loc[scored.trading_day.eq(d)], ARMS) for d in EVAL_DAYS},
              "paired_intervals": intervals(labeled)}
    complete_first_keys = set(map(tuple, rows["B"][KEYS].to_numpy()))
    continuous_keys = set(map(tuple, rows["C"][KEYS].drop_duplicates().to_numpy()))
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
              "June_HOLD_opened": False, "final_July_August_opened": False, "market_requests": 0,
              "window_ms": WINDOW_MS, "clock_manifest_sha256": manifest_hash,
              "training": {"broad_episodes": int(cohort.in_broad.sum()), "observed_episodes": int(ledger.observed_states.gt(0).sum()),
                  "clock_states": len(manifest), "complete_states": int(complete_mask(states).sum()),
                  "first_censored_later_complete_episodes": len(continuous_keys-complete_first_keys),
                  "missing_episode_reasons": ledger.missing_reason.fillna("observed").value_counts().to_dict(),
                  "arms": {a: state_support(f) for a, f in rows.items()}},
              "fits": fits, "integrity": {"frozen_score_max_errors": errors, "input_hashes_verified": True,
                  "original_evaluation_inputs_labels_missingness_preserved": True},
              "chronological_training": {"arms": ranks(chrono, ("B", "C", "Q", "H")), "folds": folds},
              "comparison": report, "gate_checks": gate(report),
              "input_sha256": {str(p.parent.name)+"/"+p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [*first_run.glob("request315*.parquet"), first_run/"request315.json", first_run/"request315-acquisition.json",
                            training/"request310-states.parquet", evaluation/"request311-states.parquet", evaluation/"request311-cohort.parquet"]},
              "limitations": "Reused May development. Continuous states are alternative cash decisions, not executed trades. C-B includes schedule and complete-label support changes; H-B matches episode support. HOT background stays fixed. Seconds are correlated, not independent examples. Fixed-fit bootstrap omits fitting uncertainty. No threshold, policy, window search or sealed-date access."}
    (output/"request317.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("training", "first-run", "evaluation", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.training, args.first_run, args.evaluation, args.output_dir)
    print(json.dumps({"training": result["training"], "arms": result["comparison"]["arms"], "gate_checks": result["gate_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
