"""R315: matched canonical first-observation and broad HOT training."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .clock_momentum_increment import CONTROL, MOMENTUM, within_day_auc
from .compact_momentum_representation import fit_linear, predict_linear
from .config import load_settings
from .elapsed_momentum_evidence import complete_mask, contexts_for, replay_inputs
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS, POPULATION_COLUMNS, causal_population, first_observation, identity_hash, validate_training_context
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .massive_client import MassiveClient
from .pending_exit_integrity import regular_bars, session_limits
from .preentry_momentum_observability import CLOCK_FEATURES, clock_features, entry_labels, metrics, snapshots
from .second_path_attention_probe import SECOND_CACHE_DIR, _second_frame

REQUEST_ID = 315
SEED = 20265005
ARMS = ("R", "S", "B", "Q")
CONTRASTS = (("B", "S"), ("B", "R"), ("B", "Q"))


def training_population(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = source[list(POPULATION_COLUMNS)].copy()
    frame = frame.loc[frame.trading_day.astype(str).isin(CROSSFIT_DAYS)].copy()
    frame["ticker"] = frame.ticker.astype(str).str.upper()
    frame["hot_t"] = frame.t.astype(np.int64)
    if set(frame.trading_day.astype(str)) != set(CROSSFIT_DAYS) or frame.duplicated(KEYS).any():
        raise ValueError("complete unique four-day training population required")
    if not frame.t.sub(frame.bar_start_t).eq(60000).all():
        raise ValueError("training HOT must follow completed minute")
    frame["sample_hash_byte"] = [identity_hash(str(r.trading_day), str(r.ticker), int(r.hot_t)) for r in frame.itertuples()]
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    audit = {"population_episodes": len(frame), "sampled_episodes": int(frame.sample_hash_byte.lt(64).sum()),
             "by_day": {d: {"population": int(frame.trading_day.eq(d).sum()),
                             "sampled": int((frame.trading_day.eq(d) & frame.sample_hash_byte.lt(64)).sum())} for d in CROSSFIT_DAYS},
             "selection": "SHA256('R311|day|uppercase_ticker|HOT_ms').digest()[0]<64"}
    return frame, audit


def training_union(population: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    selected = original[KEYS].drop_duplicates()
    membership = population.merge(selected.assign(in_selected=True), on=KEYS, how="left", validate="one_to_one")
    if int(membership.in_selected.fillna(False).sum()) != len(selected):
        raise ValueError("selected training identities absent from population")
    membership["in_selected"] = membership.in_selected.eq(True)
    membership["in_broad"] = membership.sample_hash_byte.lt(64)
    return membership.loc[membership.in_selected | membership.in_broad].sort_values(KEYS).reset_index(drop=True)


def validate_union(cohort: pd.DataFrame) -> None:
    if not set(cohort.trading_day.astype(str)).issubset(CROSSFIT_DAYS) or cohort.duplicated(KEYS).any():
        raise ValueError("acquisition outside declared unique training scope")
    expected = np.array([identity_hash(str(r.trading_day), str(r.ticker), int(r.hot_t)) for r in cohort.itertuples()])
    if not np.array_equal(expected, cohort.sample_hash_byte.to_numpy()) or not np.array_equal(expected < 64, cohort.in_broad.to_numpy(bool)):
        raise ValueError("fixed training hash membership changed")
    if not (cohort.in_broad | cohort.in_selected).all():
        raise ValueError("out-of-cohort identity")


def acquire_contexts(cohort: pd.DataFrame, inherited: pd.DataFrame, client) -> tuple[dict, pd.DataFrame, dict, set]:
    validate_union(cohort)
    pairs = set(map(tuple, cohort[["trading_day", "ticker"]].drop_duplicates().to_numpy()))
    contexts = contexts_for(inherited, CROSSFIT_DAYS, pairs)
    failures, failed, fetched, reused = [], set(), 0, 0
    for i, key in enumerate(sorted(pairs)):
        if key in contexts:
            reused += 1
            continue
        if client is None:
            raise ValueError("missing declared raw history requires configured client")
        day, ticker = key
        opening, closing = session_limits(day)
        try:
            bars = regular_bars(_second_frame(client.second_bars_range(ticker, date.fromisoformat(day), date.fromisoformat(day), adjusted=False)), opening, closing)
            contexts[key] = (bars, opening, closing)
            fetched += 1
        except Exception as error:
            failed.add(key)
            failures.append({"trading_day": day, "ticker": ticker, "error_type": type(error).__name__})
        if (i+1) % 50 == 0:
            print(f"R315 declared training history {i+1}/{len(pairs)}", flush=True)
    parts = [bars.assign(trading_day=day, ticker=ticker) for (day, ticker), (bars, _, _) in contexts.items() if len(bars)]
    raw = pd.concat(parts, ignore_index=True) if parts else inherited.iloc[:0].copy()
    stats = client.stats.to_dict() if client is not None else {"network_requests": 0, "cache_hits": 0, "retries": 0, "cache_writes": 0}
    return contexts, raw, {"declared_pairs": len(pairs), "inherited_pairs": reused, "fetched_valid_pairs": fetched,
                            "failures": failures, "client_stats": stats, "raw_regular_rows": len(raw)}, failed


def observations(cohort: pd.DataFrame, contexts: dict, failed: set | None = None) -> pd.DataFrame:
    validate_union(cohort)
    rows = []
    for item in cohort.to_dict("records"):
        key = (str(item["trading_day"]), str(item["ticker"]))
        closing = session_limits(key[0])[1]
        bars = contexts.get(key, (pd.DataFrame(columns=["t"]), 0, closing))[0]
        decision = first_observation(bars, int(item["hot_t"]), closing)
        row = {**item, "decision_t": np.nan, "observation_available": False, "label_complete": False,
               CEILING: np.nan, "missing_reason": "fetch_or_raw_validation_failure" if key in (failed or set()) else "no_eligible_first_completed_regular_print",
               **dict.fromkeys(CLOCK_FEATURES, np.nan)}
        if decision is not None:
            row.update(decision_t=decision, observation_available=True, missing_reason=None)
            try:
                row.update(clock_features(pd.DataFrame([row]), contexts).iloc[0].to_dict())
            except ValueError:
                row.update(observation_available=False, missing_reason="invalid_causal_activity_inputs")
                rows.append(row)
                continue
            row.update(entry_labels(bars, decision, min(int(item["hot_t"])+3600000, closing)))
            if not row["label_complete"]:
                row["missing_reason"] = "incomplete_entry_or_terminal_fill_label"
        rows.append(row)
    return pd.DataFrame(rows)


def fit_heads(original: pd.DataFrame, canonical: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if not set(original.trading_day.astype(str)).issubset(CROSSFIT_DAYS) or not set(canonical.trading_day.astype(str)).issubset(CROSSFIT_DAYS):
        raise ValueError("fitting crossed fixed training dates")
    if canonical.duplicated(KEYS).any():
        raise ValueError("unique canonical training observations required")
    selected = canonical.loc[canonical.in_selected & complete_mask(canonical)]
    broad = canonical.loc[canonical.in_broad & complete_mask(canonical)]
    if selected.empty or broad.empty:
        raise ValueError("nonempty selected and broad complete training support required")
    result, audits = evaluation.copy(), {}
    for arm, fitting, features, seed in (("R", original, MOMENTUM, 20264605), ("S", selected, MOMENTUM, SEED),
                                         ("B", broad, MOMENTUM, SEED), ("Q", broad, CONTROL, SEED)):
        model, transform, prior, audit = fit_linear(fitting, features, seed)
        available = result.observation_available.astype(bool)
        result[f"{arm}_p_net_5"] = np.nan
        if available.any():
            result.loc[available, f"{arm}_p_net_5"] = predict_linear(result.loc[available], model, transform, prior)
        result[f"{arm}_prior_net_5"] = prior
        audits[arm] = audit | {"fit_days": sorted(fitting.trading_day.astype(str).unique())}
    return result, audits


def chronological_training(canonical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    validate_union(canonical)
    result, folds = canonical.copy(), {}
    for arm in ("B", "Q"):
        result[f"chronological_{arm}_p_net_5"] = np.nan
        result[f"chronological_{arm}_prior_net_5"] = np.nan
    for day in CROSSFIT_DAYS[1:]:
        train = canonical.loc[canonical.in_broad & canonical.trading_day.lt(day) & complete_mask(canonical)]
        held = canonical.loc[canonical.in_broad & canonical.trading_day.eq(day)]
        if train.empty:
            raise ValueError("nonempty earlier broad training required")
        fit_days = sorted(train.trading_day.astype(str).unique())
        if any(d >= day for d in fit_days):
            raise ValueError("chronological fit crossed held day")
        audits = {}
        for arm, features in (("B", MOMENTUM), ("Q", CONTROL)):
            model, transform, prior, audits[arm] = fit_linear(train, features, SEED)
            available = held.observation_available.astype(bool)
            if available.any():
                result.loc[held.loc[available].index, f"chronological_{arm}_p_net_5"] = predict_linear(held.loc[available], model, transform, prior)
            result.loc[held.index, f"chronological_{arm}_prior_net_5"] = prior
        folds[day] = {"fit_days": fit_days, "models": audits}
    return result, folds


def ranks(frame: pd.DataFrame, arms: tuple[str, ...]) -> dict:
    labeled = frame.loc[complete_mask(frame)]
    return {a: metrics(labeled, a, 5) | {"within_day_auc": within_day_auc(labeled, a)} for a in arms}


def intervals(frame: pd.DataFrame, draws: int = 1000) -> dict:
    if frame.empty:
        return {}
    y, weights = truth(frame[CEILING], 5), _episode_day_weights(frame)
    p = {a: frame[f"{a}_p_net_5"].to_numpy(float) for a in ARMS}
    output = {}
    for i, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        groups = list(frame.groupby(columns, sort=False).indices.values())
        values = {f"{a}_minus_{b}": {"auc": [], "within_day_auc": [], "ap": [], "brier": []} for a, b in CONTRASTS}
        rng = np.random.default_rng(20265050+i)
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


def coverage(frame: pd.DataFrame) -> dict:
    return {"sampled_episodes": len(frame), "observed_episodes": int(frame.observation_available.sum()),
            "labeled_episodes": int(complete_mask(frame).sum()),
            "complete_coverage": float(complete_mask(frame).mean()) if len(frame) else 0.,
            "missing_reasons": frame.missing_reason.fillna("complete").value_counts().to_dict()}


def verify_checkpoint(current: pd.DataFrame, saved: pd.DataFrame) -> None:
    # Arrow stores absent optional booleans as null; in-memory dict rows use NaN.
    # Normalize only missing-value representation, preserving positions and values.
    other = saved[list(current)]
    pd.testing.assert_frame_equal(current.where(current.notna(), None), other.where(other.notna(), None),
                                  check_dtype=False, atol=1e-9, rtol=0)


def report(frame: pd.DataFrame, draws: int = 1000) -> dict:
    return coverage(frame) | {"arms": ranks(frame, ARMS),
        "per_day": {d: coverage(frame.loc[frame.trading_day.eq(d)]) | {"arms": ranks(frame.loc[frame.trading_day.eq(d)], ARMS)} for d in EVAL_DAYS},
        "paired_intervals": intervals(frame.loc[complete_mask(frame)], draws)}


def gate(report: dict) -> dict:
    b, others = report["arms"]["B"], [report["arms"][a] for a in ("S", "R", "Q")]
    def greater(x, y):
        return x is not None and y is not None and x > y
    days = [v for v in report["per_day"].values() if v["arms"]["S"]["auc"] is not None]
    result = {"evaluation_coverage_at_least90pct": report["complete_coverage"] >= .9,
              "at_least20_positive_episodes": b.get("positive_episodes", 0) >= 20,
              "at_least4_positive_days": sum(v["arms"]["B"].get("positive_episodes", 0) > 0 for v in report["per_day"].values()) >= 4,
              "B_within_day_auc_beats_S_R_Q": all(greater(b["within_day_auc"], o["within_day_auc"]) for o in others),
              "B_ap_beats_S_R_Q": all(greater(b["ap"], o["ap"]) for o in others),
              "B_brier_beats_S_R_Q_and_prior": all(greater(v, b.get("brier")) for v in [*(o.get("brier") for o in others), b.get("prior_brier")]),
              "B_S_auc_improves_majority_days": bool(days) and sum(greater(v["arms"]["B"]["auc"], v["arms"]["S"]["auc"]) for v in days) > len(days)/2}
    for contrast in ("B_minus_S", "B_minus_Q"):
        ci = report.get("paired_intervals", {}).get("ticker_day", {}).get("comparisons", {}).get(contrast, {}).get("within_day_auc_ci95")
        result[f"ticker_day_{contrast}_within_day_gain_ci_positive"] = ci is not None and ci[0] > 0
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("training", "training-raw", "evaluation", "population", "reference", "output", "states-output", "training-output", "raw-output", "cohort-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    dates = [*CROSSFIT_DAYS, *EVAL_DAYS]
    columns = list(POPULATION_COLUMNS)
    source, reference = (pd.read_parquet(p, columns=columns, filters=[("trading_day", "in", dates)]) for p in (args.population, args.reference))
    train_population, population_audit = training_population(source)
    ref_population, _ = training_population(reference)
    pd.testing.assert_frame_equal(train_population, ref_population, check_dtype=False, atol=1e-12, rtol=1e-12)
    original = pd.read_parquet(args.training/"request310-states.parquet")
    evaluation = pd.read_parquet(args.evaluation/"request311-states.parquet")
    eval_cohort = pd.read_parquet(args.evaluation/"request311-cohort.parquet")
    inherited = pd.read_parquet(args.training_raw)
    expected_eval, _ = causal_population(source)
    reference_eval, _ = causal_population(reference)
    for other in (reference_eval, eval_cohort):
        pd.testing.assert_frame_equal(expected_eval, other, check_dtype=False, atol=1e-12, rtol=1e-12)
    union = training_union(train_population, original)
    for path in (args.output, args.states_output, args.training_output, args.raw_output, args.cohort_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    union.to_parquet(args.cohort_output, index=False, compression="zstd")
    declared_pairs = set(map(tuple, union[["trading_day", "ticker"]].drop_duplicates().to_numpy()))
    if args.checkpoint is None:
        old_pairs = set(map(tuple, inherited[["trading_day", "ticker"]].drop_duplicates().to_numpy()))
        client = MassiveClient(load_settings().massive_api_key, cache_dir=SECOND_CACHE_DIR, request_interval_seconds=.02) if declared_pairs-old_pairs else None
        contexts, raw, acquisition, failed = acquire_contexts(union, inherited, client)
    else:
        saved_cohort = pd.read_parquet(args.checkpoint/"request315-training-cohort.parquet")
        pd.testing.assert_frame_equal(union, saved_cohort, check_dtype=False, atol=1e-12, rtol=1e-12)
        raw = pd.read_parquet(args.checkpoint/"request315-training-raw-seconds.parquet")
        contexts = contexts_for(raw, CROSSFIT_DAYS, declared_pairs)
        checkpoint_summary = args.checkpoint/"request315.json"
        if not checkpoint_summary.exists():
            checkpoint_summary = args.checkpoint/"request315-acquisition.json"
        acquisition = json.loads(checkpoint_summary.read_text())["acquisition"]
        failed = {(v["trading_day"], v["ticker"]) for v in acquisition["failures"]}
    # Preserve costly data even if a subsequent model fit or report fails.
    raw.to_parquet(args.raw_output, index=False, compression="zstd")
    args.output.with_name("request315-acquisition.json").write_text(json.dumps({"acquisition": acquisition,
        "training_cohort_sha256": hashlib.sha256(args.cohort_output.read_bytes()).hexdigest(),
        "training_raw_sha256": hashlib.sha256(args.raw_output.read_bytes()).hexdigest()}, indent=2, allow_nan=False)+"\n")
    canonical = observations(union, contexts, failed)
    canonical.to_parquet(args.training_output, index=False, compression="zstd")
    if args.checkpoint is not None:
        saved = pd.read_parquet(args.checkpoint/"request315-training-states.parquet")
        verify_checkpoint(canonical, saved)
    original_contexts = contexts_for(inherited, CROSSFIT_DAYS, set(map(tuple, original[["trading_day", "ticker"]].drop_duplicates().to_numpy())))
    eval_contexts = contexts_for(pd.read_parquet(args.evaluation/"request311-raw-seconds.parquet"), EVAL_DAYS,
                                set(map(tuple, eval_cohort[["trading_day", "ticker"]].drop_duplicates().to_numpy())))
    official, previous = (json.loads(p.read_text()) for p in (args.evaluation/"request311.json", args.training/"request310.json"))
    integrity = replay_inputs(original, evaluation, eval_cohort, original_contexts, eval_contexts, official, previous)
    integrity["source_context_alignment"] = validate_training_context(original, source)
    scored, fits = fit_heads(original, canonical, evaluation)
    available = scored.observation_available
    error = float(np.max(np.abs(scored.loc[available, "R_p_net_5"].to_numpy()-scored.loc[available, "M_p_net_5"].to_numpy())))
    if error > 1e-9:
        raise ValueError("original frozen M failed replay")
    canonical, folds = chronological_training(canonical)
    chronological = canonical.loc[canonical.in_broad & canonical.trading_day.ne(CROSSFIT_DAYS[0])].copy()
    for arm in ("B", "Q"):
        for suffix in ("p_net_5", "prior_net_5"):
            chronological[f"{arm}_{suffix}"] = chronological[f"chronological_{arm}_{suffix}"]
    first = snapshots(original, 0)
    selected = canonical.loc[canonical.in_selected]
    transition = first[KEYS+["decision_t", CEILING]].merge(selected[KEYS+["decision_t", CEILING, "observation_available", "label_complete"]], on=KEYS, suffixes=("_old", "_canonical"), validate="one_to_one")
    old_y = truth(transition[CEILING+"_old"], 5)
    new_ok = transition.observation_available & transition.label_complete & np.isfinite(transition[CEILING+"_canonical"])
    new_y = truth(transition[CEILING+"_canonical"], 5)
    training_report = {a: coverage(canonical.loc[canonical[flag]]) for a, flag in (("S", "in_selected"), ("B", "in_broad"))}
    training_report["selected_clock_changes"] = {"same_clock": int(transition.decision_t_old.eq(transition.decision_t_canonical).sum()),
        "earlier_canonical_clock": int(transition.decision_t_canonical.lt(transition.decision_t_old).sum()),
        "later_canonical_clock": int(transition.decision_t_canonical.gt(transition.decision_t_old).sum()),
        "old_first_net5_positive": int(old_y.sum()), "canonical_net5_positive": int((new_y & new_ok).sum()),
        "complete_pair_lost_positive": int((old_y & ~new_y & new_ok).sum()), "complete_pair_gained_positive": int((~old_y & new_y & new_ok).sum())}
    print("R315 frozen and chronological fits complete; computing fixed intervals", flush=True)
    comparison = report(scored)
    paths = [args.training/"request310-states.parquet", args.training/"request310.json", args.training_raw,
             args.evaluation/"request311-states.parquet", args.evaluation/"request311-cohort.parquet", args.evaluation/"request311-raw-seconds.parquet", args.evaluation/"request311.json"]
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False, "June_HOLD_opened": False,
              "final_July_August_opened": False, "entries_retuned": False, "evaluation_market_requests": 0,
              "population": population_audit, "training_union_episodes": len(union), "training_overlap": int((union.in_selected & union.in_broad).sum()),
              "acquisition": acquisition, "training": training_report, "fits": fits,
              "chronological_training": coverage(chronological) | {"arms": ranks(chronological, ("B", "Q")), "folds": folds},
              "integrity": integrity | {"maximum_frozen_M_replay_error": error}, "comparison": comparison, "gate_checks": gate(comparison),
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "population_allowed_columns_sha256": hashlib.sha256(source.sort_values(["trading_day", "ticker", "t"]).to_json(orient="split", double_precision=15).encode()).hexdigest(),
              "limitations": "Reused May development, fixed selected versus hash-sampled training supports; complete-label selection and unobserved paths remain. Canonical clock changes labels and fitting rows, not a pure clock effect. Chronological training excludes May5; intervals condition on fixed fits. No inference outcome filters, policy, threshold search or sealed-date access."}
    scored.to_parquet(args.states_output, index=False, compression="zstd")
    canonical.to_parquet(args.training_output, index=False, compression="zstd")
    raw.to_parquet(args.raw_output, index=False, compression="zstd")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"training": training_report, "arms": comparison["arms"], "gate_checks": result["gate_checks"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
