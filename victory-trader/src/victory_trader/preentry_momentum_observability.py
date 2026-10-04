"""R306: diagnose cash-state upside with risk/cost labels and clock momentum.

Future paths label outcomes; no hindsight entry or trading policy is selected.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import load_settings
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, modeled_buy_fill
from .feasible_upside_observability import CEILING, classification_metrics, truth
from .frozen_entry_second_hold_exit import HARD_STOP_PCT, KEYS, frozen_entries, net
from .hierarchical_crack_entry_controller import EVENT_FEATURES, _episode_day_weights, _fit, _predict
from .lagged_minute_context import CROSSFIT_DAYS
from .local_turn_entry import TURN_FEATURES
from .massive_client import MassiveClient
from .pending_exit_integrity import regular_bars, session_limits
from .risk_reachable_hold_value import validate_fit_days
from .second_path_attention_probe import SECOND_CACHE_DIR, _second_frame

REQUEST_ID = 306
LEVELS = (0, 5, 10, 20)
SNAPSHOTS = (0, 20, 60, 120)
STATIC_FEATURES = (
    "log_current_price", "minutes_since_open", "minutes_to_close",
    "return_from_previous_close_pct", "minute_body_return_pct", "minute_range_pct",
    "minute_return_1m_pct", "return_accel_1m_pct", "volume_ratio_prev1", "transactions_ratio_prev1",
)
CLOCK_FEATURES = tuple(f"momentum_{name}_{window}s" for window in (10, 30, 120) for name in (
    "return", "efficiency", "volume_rate_ratio", "transactions_rate_ratio",
    "signed_volume_proxy", "reclaim_prior_high", "activity_fraction",
)) + ("momentum_gap_s", "known_cost_drag_pct", "momentum_acceleration_10_30")


def validate_states(source: pd.DataFrame) -> None:
    if (set(source.trading_day.astype(str)) != set(CROSSFIT_DAYS)
            or source.duplicated([*KEYS, "decision_t"]).any()
            or len(source[KEYS].drop_duplicates()) != 97
            or not source.entry_held_out_day.astype(str).eq(source.trading_day.astype(str)).all()):
        raise ValueError("R306 requires unique97 original May5-8 OOF candidate episodes")
    if source.decision_t.le(source.hot_t).any() or source.decision_t.gt(source.hot_t+300000).any():
        raise ValueError("saved cash observations exceed original five-minute window")


def baseline_columns(source: pd.DataFrame) -> tuple[str, ...]:
    # Explicit unsupervised allowlist: no saved fitted/model score is admitted.
    current = tuple(name for name in source if name.startswith("current_"))
    columns = tuple(dict.fromkeys((*STATIC_FEATURES, *current, *EVENT_FEATURES, *TURN_FEATURES)))
    if any(name not in source for name in columns):
        raise ValueError("missing declared causal baseline feature")
    if any(any(word in name for word in ("predicted", "future", "oracle", "execution", "target")) for name in columns):
        raise ValueError("learned/future feature forbidden")
    return columns


def collect_raw(source: pd.DataFrame, raw: pd.DataFrame, *, fetch_missing: bool) -> tuple[dict, dict, pd.DataFrame]:
    if raw.empty or not set(raw.trading_day.astype(str)).issubset(CROSSFIT_DAYS):
        raise ValueError("new raw dates forbidden")
    existing = {(str(d), str(t)): f for (d, t), f in raw.groupby(["trading_day", "ticker"], sort=False)}
    pairs = sorted(set(map(tuple, source[["trading_day", "ticker"]].to_numpy())))
    missing = [key for key in pairs if key not in existing]
    if len(missing) > 11:
        raise ValueError("unexpected acquisition scope; at most11 original missing pairs")
    client = None
    if missing and fetch_missing:
        settings = load_settings()
        client = MassiveClient(settings.massive_api_key, cache_dir=SECOND_CACHE_DIR / "request306", request_interval_seconds=.02)
        for day, ticker in missing:
            if day not in CROSSFIT_DAYS:
                raise ValueError("June acquisition forbidden")
            bars = _second_frame(client.second_bars_range(ticker, date.fromisoformat(day), date.fromisoformat(day), adjusted=False))
            existing[(day, ticker)] = bars.assign(trading_day=day, ticker=ticker)
            print(f"R306 recovered original May raw pair {day} {ticker}", flush=True)
    contexts, parts, uncovered = {}, [], []
    for day, ticker in pairs:
        opening, closing = session_limits(day)
        bars = existing.get((day, ticker))
        if bars is None or bars.empty:
            uncovered.append([day, ticker])
            continue
        bars = regular_bars(bars, opening, closing)
        if bars.empty:
            uncovered.append([day, ticker])
            continue
        contexts[(day, ticker)] = (bars, opening, closing)
        parts.append(bars.assign(trading_day=day, ticker=ticker))
    audit = {"original_pairs": len(pairs), "inherited_pairs": len(existing)-len(missing) if client else len(existing),
             "missing_original_pairs": [list(k) for k in missing], "uncovered_pairs": uncovered,
             "acquisition_scope": "Only missing original May5-8 ticker-days; unadjusted seconds",
             "client_stats": client.stats.to_dict() if client else {"network_requests": 0}}
    return contexts, audit, pd.concat(parts, ignore_index=True) if parts else raw.iloc[:0]


def clock_features(source: pd.DataFrame, contexts: dict) -> pd.DataFrame:
    frame = source.copy()
    for name in CLOCK_FEATURES:
        frame[name] = np.nan
    for key, group in frame.groupby(["trading_day", "ticker"], sort=False):
        if key not in contexts:
            continue
        bars, _, _ = contexts[key]
        ends = bars.t.to_numpy(np.int64)+1000
        close, volume, transactions = (bars[c].to_numpy(float) for c in ("c", "v", "n"))
        if not np.isfinite(np.c_[close, volume, transactions]).all() or (volume < 0).any() or (transactions < 0).any():
            raise ValueError("invalid causal activity data")
        signs = np.sign(np.r_[0., np.diff(close)])
        changes = np.r_[0., np.diff(np.log(close))]
        cumulative = {"volume": np.r_[0., np.cumsum(volume)], "transactions": np.r_[0., np.cumsum(transactions)],
                      "signed": np.r_[0., np.cumsum(signs*volume)], "movement": np.r_[0., np.cumsum(np.abs(changes))]}
        added = []
        for row in group.itertuples():
            t = int(row.decision_t)
            right = int(np.searchsorted(ends, t, side="right"))
            out = {name: np.nan for name in CLOCK_FEATURES}
            if right:
                current = close[right-1]
                out["momentum_gap_s"] = (t-ends[right-1])/1000
                out["known_cost_drag_pct"] = -net(current, current)
                for window in (10, 30, 120):
                    left = int(np.searchsorted(ends, t-window*1000, side="right"))
                    previous = int(np.searchsorted(ends, t-window*2000, side="right"))
                    out[f"momentum_activity_fraction_{window}s"] = (right-left)/window
                    if left > 0:
                        out[f"momentum_return_{window}s"] = 100*(current/close[left-1]-1)
                        movement = cumulative["movement"][right]-cumulative["movement"][left]
                        out[f"momentum_efficiency_{window}s"] = np.log(current/close[left-1])/movement if movement > 0 else 0.
                    current_volume = cumulative["volume"][right]-cumulative["volume"][left]
                    if current_volume > 0:
                        out[f"momentum_signed_volume_proxy_{window}s"] = (cumulative["signed"][right]-cumulative["signed"][left])/current_volume
                    if previous > 0:
                        for name in ("volume", "transactions"):
                            denominator = cumulative[name][left]-cumulative[name][previous]
                            numerator = cumulative[name][right]-cumulative[name][left]
                            out[f"momentum_{name}_rate_ratio_{window}s"] = numerator/denominator if denominator > 0 else np.nan
                        prior_close = close[previous:left]
                        if len(prior_close):
                            out[f"momentum_reclaim_prior_high_{window}s"] = 100*(current/prior_close.max()-1)
                if np.isfinite(out["momentum_return_10s"]) and np.isfinite(out["momentum_return_30s"]):
                    out["momentum_acceleration_10_30"] = out["momentum_return_10s"]-out["momentum_return_30s"]/3
            added.append(out)
        frame.loc[group.index, list(CLOCK_FEATURES)] = pd.DataFrame(added, index=group.index)[list(CLOCK_FEATURES)]
    return frame


def vector_net(entry: float, exits: np.ndarray, scenario) -> np.ndarray:
    buy = modeled_buy_fill(entry, scenario)
    spreads = np.maximum(exits*scenario.half_spread_bps/10000, scenario.min_half_spread_cents/100)
    sell = np.maximum(exits*(1-scenario.slippage_bps/10000)-spreads, 0)
    return 100*(sell*(1-scenario.sell_fee_bps/10000)/buy-1)


def entry_labels(bars: pd.DataFrame, decision_t: int, cap: int) -> dict:
    times, opens, closes = (bars[c].to_numpy() for c in ("t", "o", "c"))
    entry_index = int(np.searchsorted(times, decision_t))
    out = {CEILING: np.nan, "label_complete": False}
    if entry_index == len(times) or times[entry_index] >= cap:
        return out
    entry_t, price = int(times[entry_index]), float(opens[entry_index])
    observed = np.flatnonzero((times >= entry_t) & (times+1000 <= cap))
    if not len(observed):
        return out
    base = DEFAULT_EXECUTION_SCENARIOS[1]
    stop_mask = vector_net(price, closes[observed].astype(float), base) <= HARD_STOP_PCT
    stop_t = int(times[observed[np.flatnonzero(stop_mask)[0]]])+1000 if stop_mask.any() else cap
    terminal = min(stop_t, cap)
    submissions = np.unique(np.r_[times[observed][times[observed]+1000 < terminal]+1000, terminal])
    fills = np.searchsorted(times, submissions)
    if (fills >= len(times)).any():
        return out
    prices = opens[fills].astype(float)
    nets = vector_net(price, prices, base)
    whole_submissions = np.unique(np.r_[times[observed][times[observed]+1000 < cap]+1000, cap])
    whole_fills = np.searchsorted(times, whole_submissions)
    whole_complete = bool((whole_fills < len(times)).all())
    out.update({CEILING: float(nets.max()), "label_complete": True,
                "label_entry_t": entry_t, "label_entry_price": price,
                "label_entry_gap_s": (entry_t-decision_t)/1000,
                "label_stop_t": terminal, "label_stop_cost_only": net(price, price) <= HARD_STOP_PCT,
                "label_reachable_gross_pct": float((prices.max()/price-1)*100),
                "label_whole_max_net_pct": float(vector_net(price, opens[whole_fills].astype(float), base).max()) if whole_complete else np.nan})
    for scenario in DEFAULT_EXECUTION_SCENARIOS:
        out[f"label_max_{scenario.name}_net_pct"] = float(vector_net(price, prices, scenario).max())
    return out


def attach_labels(source: pd.DataFrame, contexts: dict) -> pd.DataFrame:
    labels = []
    for row in source.itertuples():
        context = contexts.get((str(row.trading_day), str(row.ticker)))
        if context is None:
            labels.append({CEILING: np.nan, "label_complete": False})
            continue
        bars, _, closing = context
        labels.append(entry_labels(bars, int(row.decision_t), min(int(row.hot_t)+3600000, closing)))
    return pd.concat([source.reset_index(drop=True), pd.DataFrame(labels)], axis=1)


def fit_probability(train: pd.DataFrame, features: tuple[str, ...], level: int, seed: int):
    y = truth(train[CEILING], level)
    weights = _episode_day_weights(train)
    prior = float(np.average(y, weights=weights))
    if len(np.unique(y)) == 1:
        return None, prior
    model = HistGradientBoostingClassifier(learning_rate=.04, max_iter=220, max_leaf_nodes=15,
        min_samples_leaf=75, l2_regularization=3., early_stopping=False, random_state=seed)
    model.fit(train[list(features)], y, sample_weight=weights)
    return model, prior


def crossfit(frame: pd.DataFrame, features: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    labeled = frame.loc[frame.label_complete].copy()
    parts, provenance = [], {}
    for fold, day in enumerate(CROSSFIT_DAYS):
        train = labeled.loc[labeled.trading_day.ne(day)]
        held = labeled.loc[labeled.trading_day.eq(day)].copy()
        days = sorted(train.trading_day.unique())
        validate_fit_days(days, excluded=[day])
        seed = 20264100+fold*100
        legacy = _fit(train, features, "entry_utility_fixed_pct", seed, log_target=False, episode_weighting=True)
        held["legacy_utility_score"] = _predict(held, legacy)
        audit = {"fit_days": days, "seed": seed, "train_states": len(train),
                 "train_episodes": len(train[KEYS].drop_duplicates()), "models": {}}
        for arm, columns in (("B", features), ("C", features+CLOCK_FEATURES), ("D", features+("known_cost_drag_pct",))):
            for level in LEVELS:
                model, prior = fit_probability(train, columns, level, seed+level)
                held[f"{arm}_p_net_{level}"] = model.predict_proba(held[list(columns)])[:, 1] if model is not None else prior
                held[f"{arm}_prior_net_{level}"] = prior
                audit["models"][f"{arm}_net_{level}"] = {"prior": prior, "single_class_fallback": model is None, "features": list(columns)}
        parts.append(held)
        provenance[day] = audit
        print(f"R306 matched objective/clock fold complete {day}", flush=True)
    return pd.concat(parts, ignore_index=True), provenance


def snapshots(source: pd.DataFrame, seconds: int) -> pd.DataFrame:
    rows = []
    for _, group in source.groupby(KEYS, sort=False):
        ordered = group.sort_values("decision_t")
        target = int(ordered.decision_t.iloc[0])+seconds*1000
        available = ordered.loc[ordered.decision_t.ge(target)]
        if len(available):
            rows.append(available.iloc[:1])
    return pd.concat(rows, ignore_index=True) if rows else source.iloc[:0].copy()


def metrics(frame: pd.DataFrame, arm: str, level: int) -> dict:
    if arm == "A":
        if frame.empty:
            return {"states": 0, "episodes": 0, "auc": None, "ap": None}
        y = truth(frame[CEILING], level)
        weights = _episode_day_weights(frame)
        both = len(np.unique(y)) == 2
        return {"states": len(frame), "episodes": len(frame[KEYS].drop_duplicates()),
                "auc": float(roc_auc_score(y, frame.legacy_utility_score, sample_weight=weights)) if both else None,
                "ap": float(average_precision_score(y, frame.legacy_utility_score, sample_weight=weights)) if both else None,
                "brier": None, "nonprobability_score": True}
    renamed = frame.rename(columns={f"{arm}_p_net_{level}": f"p_net_{level}", f"{arm}_prior_net_{level}": f"prior_net_{level}"})
    return classification_metrics(renamed, level)


def paired_intervals(first: pd.DataFrame) -> dict:
    output = {}
    for index, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        groups = list(first.groupby(columns, sort=False).indices.values())
        if not groups:
            continue
        rng = np.random.default_rng(20264120+index)
        original_weights = _episode_day_weights(first)
        differences = {f"{comparison}_{metric}": [] for comparison in ("B", "D") for metric in ("auc", "ap", "brier")}
        for _ in range(1000):
            indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            sample = first.iloc[indices]
            y, w = truth(sample[CEILING], 5), original_weights[indices]
            if len(np.unique(y)) != 2:
                continue
            c = sample.C_p_net_5.to_numpy(float)
            for comparison in ("B", "D"):
                other = sample[f"{comparison}_p_net_5"].to_numpy(float)
                differences[f"{comparison}_auc"].append(roc_auc_score(y, c, sample_weight=w)-roc_auc_score(y, other, sample_weight=w))
                differences[f"{comparison}_ap"].append(average_precision_score(y, c, sample_weight=w)-average_precision_score(y, other, sample_weight=w))
                differences[f"{comparison}_brier"].append(float(np.average((y-c)**2-(y-other)**2, weights=w)))
        output["day" if index == 0 else "ticker_day"] = {"clusters": len(groups), "draws": 1000,
            **{f"C_minus_{name}_ci95": np.quantile(values, [.025, .975]).tolist() if values else None for name, values in differences.items()}}
    return output


def frozen_reference(source: pd.DataFrame, reference: pd.DataFrame) -> dict:
    entries, candidates = frozen_entries(source)
    pair = entries.merge(reference[KEYS+["entry_t", "entry_price"]], on=KEYS, suffixes=("", "_reference"), validate="one_to_one")
    if len(entries) != 86 or len(pair) != 86 or not pair.entry_t.eq(pair.entry_t_reference).all() or not np.allclose(pair.entry_price, pair.entry_price_reference, atol=1e-9, rtol=0):
        raise ValueError("frozen86 entry reference mismatch")
    return {"candidate_episodes": candidates, "entries": len(entries), "entry_times_prices_identical": True, "no_entry_or_exit_policy_changes": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("states", "raw-seconds", "reference-episodes", "output", "states-output", "raw-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--partial-build-only", action="store_true")
    args = parser.parse_args()
    source = pd.read_parquet(args.states).reset_index(drop=True)
    validate_states(source)
    features = baseline_columns(source)
    reference = frozen_reference(source, pd.read_parquet(args.reference_episodes))
    contexts, acquisition, raw = collect_raw(source, pd.read_parquet(args.raw_seconds), fetch_missing=args.fetch_missing)
    enriched = clock_features(source, contexts)
    frame = attach_labels(enriched, contexts)
    pd.testing.assert_frame_equal(frame[list(features)], source[list(features)])
    for path in (args.output, args.states_output, args.raw_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(args.raw_output, index=False)
    if args.partial_build_only:
        frame.to_parquet(args.states_output, index=False)
        args.output.write_text(json.dumps({"engineering_only": True, "model_fit_performed": False, "acquisition": acquisition, "reference": reference, "labeled_states": int(frame.label_complete.sum())}, indent=2)+'\n')
        return 0
    if acquisition["uncovered_pairs"]:
        raise ValueError("primary research refuses incomplete original97 raw coverage")
    scored, folds = crossfit(frame, features)
    report = {"all_states": {arm: {f"net_{level}": metrics(scored, arm, level) for level in LEVELS} for arm in ("A", "B", "C", "D")}}
    for seconds in SNAPSHOTS:
        saved = snapshots(frame, seconds)
        joined = saved[KEYS+["decision_t"]].merge(scored, on=KEYS+["decision_t"], validate="one_to_one")
        report[f"cash_snapshot_{seconds}s"] = {"original97_episodes": 97, "snapshot_available": len(saved),
            "labeled_snapshots": len(joined), "missing_snapshot": 97-len(saved), "censored_snapshot": len(saved)-len(joined),
            "arms": {arm: {f"net_{level}": metrics(joined, arm, level) for level in LEVELS} for arm in ("A", "B", "C", "D")},
            "per_day_net5": {str(day): {arm: metrics(g, arm, 5) for arm in ("A", "B", "C", "D")} for day, g in joined.groupby("trading_day")}}
        if seconds == 0:
            report[f"cash_snapshot_{seconds}s"]["paired_net5_cluster_intervals"] = paired_intervals(joined)
    initial = report["cash_snapshot_0s"]
    b, c = initial["arms"]["B"]["net_5"], initial["arms"]["C"]["net_5"]
    d = initial["arms"]["D"]["net_5"]
    positive_days = sum(day["B"]["auc"] is not None and day["C"]["auc"] > day["B"]["auc"] for day in initial["per_day_net5"].values())
    signal_checks = {"first_net5_auc_gain": c["auc"] is not None and c["auc"] > b["auc"],
                     "first_net5_ap_gain": c["ap"] is not None and c["ap"] > b["ap"],
                     "first_net5_brier_improvement": c["brier"] < b["brier"], "at_least3_daily_auc_gains": positive_days >= 3,
                     "beats_cost_only_auc": c["auc"] is not None and c["auc"] > d["auc"],
                     "beats_cost_only_ap": c["ap"] is not None and c["ap"] > d["ap"],
                     "beats_cost_only_brier": c["brier"] < d["brier"]}
    labeled = frame.loc[frame.label_complete]
    summary = {"original_states": len(frame), "labeled_states": len(labeled), "original_episodes": 97,
        "labeled_episodes": len(labeled[KEYS].drop_duplicates()), "cost_only_stop_states": int(labeled.label_stop_cost_only.sum()),
        "old_utility_vs_reachable_net_spearman": float(labeled.entry_utility_fixed_pct.corr(labeled[CEILING], method="spearman")),
        "positive_utility_but_no_feasible_net_states": int((labeled.entry_utility_fixed_pct.gt(0) & labeled[CEILING].le(0)).sum()),
        "whole_net5_but_stopped_before_net5_states": int((labeled.label_whole_max_net_pct.ge(5) & labeled[CEILING].lt(5)).sum()),
        "clock_feature_availability": {name: int(frame[name].notna().sum()) for name in CLOCK_FEATURES},
        "entry_gap_seconds_quantiles": labeled.label_entry_gap_s.quantile([.5, .9, .99, 1.]).to_dict()}
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False, "June_HOLD_opened": False,
        "reference": reference, "acquisition": acquisition, "support_and_target_mismatch": summary,
        "baseline_features": list(features), "new_clock_features": list(CLOCK_FEATURES), "outer_folds": folds,
        "observability": report, "primary_signal_checks": signal_checks, "primary_signal_gate_pass": all(signal_checks.values()),
        "integrity_pass": True, "limits": "Reachable upside maxima are future labels, not actual policy return or entry selection. Cash snapshots do not force waiting or predict optimal entry. Original97 shortlist is upstream OOF conditional, not nested. Four reused days, sparse tails and next-print fill assumptions prohibit promotion. No post-result feature/threshold/seed selection."}
    scored.to_parquet(args.states_output, index=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({"primary_signal_checks": signal_checks, "first_net5_B": b, "first_net5_C": c}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
