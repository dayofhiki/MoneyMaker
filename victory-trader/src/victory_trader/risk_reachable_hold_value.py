"""R299: reachable-state ablation and nested two-stage HOLD policy improvement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .frozen_entry_second_hold_exit import FEATURES, HARD_STOP_PCT, KEYS, net, report
from .hierarchical_crack_entry_controller import _episode_day_weights, _fit, _predict
from .lagged_minute_context import CROSSFIT_DAYS
from .pending_exit_integrity import fill_reference, pending_replay, regular_bars, session_limits

REQUEST_ID = 299
TARGET_PI0 = "wait_follow_pi0_advantage_pct"
TARGET_PI1 = "wait_follow_pi1_advantage_pct"


def mark_reachable(states: pd.DataFrame) -> pd.DataFrame:
    frame = states.sort_values([*KEYS, "decision_t"]).copy()
    frame["risk_reachable_hold"] = False
    frame["forced_stop_t"] = np.nan
    for _, group in frame.groupby(KEYS, sort=False):
        stops = group.loc[group.observed_net_return_pct.le(HARD_STOP_PCT)]
        stop_t = int(stops.decision_t.iloc[0]) if len(stops) else np.inf
        frame.loc[group.index, "risk_reachable_hold"] = group.decision_t.lt(stop_t)
        if np.isfinite(stop_t):
            frame.loc[group.index, "forced_stop_t"] = stop_t
    return frame


def liquidation(bars: pd.DataFrame, entry_price: float, submission_t: int) -> float:
    fill = fill_reference(bars, submission_t)
    return net(entry_price, fill[1]) if fill else np.nan


def respect_risk_horizons(states: pd.DataFrame, contexts: dict) -> pd.DataFrame:
    frame = states.copy()
    for key, group in frame.groupby(KEYS, sort=False):
        bars, _, closing = contexts[key]
        price = float(group.entry_price.iloc[0])
        cap = min(int(group.hot_t.iloc[0]) + 3600000, closing)
        stop_t = float(group.forced_stop_t.iloc[0])
        for horizon in (60, 300):
            values = []
            for row in group.itertuples():
                if not row.risk_reachable_hold:
                    values.append(np.nan)
                    continue
                submission = min(int(row.decision_t) + horizon*1000, cap)
                if np.isfinite(stop_t):
                    submission = min(submission, int(stop_t))
                values.append(liquidation(bars, price, submission) - float(row.exit_now_pct))
            frame.loc[group.index, f"hold_{horizon}s_advantage_pct"] = values
    return frame


def continuation_targets(states: pd.DataFrame, contexts: dict, *, prediction_column: str | None, output_column: str) -> pd.DataFrame:
    """WAIT one observed transition, then follow pi0 or a fixed learned pi1.

    Future fills label the action. Earlier features and runtime actions never
    inspect these values. Stop submission is absorbing even after a rebound.
    """
    frame = states.copy()
    frame[output_column] = np.nan
    for key, group in frame.groupby(KEYS, sort=False):
        g = group.sort_values("decision_t")
        bars, _, closing = contexts[key]
        cap = min(int(g.hot_t.iloc[0]) + 3600000, closing)
        timer = min(int(g.entry_t.iloc[0]) + 600000, cap) if prediction_column is None else cap
        price = float(g.entry_price.iloc[0])
        terminal_value = liquidation(bars, price, timer)
        times = g.decision_t.to_numpy(np.int64)
        exits = g.exit_now_pct.to_numpy(float)
        eligible = g.risk_reachable_hold.to_numpy(bool)
        stop_t = float(g.forced_stop_t.iloc[0])
        stop_value = liquidation(bars, price, int(stop_t)) if np.isfinite(stop_t) else np.nan
        risk = g.observed_net_return_pct.le(HARD_STOP_PCT).to_numpy(bool)
        if prediction_column is None:
            voluntary = g.drawdown_pct.le(-2).to_numpy(bool)
        else:
            predictions = g[prediction_column].to_numpy(float)
            voluntary = ~np.isfinite(predictions) | (predictions <= 0)
        downstream = np.full(len(g), np.nan)
        target = np.full(len(g), np.nan)
        for i in range(len(g)-1, -1, -1):
            if not eligible[i]:
                downstream[i] = stop_value
            elif times[i] >= timer:
                # A counterfactual pi0 whose timer is already expired exits now;
                # it cannot sell retrospectively at an earlier timestamp.
                downstream[i] = exits[i]
            elif risk[i] or voluntary[i]:
                downstream[i] = exits[i]
            elif i+1 == len(g) or times[i+1] >= timer:
                downstream[i] = terminal_value
            else:
                downstream[i] = downstream[i+1]
            if eligible[i]:
                if times[i] >= timer:
                    value = exits[i]
                elif i+1 == len(g) or times[i+1] >= timer:
                    value = terminal_value
                else:
                    value = downstream[i+1]
                # A mandatory stop tied with the timer wins, as in pending_replay.
                if np.isfinite(stop_t) and times[i] < stop_t <= timer and (i+1 == len(g) or times[i+1] >= stop_t):
                    value = stop_value
                target[i] = value - exits[i]
        frame.loc[g.index, output_column] = target
    return frame


def validate_fit_days(train_days: list[str], *, excluded: list[str]) -> None:
    if set(train_days) & set(excluded) or not set(train_days).issubset(CROSSFIT_DAYS):
        raise ValueError("target policy training crossed an excluded day")


def model_fit(frame: pd.DataFrame, target: str, seed: int):
    return _fit(frame.loc[frame.risk_reachable_hold], FEATURES, target, seed, log_target=False, episode_weighting=True)


def crossfit_horizon(states: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for index, held_day in enumerate(CROSSFIT_DAYS):
        train = states.loc[states.trading_day.ne(held_day)]
        held = states.loc[states.trading_day.eq(held_day)].copy()
        validate_fit_days(sorted(train.trading_day.unique()), excluded=[held_day])
        for h in (60, 300):
            model = model_fit(train, f"hold_{h}s_advantage_pct", 20263700+index*10+h)
            held[f"predicted_hold_{h}s_pct"] = _predict(held, model)
        pieces.append(held)
    return pd.concat(pieces, ignore_index=True)


def crossfit_improvement(base_targets: pd.DataFrame, contexts: dict) -> tuple[pd.DataFrame, dict]:
    pieces, provenance = [], {}
    for fold, outer_day in enumerate(CROSSFIT_DAYS):
        outer_train = base_targets.loc[base_targets.trading_day.ne(outer_day)]
        outer_test = base_targets.loc[base_targets.trading_day.eq(outer_day)].copy()
        train_days = sorted(outer_train.trading_day.unique())
        validate_fit_days(train_days, excluded=[outer_day])
        inner_pieces, inner_log = [], []
        for inner_index, inner_day in enumerate(train_days):
            fit = outer_train.loc[outer_train.trading_day.ne(inner_day)]
            labeled = outer_train.loc[outer_train.trading_day.eq(inner_day)].copy()
            fit_days = sorted(fit.trading_day.unique())
            validate_fit_days(fit_days, excluded=[outer_day, inner_day])
            pi1 = model_fit(fit, TARGET_PI0, 20263900+fold*100+inner_index)
            labeled["pi1_prediction_pct"] = _predict(labeled, pi1)
            labeled = continuation_targets(labeled, contexts, prediction_column="pi1_prediction_pct", output_column=TARGET_PI1)
            inner_pieces.append(labeled)
            inner_log.append({"target_day":inner_day,"pi1_fit_days":fit_days,"outer_excluded":outer_day})
        inner_targets = pd.concat(inner_pieces, ignore_index=True)
        pi2 = model_fit(inner_targets, TARGET_PI1, 20263950+fold*100)
        outer_test["predicted_next_event_advantage_pct"] = _predict(outer_test, pi2)
        # Outer diagnostic label follows a pi1 fitted only on outer training days.
        # Neither its policy fit nor pi2 has seen the outer day's target labels.
        outer_pi1 = model_fit(outer_train, TARGET_PI0, 20263975+fold*100)
        outer_test["pi1_prediction_pct"] = _predict(outer_test, outer_pi1)
        outer_test = continuation_targets(outer_test, contexts, prediction_column="pi1_prediction_pct", output_column=TARGET_PI1)
        pieces.append(outer_test)
        provenance[outer_day] = {"pi2_fit_days":train_days,"inner_target_policies":inner_log,"eligible_fit_states":int(inner_targets.risk_reachable_hold.sum()),"eligible_test_states":int(outer_test.risk_reachable_hold.sum())}
        print(f"R299 nested HOLD fold complete {outer_day}",flush=True)
    return pd.concat(pieces, ignore_index=True), provenance


def signal(frame: pd.DataFrame, target: str, prediction: str) -> dict:
    eligible = frame.loc[frame.risk_reachable_hold].copy()
    valid = eligible.loc[np.isfinite(eligible[target]) & np.isfinite(eligible[prediction])]
    if len(valid) < 2:
        return {"eligible_states":len(eligible),"valid_states":len(valid),"spearman":None,"day_episode_weighted_rank_correlation":None}
    x = valid[target].rank().to_numpy(float)
    y = valid[prediction].rank().to_numpy(float)
    w = _episode_day_weights(valid)
    x = x - np.average(x,weights=w)
    y = y - np.average(y,weights=w)
    denominator = np.sqrt(np.average(x*x,weights=w)*np.average(y*y,weights=w))
    value = np.average(x*y,weights=w)/denominator if denominator > 0 else np.nan
    correlation = valid[target].corr(valid[prediction],method="spearman")
    return {"eligible_states":len(eligible),"valid_states":len(valid),"spearman":float(correlation) if np.isfinite(correlation) else None,"day_episode_weighted_rank_correlation":float(value) if np.isfinite(value) else None}


def replay_scored(states: pd.DataFrame, contexts: dict, arm: str, *, horizon: int=300, next_event: bool=False) -> pd.DataFrame:
    rows=[]
    for key, group in states.groupby(KEYS,sort=False):
        bars, opening, closing = contexts[key]
        g=group.copy()
        if next_event:
            # The engine parses this field; D has no 300-second decision horizon.
            g["predicted_hold_300s_pct"] = g.predicted_next_event_advantage_pct
        r=pending_replay(g,bars,f"dynamic_{horizon}s",opening,closing)
        r["arm"]=arm
        if next_event:
            r["policy"]="next_event_primary"
        rows.append(r)
    return pd.DataFrame(rows)


def arm_report(primary: pd.DataFrame, old: pd.DataFrame) -> dict:
    # Normalize only the report identifier to reuse paired cluster statistics.
    rows=pd.concat([primary.assign(policy="dynamic_300s"),old],ignore_index=True)
    metrics=report(rows,97)
    output=metrics["dynamic_300s"]
    output["matched_vs_old"]=metrics.get("matched_primary_vs_old",{})
    valid=primary.loc[primary.resolved]
    y=valid.base_net_return_pct.sort_values(ascending=False)
    output["mean_without_largest_trade_pct"]=float(y.iloc[1:].mean()) if len(y)>1 else None
    output["delay_buckets"]=primary.delay_bucket.value_counts().to_dict()
    output["cost_only_stop_breaches"]=int(primary.stop_cost_only_breach.sum())
    best=valid.loc[valid.base_net_return_pct.idxmax()] if len(valid) else None
    output["largest_trade"]={"ticker":str(best.ticker),"day":str(best.trading_day),"base_net_pct":float(best.base_net_return_pct),"delay_s":float(best.execution_delay_s)} if best is not None else None
    return output


def paired_gain(a: pd.DataFrame,b: pd.DataFrame) -> dict:
    pair=a.loc[a.resolved].merge(b.loc[b.resolved],on=KEYS,suffixes=("_a","_b"),validate="one_to_one")
    delta=pair.base_net_return_pct_a-pair.base_net_return_pct_b
    return {"matched_entries":len(pair),"mean_gain_pp":float(delta.mean()) if len(pair) else None}


def verify_reproduction(replayed: pd.DataFrame,original: pd.DataFrame) -> dict:
    pair=replayed.merge(original,on=[*KEYS,"policy"],suffixes=("_replay","_original"),validate="one_to_one")
    if len(pair)!=len(original) or not pair.resolved_replay.eq(pair.resolved_original).all() or not pair.exit_reason_replay.eq(pair.exit_reason_original).all() or not pair.submission_t_replay.eq(pair.submission_t_original).all():
        raise ValueError("R298 primary reproduction mismatch")
    valid=pair.resolved_original
    error=(pair.loc[valid,"base_net_return_pct_replay"]-pair.loc[valid,"base_net_return_pct_original"]).abs()
    if not np.allclose(error,0,atol=1e-9,rtol=0):
        raise ValueError("R298 primary return mismatch")
    return {"entries":len(pair),"maximum_return_error":float(error.max()),"submissions_and_reasons_identical":True}


def main() -> int:
    parser=argparse.ArgumentParser()
    for name in ("states","decisions","raw-seconds","output","states-output","decisions-output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    source=pd.read_parquet(args.states)
    if set(source.trading_day.astype(str))!=set(CROSSFIT_DAYS) or source.duplicated([*KEYS,"decision_t"]).any():
        raise ValueError("R299 accepts unique May5-8 development states only")
    if len(source[KEYS].drop_duplicates())!=86:
        raise ValueError("R299 requires exact 86 positions")
    raw=pd.read_parquet(args.raw_seconds)
    if not set(raw.trading_day.astype(str)).issubset(CROSSFIT_DAYS):
        raise ValueError("new dates in raw second dependency")
    contexts={}
    for key,_ in source.groupby(KEYS,sort=False):
        opening,closing=session_limits(str(key[0]))
        bars=raw.loc[raw.trading_day.astype(str).eq(str(key[0])) & raw.ticker.astype(str).eq(str(key[1]))]
        contexts[key]=(regular_bars(bars,opening,closing),opening,closing)
    source=mark_reachable(source)
    original=pd.read_parquet(args.decisions)
    original=original.loc[original.stage.eq("rebuilt_label_refit")]
    old=original.loc[original.policy.eq("old_10m_2pct_trailing")].copy()
    A=replay_scored(source,contexts,"A_R298")
    reproduction=verify_reproduction(A,original.loc[original.policy.eq("dynamic_300s")])
    print("R299 arm A reproduces R298 exactly",flush=True)
    Bstates=crossfit_horizon(source)
    B=replay_scored(Bstates,contexts,"B_reachable_only")
    risk_labels=respect_risk_horizons(source,contexts)
    Cstates=crossfit_horizon(risk_labels)
    C=replay_scored(Cstates,contexts,"C_risk_consistent_horizon")
    C60=replay_scored(Cstates,contexts,"C_60s_sensitivity",horizon=60)
    pi0targets=continuation_targets(risk_labels,contexts,prediction_column=None,output_column=TARGET_PI0)
    Dstates,folds=crossfit_improvement(pi0targets,contexts)
    D=replay_scored(Dstates,contexts,"D_next_event",next_event=True)
    arms={"A":A,"B":B,"C":C,"D":D,"C60_sensitivity":C60}
    stats={name:arm_report(frame,old) for name,frame in arms.items()}
    signals={"B":signal(Bstates,"hold_300s_advantage_pct","predicted_hold_300s_pct"),"C":signal(Cstates,"hold_300s_advantage_pct","predicted_hold_300s_pct"),"D":signal(Dstates,TARGET_PI1,"predicted_next_event_advantage_pct")}
    attribution={"B_minus_A":paired_gain(B,A),"C_minus_B":paired_gain(C,B),"D_minus_C":paired_gain(D,C),"D_minus_A":paired_gain(D,A)}
    primary=stats["D"]
    rank=signals["D"]["day_episode_weighted_rank_correlation"]
    checks={
        "all_entries_reconciled":all(len(f)==86 for f in arms.values()),
        "resolution_at_least_95pct":primary["resolution_rate"]>=.95,
        "positive_base_mean":(primary["mean_base_net_pct_resolved"] or 0)>0,
        "positive_at_least_three_days":sum(v>0 for v in primary["daily_base_net_pct"].values())>=3,
        "beats_old_matched":primary["matched_vs_old"].get("mean_gain_pp",-np.inf)>0,
        "beats_R298_A":(attribution["D_minus_A"]["mean_gain_pp"] or 0)>0,
        "positive_without_largest_trade":(primary["mean_without_largest_trade_pct"] or 0)>0,
        "positive_weighted_hold_ranking":rank is not None and rank>0,
    }
    result={
        "request_id":REQUEST_ID,"development_only":True,"promotion_eligible":False,"opens_new_dates":False,"June_HOLD_opened":False,
        "candidate_episodes":97,"entries":86,"position_states":len(source),"reachable_hold_states":int(source.risk_reachable_hold.sum()),
        "nonlearned_forced_or_post_stop_states":int((~source.risk_reachable_hold).sum()),
        "entry_risk_costs_features_retuned":False,"network_requests":0,"R298_reproduction":reproduction,
        "primary":"D_next_event","arms":stats,"signals":signals,"attribution":attribution,"outer_and_inner_folds":folds,
        "checks":checks,"research_gate_pass":all(checks.values()),
        "D_target":"WAIT to next completed observed second, then follow pi1; forced risk/cap and pending fills take precedence. pi2 is a bounded two-stage improvement, not a converged fixed point.",
        "pi0_expired_timer":"If the reference pi0 timer has already expired at the counterfactual current state, it exits now and its WAIT advantage is zero; no retrospective sale.",
        "limitations":"Four reused days, upstream entry OOF not nested within HOLD, historical next-print reference fills not guaranteed quotes/fills, event means not account return.",
        "next_boundary":"Analyze ablations without choosing another arm after observing results. If D passes, freeze for separate later-May development; June stays sealed until full HOLD/evaluation freeze.",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.states_output.parent.mkdir(parents=True,exist_ok=True)
    args.decisions_output.parent.mkdir(parents=True,exist_ok=True)
    pd.concat([frame.assign(arm=name) for name,frame in {"A":source,"B":Bstates,"C":Cstates,"D":Dstates}.items()],ignore_index=True).to_parquet(args.states_output,index=False)
    pd.concat(list(arms.values()),ignore_index=True).to_parquet(args.decisions_output,index=False)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({"checks":checks,"research_gate_pass":result["research_gate_pass"]},indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
