"""Request248: fixed-horizon position-signal audit, no controller changes."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from .causal_tradability_admission import TRAIN_DAYS, fetch_second_paths, fit_admission, label_tradability, probability
from .chronological_action_value import CAL_DAYS, CAUSAL_SOURCE_FEATURES, FIT_DAYS, KEYS, PATH_FEATURES, build_states, features
from .config import load_settings
from .execution_costs import modeled_sell_fill
from .massive_client import MassiveClient
from .second_execution_reconstruction import EXPIRY_MS, PRIMARY_LATENCY_MS, first_observed_open

REQUEST_ID=248
VALUE_TRAIN_DAYS=["2026-05-05","2026-05-06","2026-05-07","2026-05-08"]
TEST_DAYS=CAL_DAYS
HORIZONS=(1,2,3,5)
ADMISSION_THRESHOLD=0.60
MIN_TEST_ROWS=250
MIN_SIGN_AUC=0.55
MIN_SPEARMAN=0.05
MIN_SPREAD=0.30

def continuation(now_ref, later_ref):
    a=modeled_sell_fill(float(now_ref), __import__("victory_trader.chronological_action_value",fromlist=["BASE"]).BASE)
    b=modeled_sell_fill(float(later_ref), __import__("victory_trader.chronological_action_value",fromlist=["BASE"]).BASE)
    return float((b/a-1)*100) if np.isfinite(a) and np.isfinite(b) and a>0 else np.nan

def horizon_targets(states, probs, paths, horizon):
    work=states.reset_index(drop=True).copy()
    work["tradability_probability"]=probs
    out=[]
    for key,part in work.groupby(KEYS,sort=False):
        part=part.sort_values("state_t")
        ids=list(part.index)
        times,opens=paths.get((str(key[0]),str(key[1]).upper()),(np.array([],dtype=np.int64),np.array([],dtype=float)))
        for pos,idx in enumerate(ids):
            j=pos+horizon
            if j>=len(ids) or bool(work.at[idx,"terminal"]):
                continue
            jdx=ids[j]
            _,a=first_observed_open(times,opens,decision_t=int(work.at[idx,"state_t"]),latency_ms=PRIMARY_LATENCY_MS,expiry_ms=EXPIRY_MS)
            _,b=first_observed_open(times,opens,decision_t=int(work.at[jdx,"state_t"]),latency_ms=PRIMARY_LATENCY_MS,expiry_ms=EXPIRY_MS)
            r=work.loc[idx].to_dict()
            r["target"]=continuation(a,b) if a is not None and b is not None else np.nan
            out.append(r)
    return pd.DataFrame(out)

def columns(frame):
    return tuple(c for c in dict.fromkeys([*CAUSAL_SOURCE_FEATURES,*PATH_FEATURES,"tradability_probability"]) if c in frame and pd.to_numeric(frame[c],errors="coerce").notna().any())

def fit_model(train):
    valid=np.isfinite(pd.to_numeric(train.target,errors="coerce"))
    train=train.loc[valid].copy()
    cols=columns(train)
    y=train.target.to_numpy(float)
    lo,hi=np.quantile(y,[.005,.995])
    sizes=train.groupby(KEYS)["state_t"].transform("size").to_numpy(float)
    epd=train[KEYS].drop_duplicates().groupby("trading_day").size()
    w=1/sizes/train.trading_day.map(epd).to_numpy(float); w/=w.mean()
    m=HistGradientBoostingRegressor(learning_rate=.05,max_iter=140,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=2.0,early_stopping=False,random_state=20261348)
    m.fit(features(train,cols),np.clip(y,lo,hi),sample_weight=w)
    return m,cols,len(train)

def score(model,cols,test):
    valid=np.isfinite(pd.to_numeric(test.target,errors="coerce"))
    t=test.loc[valid].copy()
    t=t.loc[t.tradability_probability>=ADMISSION_THRESHOLD].copy()
    y=t.target.to_numpy(float)
    pred=model.predict(features(t,cols)) if len(t) else np.array([])
    auc=roc_auc_score(y>0,pred) if len(t) and len(np.unique(y>0))==2 else None
    spear=float(pd.Series(pred).corr(pd.Series(y),method="spearman")) if len(t)>1 else None
    if len(t):
        q25,q75=np.quantile(pred,[.25,.75])
        top=y[pred>=q75]; bot=y[pred<=q25]
        spread=float(np.mean(top)-np.mean(bot)) if len(top) and len(bot) else None
        top_mean=float(np.mean(top)) if len(top) else None
        top_pos=float(np.mean(top>0)) if len(top) else None
    else:
        spread=top_mean=top_pos=None
    return {"rows":int(len(t)),"sign_auc":None if auc is None else float(auc),"spearman":spear,"top_bottom_spread_pct":spread,"top_quartile_mean_pct":top_mean,"top_quartile_positive_rate":top_pos}

def evaluate(fit_path,test_path,output):
    fit=build_states(pd.read_parquet(fit_path)); test=build_states(pd.read_parquet(test_path))
    if sorted(fit.trading_day.astype(str).unique())!=FIT_DAYS: raise ValueError("fit dates changed")
    if sorted(test.trading_day.astype(str).unique())!=TEST_DAYS: raise ValueError("test dates changed")
    all_states=pd.concat([fit,test],ignore_index=True)
    client=MassiveClient(load_settings().massive_api_key,cache_dir=Path("data/cache/request248-second-bars"),request_interval_seconds=.05)
    paths=fetch_second_paths(all_states,client)
    labeled=label_tradability(all_states,paths)
    early=labeled.loc[labeled.trading_day.astype(str).isin(TRAIN_DAYS)]
    adm,adm_cols=fit_admission(early)
    train_states=fit.loc[fit.trading_day.astype(str).isin(VALUE_TRAIN_DAYS)].reset_index(drop=True)
    train_probs=probability(adm,adm_cols,train_states); test_probs=probability(adm,adm_cols,test)
    reports={}
    for h in HORIZONS:
        tr=horizon_targets(train_states,train_probs,paths,h)
        te=horizon_targets(test,test_probs,paths,h)
        m,cols,n=fit_model(tr)
        rep=score(m,cols,te); rep["train_rows"]=int(n)
        rep["passes"]=bool(rep["rows"]>=MIN_TEST_ROWS and (rep["sign_auc"] or 0)>=MIN_SIGN_AUC and (rep["spearman"] or 0)>=MIN_SPEARMAN and (rep["top_bottom_spread_pct"] or 0)>=MIN_SPREAD)
        reports[str(h)]=rep
    result={"request_id":REQUEST_ID,"development_only":True,"opens_new_dates":False,"promotion_eligible":False,"horizons":reports,"signal_horizon_found":any(x["passes"] for x in reports.values()),"api_stats":client.stats.to_dict(),"interpretation":"Diagnostic only. Each horizon is fixed before outcomes; no row chooses its best future exit."}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False)); return 0

def main():
    p=argparse.ArgumentParser(); p.add_argument("--fit-positions",type=Path,required=True); p.add_argument("--calibration-positions",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); return evaluate(a.fit_positions,a.calibration_positions,a.output)
if __name__=="__main__": raise SystemExit(main())
