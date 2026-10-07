"""Training-only availability audit; no fitted model or evaluation is read."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from victory_trader.prediction_only_sequence_momentum import KEYS, complete_mask, history, target_weights
from victory_trader.state_evolution_momentum import LAG_INPUTS


def audit(previous=Path('prepared'), output=Path('research/request327-training-support.json')):
    source = previous/'official321/request321-chronological-states.parquet'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '637b08447d038361b0027c7503dfe08294c136bb5c80efdf0a4cd9d187383807'
    frame = pd.read_parquet(source)
    clock, values = history(frame), frame[list(LAG_INPUTS)].to_numpy(float)
    ema, anchor, innovations, fixed = (np.full_like(values, np.nan) for _ in range(4))
    last_t = np.full_like(values, np.nan)
    times = frame.decision_t.to_numpy(float)
    ema[clock.first] = anchor[clock.first] = values[clock.first]
    last_t[clock.first] = np.where(np.isfinite(values[clock.first]), times[clock.first, None], np.nan)
    for now in clock.steps:
        prev = clock.previous[now]
        prior, prior_anchor, current = ema[prev], anchor[prev], values[now]
        valid, known = np.isfinite(current), np.isfinite(prior)
        z = np.exp(-(times[now, None]-last_t[prev])/1000/30.)
        innovations[now] = np.where(valid & known, current-prior, np.nan)
        fixed[now] = np.where(valid & np.isfinite(prior_anchor), current-prior_anchor, np.nan)
        ema[now] = np.where(valid, np.where(known, z*prior+(1-z)*current, current), prior)
        anchor[now] = np.where(np.isfinite(prior_anchor), prior_anchor, current)
        last_t[now] = np.where(valid, times[now, None], last_t[prev])
    complete, weights = complete_mask(frame).to_numpy(), target_weights(frame)
    def counts(mask):
        return {'observations': int(mask.sum()), 'complete_target_rows': int((mask & complete).sum()),
            'complete_target_episodes': len(frame.loc[mask & complete, KEYS].drop_duplicates()),
            'complete_target_weight': float(weights[mask].sum()), 'target_days': sorted(frame.loc[mask & complete, 'trading_day'].unique())}
    def digest(array):
        return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()
    result = {'request_id': 327, 'training_support_only': True, 'real_model_fits': 0, 'evaluation_read': False,
        'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'tau_s': 30., 'observations': len(frame),
        'observed_episodes': len(frame[KEYS].drop_duplicates()), 'complete_targets': int(complete.sum()),
        'complete_target_weight': float(weights.sum()),
        'features': {feature: {'EMA_innovation': counts(np.isfinite(innovations[:, j])),
            'first_anchor_innovation': counts(np.isfinite(fixed[:, j]))} for j, feature in enumerate(LAG_INPUTS)},
        'all_seven_EMA_innovations': counts(np.isfinite(innovations).all(axis=1)),
        'at_least_one_EMA_innovation': counts(np.isfinite(innovations).any(axis=1)),
        'per_day': {day: counts(np.isfinite(innovations).any(axis=1) & frame.trading_day.eq(day).to_numpy()) for day in frame.trading_day.unique()},
        'training_path_sha256_little_endian_float64': {'EMA_innovation': digest(innovations), 'first_anchor_innovation': digest(fixed)},
        'method': 'Emit current minus PAST EMA/first available anchor before updating at each original observation.EMA exp(-elapsed since last FINITE feature clock/30).Missing current emits missing and carries feature state/time; first available emits missing then initializes.Censored labels never alter paths.All original rows remain.No real fit/evaluation/window/feature selection.'}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
