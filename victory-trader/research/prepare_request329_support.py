"""Training-only feasibility for accumulated current evidence; no real fit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from victory_trader import gated_trajectory_residual_momentum as previous
from victory_trader.compact_momentum_representation import fit_transform
from victory_trader.within_episode_momentum import frozen_transform


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()


def memory_basis(frame, transform):
    """Current evidence enters the EMA at this clock; gate0 only carries state."""
    paths = previous.old.feature_paths(frame)
    active, clock = previous.gate_mask(paths), previous.previous.history(paths)
    current = previous.design(paths, active, 'S', transform)
    times = paths.decision_t.to_numpy(float)
    ema, anchor, before = (np.full((len(frame), 7), np.nan) for _ in range(3))
    last = np.full(len(frame), np.nan)
    first_eligible = np.zeros(len(frame), bool)
    for now in clock.steps:
        past = clock.previous[now]
        before[now] = ema[past]
        ema[now], anchor[now], last[now] = ema[past], anchor[past], last[past]
        chosen = now[active[now]]
        if len(chosen):
            old = clock.previous[chosen]
            known = np.isfinite(last[old])
            z = np.exp(-(times[chosen]-last[old])/1000/30.)
            ema[chosen] = np.where(known[:, None], z[:, None]*ema[old]+(1-z[:, None])*current[chosen], current[chosen])
            anchor[chosen] = np.where(known[:, None], anchor[old], current[chosen])
            last[chosen] = times[chosen]
            first_eligible[chosen] = ~known
    accumulated, anchored = np.zeros_like(current), np.zeros_like(current)
    accumulated[active], anchored[active] = ema[active], anchor[active]
    if not np.isfinite(accumulated).all() or not np.isfinite(anchored).all():
        raise ValueError('finite gated memory coordinates required')
    np.testing.assert_array_equal(accumulated[first_eligible], current[first_eligible])
    np.testing.assert_array_equal(anchored[first_eligible], current[first_eligible])
    return active, current, accumulated, anchored, first_eligible, before, ema, last


def audit(previous_dir=Path('prepared'), output=Path('research/request329-training-support.json')):
    pins = {
        'official321/request321-chronological-states.parquet': '637b08447d038361b0027c7503dfe08294c136bb5c80efdf0a4cd9d187383807',
        'official328/request328-fit-audit.json': 'b2aff1376605824e6515710c7639110f84dc7696caa402d0b72b610fb0b3ea8b',
        'official328/source-sha.txt': '4f785bd56b7bcfb84dedb06e91b12b5c9a09c68f27a0bb626e85965c464a98cb',
    }
    for name, expected in pins.items():
        if hashlib.sha256((previous_dir/name).read_bytes()).hexdigest() != expected:
            raise ValueError('verified input changed: '+name)
    frame = pd.read_parquet(previous_dir/'official321/request321-chronological-states.parquet').reset_index(drop=True)
    audits = json.loads((previous_dir/'official328/request328-fit-audit.json').read_text())
    output_scopes = {}
    for name, source, fit in [('full', frame, audits['full']['S'])]+[
            (day, frame.loc[frame.trading_day.lt(day)], item['fits']['S']) for day, item in audits['folds'].items()]:
        if source.empty:
            output_scopes[name] = {'fit_days': [], 'no_preceding_meta_day': True}
            continue
        if fit['fit_days'] != sorted(source.trading_day.unique()):
            raise ValueError('frozen current transform must match preceding scope')
        transform = frozen_transform(fit['preprocessing'])
        manifest, _ = previous.transition_manifest(source)
        replay = fit_transform(source.loc[manifest.current_index], previous.LAG_INPUTS, manifest.transform_weight.to_numpy()).audit()
        for k in ('lower', 'upper', 'median', 'mean', 'scale'):
            np.testing.assert_allclose(transform.audit()[k], replay[k], atol=1e-9, rtol=0)
        active, current, ema, anchor, first, before, after, last = memory_basis(source, transform)
        complete, y, weights = previous.complete_mask(source).to_numpy(), previous.truth(source[previous.CEILING], 5), previous.previous.target_weights(source)
        carried = active & ~first
        changed = carried & (np.max(np.abs(ema-current), axis=1) > 1e-12)
        def counts(mask):
            return {'observations': int(mask.sum()), 'complete_targets': int((mask & complete).sum()),
                'complete_target_episodes': len(source.loc[mask & complete, previous.KEYS].drop_duplicates()),
                'positive_target_episodes': len(source.loc[mask & complete & y, previous.KEYS].drop_duplicates()),
                'original_complete_weight_mass': float(weights[mask].sum()),
                'positive_target_days': sorted(source.loc[mask & complete & y, 'trading_day'].unique()),
                'negative_target_days': sorted(source.loc[mask & complete & ~y, 'trading_day'].unique()),
                'target_days': sorted(source.loc[mask & complete, 'trading_day'].unique())}
        output_scopes[name] = {'fit_days': sorted(source.trading_day.unique()), 'observations': len(source),
            'complete_weight_mass': float(weights.sum()), 'active': counts(active),
            'first_eligible': counts(first), 'carried_history': counts(carried), 'changed_from_current': counts(changed),
            'maximum_first_eligible_basis_difference': float(np.max(np.abs(ema[first]-current[first]))),
            'path_sha256': {k: digest(v) for k, v in {'current': current, 'EMA_evidence': ema,
                'first_eligible_anchor': anchor, 'EMA_before': before, 'EMA_after': after, 'last_eligible_t': last}.items()}}
    result = {'request_id': 329, 'training_support_only': True, 'evaluation_read': False, 'real_model_fits': 0,
        'input_sha256': pins, 'tau_s': 30., 'scopes': output_scopes,
        'method': 'Freeze corresponding verified R328 S numeric transform. Same all-seven-history gate. First eligible current vector initializes EMA/anchor; subsequent eligible clocks update EMA using elapsed since last ELIGIBLE clock. Inactive clocks carry state/time and emit zero design. Original rows/complete weights retained. No labels enter memory; no fit/evaluation/parameter search.'}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
