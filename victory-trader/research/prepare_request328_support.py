"""R328 training-only gate support; no evaluation or real model fit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from victory_trader.canonical_population_momentum import verify_checkpoint
from victory_trader.feasible_upside_observability import CEILING, truth
from victory_trader.lagged_minute_context import CROSSFIT_DAYS
from victory_trader.latent_state_evolution_momentum import transition_manifest
from victory_trader.observed_path_innovation_momentum import ANCHOR, EMA, feature_paths, path_digest
from victory_trader.prediction_only_sequence_momentum import KEYS, complete_mask, target_weights


def audit(previous=Path('prepared'), output=Path('research/request328-training-support.json')):
    pins = {
        'official321/request321-chronological-states.parquet': '637b08447d038361b0027c7503dfe08294c136bb5c80efdf0a4cd9d187383807',
        'official326/request326-training-states.parquet': '2750032a761eb341a377fb51365e38e258d28b22540615d626ffe9f734464713',
    }
    for name, digest in pins.items():
        if hashlib.sha256((previous/name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen training input changed: '+name)
    raw = pd.read_parquet(previous/next(iter(pins))).reset_index(drop=True)
    saved = pd.read_parquet(previous/'official326/request326-training-states.parquet')
    verify_checkpoint(raw, saved[list(raw)])
    frame = feature_paths(raw)
    gate = np.isfinite(frame[list(EMA)].to_numpy(float)).all(axis=1)
    anchor_gate = np.isfinite(frame[list(ANCHOR)].to_numpy(float)).all(axis=1)
    if not np.array_equal(gate, anchor_gate):
        raise ValueError('EMA/anchor availability differs')
    baseline = saved.R326_B_p_net_5.to_numpy(float)
    if not np.isfinite(baseline).all() or not ((baseline > 0) & (baseline < 1)).all():
        raise ValueError('strictly interior frozen prefix G required')
    complete = complete_mask(frame).to_numpy()
    labels = truth(frame[CEILING], 5)
    weights = target_weights(frame)

    def counts(mask, scope=frame, loss_weights=weights):
        index = scope.index.to_numpy()
        chosen = mask & complete
        active = chosen[index]
        y = labels[index]
        return {
            'observations': int(mask[index].sum()),
            'complete_targets': int(active.sum()),
            'complete_target_episodes': len(scope.loc[active, KEYS].drop_duplicates()),
            'positive_target_rows': int((active & y).sum()),
            'positive_target_episodes': len(scope.loc[active & y, KEYS].drop_duplicates()),
            'negative_target_rows': int((active & ~y).sum()),
            'negative_target_episodes': len(scope.loc[active & ~y, KEYS].drop_duplicates()),
            'original_complete_weight_mass': float(loss_weights[mask[index]].sum()),
            'target_days': sorted(scope.loc[active, 'trading_day'].unique()),
        }

    manifest, _ = transition_manifest(frame)
    current_gate = gate[manifest.current_index.to_numpy()]
    per_day = {day: counts(gate & frame.trading_day.eq(day).to_numpy()) for day in CROSSFIT_DAYS[1:]}
    folds = {}
    for day in CROSSFIT_DAYS[1:]:
        prefix = frame.loc[frame.trading_day.lt(day)]
        fold_weights = target_weights(prefix) if len(prefix) else np.array([])
        folds[day] = {'fit_days': sorted(prefix.trading_day.unique()), 'observations': len(prefix),
            'complete_weight_mass': float(fold_weights.sum()),
            'active': counts(gate, prefix, fold_weights),
            'inactive': counts(~gate, prefix, fold_weights),
            'no_preceding_meta_fit': prefix.empty}
    result = {
        'request_id': 328, 'training_support_only': True, 'evaluation_read': False,
        'real_model_fits': 0, 'input_sha256': pins, 'tau_s': 30.,
        'observations': len(frame), 'observed_episodes': len(frame[KEYS].drop_duplicates()),
        'complete_targets': int(complete.sum()), 'complete_weight_mass': float(weights.sum()),
        'active_all_seven_history': counts(gate), 'inactive': counts(~gate),
        'per_day_active': per_day, 'strict_prefix_fit_support': folds,
        'transform_original_transition_currents': len(manifest),
        'transform_original_weight_mass': float(manifest.transform_weight.sum()),
        'active_transform_currents': int(current_gate.sum()),
        'active_transform_weight_mass': float(manifest.loc[current_gate, 'transform_weight'].sum()),
        'active_prefix_G_min': float(baseline[gate].min()), 'active_prefix_G_max': float(baseline[gate].max()),
        'all_prefix_G_strictly_interior': True, 'EMA_anchor_gate_identical': True,
        'training_path_sha256': {'EMA_innovation': path_digest(frame, EMA), 'first_anchor_innovation': path_digest(frame, ANCHOR)},
        'gate_sha256_uint8': hashlib.sha256(gate.astype('uint8').tobytes()).hexdigest(),
        'prefix_G_sha256_little_endian_float64': hashlib.sha256(np.ascontiguousarray(baseline, dtype='<f8').tobytes()).hexdigest(),
        'method': 'Only frozen out-of-time training records/prefix G; original weights never renormalized by gate. Causal R327 feature paths; all seven past innovations finite. No evaluation outcomes, real fitting, threshold or window selection.',
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
