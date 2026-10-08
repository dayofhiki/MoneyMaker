"""Post-census gap anatomy from already registered R331 outputs; no new labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from victory_trader import action_target_support_census as census


def gap_bins(values):
    values = np.asarray(values, float)
    return {'at_most_registered3s': int((values <= 3).sum()),
        'over3s_through_bucket30s': int(((values > 3) & (values <= 30)).sum()),
        'over30s_through_hold60s': int(((values > 30) & (values <= 60)).sum()),
        'over60s': int((values > 60).sum()), 'no_later_reference': int(np.isnan(values).sum())}


def analyze(directory):
    result = {'request_id': 331, 'post_result_descriptive_only': True,
        'new_labels': 0, 'model_fits': 0, 'threshold_or_phase_selection': False,
        'provider_gap_is_not_proof_of_actual_fill_failure': True, 'scopes': {}}
    for scope in ('training_primary', 'evaluation'):
        prefix = 'training' if scope == 'training_primary' else 'evaluation'
        states = pd.read_parquet(directory/f'request331-{prefix}-targets.parquet')
        ledger = pd.read_parquet(directory/f'request331-{prefix}-support-ledger.parquet')
        if scope == 'training_primary':
            states = states.loc[states.trading_day.isin(census.PRIMARY_DAYS)]
            ledger = ledger.loc[ledger.trading_day.isin(census.PRIMARY_DAYS)]
        day_counts = ledger.trading_day.value_counts()
        episode_weights = np.array([1/(len(day_counts)*day_counts[d]) for d in ledger.trading_day])
        result['scopes'][scope] = {}
        for arm in census.ARMS:
            frame = census.choose_arm(states, arm)
            compatible = frame.loc[frame.cost_compatible]
            entry_known = compatible.loc[compatible.entry_fill_t.notna()]
            numerator = float(np.sum(episode_weights*ledger[f'{arm}_compatible_resolution_with_original_weights']))
            denominator = float(np.sum(episode_weights*ledger[f'{arm}_compatible_observation_mass']))
            result['scopes'][scope][arm] = {'compatible_clock_count': len(compatible),
                'compatible_reason_counts': compatible.enter_reason.value_counts().to_dict(),
                'compatible_entry_reference_gap_bins_s': gap_bins(compatible.next_entry_reference_gap_s),
                'compatible_exit_reference_gap_bins_given_resolved_entry_s': gap_bins(entry_known.next_exit_reference_gap_s),
                'equal_day_identity_observation_mass_resolution': numerator/denominator if denominator else None,
                'pooled_clock_resolution_registered_gate_estimator': float(compatible.enter_resolved.mean()) if len(compatible) else None,
                'these_two_resolution_estimators_are_distinct': True}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.directory), indent=2, allow_nan=False)+'\n')
