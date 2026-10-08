"""Post-result support/value diagnostics; no new fit, label or selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from victory_trader import current_state_action_policy as experiment


def analyze(directory):
    result = {}
    for scope, suffix in (('training', 'training-states'), ('evaluation', 'states'), ('chronological', 'chronological-states')):
        frame = pd.read_parquet(directory/f'request330-{suffix}.parquet')
        weights = experiment.original_weights(frame)
        valid = frame.enter_base_pct.notna()
        both = valid & frame.wait_base_pct.notna()
        positive = frame.enter_base_pct.gt(0)
        ranked = valid & frame.R330_Q_ENTER.notna()
        result[scope] = {'all_enter_finite_rows': int(valid.sum()),
            'all_enter_finite_episodes': len(frame.loc[valid, experiment.KEYS].drop_duplicates()),
            'all_enter_original_weight_mass': float(weights[valid].sum()),
            'all_original_episode_mass': float(weights.sum()),
            'all_enter_weighted_mean_pct': float(np.average(frame.loc[valid, 'enter_base_pct'], weights=weights[valid])),
            'all_enter_positive_episodes': len(frame.loc[positive, experiment.KEYS].drop_duplicates()),
            'paired_original_weight_mass': float(weights[both].sum()),
            'predicted_enter_range_pct': [float(frame.R330_Q_ENTER.min()), float(frame.R330_Q_ENTER.max())],
            'predicted_wait_range_pct': [float(frame.R330_Q_WAIT_pi0.min()), float(frame.R330_Q_WAIT_pi0.max())],
            'conditional_enter_positive_auc': float(roc_auc_score(positive[ranked], frame.loc[ranked, 'R330_Q_ENTER'], sample_weight=weights[ranked])),
            'positive_target_quantiles_pct': {str(q): float(v) for q, v in frame.loc[positive, 'enter_base_pct'].quantile([.1, .5, .9, 1.]).items()}}
    frame = pd.read_parquet(directory/'request330-states.parquet')
    episodes = pd.read_parquet(directory/'request330-episode-decisions.parquet')
    result.update(G_first_vs_continuous_identical_selected_entries=bool(episodes.G_selected_index.equals(episodes.G0_selected_index)),
        G_selected_order_reasons=frame.loc[episodes.loc[episodes.G_entered, 'G_selected_index'], 'enter_reason'].value_counts().to_dict(),
        G_resolved_trade_count=int((episodes.G_entered & episodes.G_resolved).sum()),
        G_full_return_is_unknown=True, post_result_descriptive_only=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.directory), indent=2)+'\n')
