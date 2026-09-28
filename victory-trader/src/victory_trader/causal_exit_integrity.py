"""Request274: exact-deadline audit of the Request273 exit family.

Only already-opened May5-7 fitting dates are replayed. No new rule is chosen
on development test dates, and missing exits are never cash observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .causal_minute_controller_rebuild import causal_execution_scan
from .full_hot_fixed_policy_value import attach_fixed_value
from .pullback_trailing_exit import (
    RULE_FIT_DAYS, STOP_LOSSES, TRAILS, MAX_HOLD_MINUTES,
    apply_exit_rule, build_pullback_episodes, evaluate_rule_grid,
)
from .rich_post_hot_state import score_and_select_candidates

REQUEST_ID = 274


def evaluate(first_hot_path: Path, scan_path: Path, output: Path, rows_output: Path):
    first_hot = pd.read_parquet(first_hot_path).drop(columns=[
        'fixed_first_watch_value_pct', 'fixed_entry_elapsed_minutes',
    ], errors='ignore')
    scan = causal_execution_scan(pd.read_parquet(scan_path))
    first_hot = attach_fixed_value(first_hot, scan)
    scored, threshold, _ = score_and_select_candidates(first_hot)
    selected = scored.loc[
        scored.trading_day.astype(str).isin(RULE_FIT_DAYS)
        & scored.candidate_probability.ge(threshold)
    ].copy()
    episodes = build_pullback_episodes(selected, scan)
    chosen, grid, baseline = evaluate_rule_grid(episodes)
    rows = []
    for stop in STOP_LOSSES:
        for trail in TRAILS:
            for hold in MAX_HOLD_MINUTES:
                frame = apply_exit_rule(
                    episodes, stop_loss_pct=stop, trail_pct=trail,
                    max_hold_minutes=hold,
                    policy_name=f'stop{stop}_trail{trail}_hold{hold}',
                )
                frame['stop_loss_pct'] = stop
                frame['trail_pct'] = trail
                frame['max_hold_minutes'] = hold
                rows.append(frame)
    all_rows = pd.concat(rows, ignore_index=True)
    result = dict(
        request_id=REQUEST_ID, parent_request=273,
        development_only=True, promotion_eligible=False, opens_new_dates=False,
        evaluated_days=list(RULE_FIT_DAYS),
        candidate_threshold=threshold,
        exit_contract='risk/stop at observed state before deadline; otherwise exact deadline or unresolved',
        fit_baseline=baseline, fit_grid=grid,
        fit_rule_selected=chosen is not None, chosen_rule=chosen,
        passing_rule_count=sum(row['passes'] for row in grid),
        economic_gate_pass=False,
        exit_reason_counts={str(k): int(v) for k, v in all_rows.exit_reason.value_counts().items()},
        input_sha256={str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [first_hot_path, scan_path]},
        interpretation='Conditional resolved means are not full-population or portfolio returns. '
                       'Passing fit rules, if any, require separate calibration and development evaluation.',
        next_boundary=('calibration check before development evaluation' if chosen
                       else 'joint candidate and pullback-state admission with fixed causal downstream exits'),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    all_rows.to_parquet(rows_output, index=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'fit_grid'}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-hot', type=Path, required=True)
    parser.add_argument('--opportunity-scan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows-output', type=Path, required=True)
    args = parser.parse_args()
    return evaluate(args.first_hot, args.opportunity_scan, args.output, args.rows_output)


if __name__ == '__main__':
    raise SystemExit(main())
