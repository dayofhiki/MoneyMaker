"""Request242: entry-opportunity ceiling decomposition.

Request241 showed that continuation availability transfers well, while learned
entry timing still loses money. This diagnostic keeps the Request241
availability stage frozen and asks whether the admitted episodes actually
contain economically usable entry moments.

For each availability-admitted development episode we compare:
* FIRST: enter at the first observed WATCH state;
* MULTI-ORACLE: choose the WATCH state with the highest positive multi-event
  value, then measure its event-3 BASE return;
* EVENT3-ORACLE: choose the WATCH state with the highest positive event-3 BASE
  return.

The oracles are hindsight diagnostics only. They are never deployable policies.
No new dates are opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .availability_conditioned_entry import (
    _availability_oof,
    _score_episode_probability,
    build_event_watch_states,
)
from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS

REQUEST_ID = 242


def _policy_metrics(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {
            "episodes": 0,
            "mean_pct": None,
            "day_balanced_mean_pct": None,
            "positive_rate": None,
            "positive_days": 0,
            "severe_loss_rate_lt_minus2": None,
            "severe_loss_rate_lt_minus5": None,
        }
    values = pd.to_numeric(
        rows["realized_base_return_pct"],
        errors="coerce",
    ).fillna(0.0)
    work = rows.copy()
    work["_value"] = values
    daily = work.groupby(
        work["trading_day"].astype(str),
        sort=True,
    )["_value"].mean()
    return {
        "episodes": int(len(work)),
        "mean_pct": float(values.mean()),
        "day_balanced_mean_pct": float(daily.mean()),
        "positive_rate": float(values.gt(0).mean()),
        "positive_days": int((daily > 0).sum()),
        "severe_loss_rate_lt_minus2": float(
            values.lt(-2).mean()
        ),
        "severe_loss_rate_lt_minus5": float(
            values.lt(-5).mean()
        ),
        "by_day": {
            str(day): float(value)
            for day, value in daily.items()
        },
    }


def _safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> float | None:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    valid = a.notna() & b.notna()
    if int(valid.sum()) < 10:
        return None
    value = a.loc[valid].corr(
        b.loc[valid],
        method="spearman",
    )
    return None if pd.isna(value) else float(value)


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    fresh_paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    if len(fresh_paths) != len(FRESH_DAYS):
        raise ValueError(
            "request242 requires complete Request178 shards"
        )
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in fresh_paths],
        ignore_index=True,
    )

    fit_states = build_event_watch_states(fit_positions)
    cal_states = build_event_watch_states(
        calibration_positions
    )
    fresh_states = build_event_watch_states(
        fresh_positions
    )

    _, availability_model = _availability_oof(
        fit_states
    )
    cal_states, cal_first = _score_episode_probability(
        cal_states,
        availability_model,
    )
    fresh_states, fresh_first = _score_episode_probability(
        fresh_states,
        availability_model,
    )

    availability_gate = float(
        np.quantile(
            pd.to_numeric(
                cal_first[
                    "episode_availability_probability"
                ],
                errors="coerce",
            ).dropna(),
            0.75,
        )
    )

    admitted_keys = fresh_first.loc[
        pd.to_numeric(
            fresh_first[
                "episode_availability_probability"
            ],
            errors="coerce",
        ).ge(availability_gate),
        EPISODE_KEYS,
    ].copy()
    admitted = fresh_states.merge(
        admitted_keys,
        on=EPISODE_KEYS,
        how="inner",
        validate="many_to_one",
    )

    first_records: list[dict[str, object]] = []
    multi_records: list[dict[str, object]] = []
    event3_records: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []

    for keys, group in admitted.groupby(
        EPISODE_KEYS,
        sort=False,
    ):
        ordered = group.sort_values(
            "watch_event_index",
            kind="stable",
        )
        first = ordered.iloc[0]
        first_value = pd.to_numeric(
            pd.Series(
                [
                    first.get(
                        "entry_event3_realized_base_pct"
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        multi = pd.to_numeric(
            ordered["entry_multi_event_value_pct"],
            errors="coerce",
        )
        event3 = pd.to_numeric(
            ordered[
                "entry_event3_realized_base_pct"
            ],
            errors="coerce",
        )

        valid_multi = ordered.loc[
            multi.notna()
        ].copy()
        valid_multi["_multi"] = multi.loc[
            multi.notna()
        ]
        if len(valid_multi):
            best_multi_row = valid_multi.sort_values(
                ["_multi", "watch_event_index"],
                ascending=[False, True],
                kind="stable",
            ).iloc[0]
            best_multi = float(
                best_multi_row["_multi"]
            )
            multi_realized = pd.to_numeric(
                pd.Series(
                    [
                        best_multi_row.get(
                            "entry_event3_realized_base_pct"
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]
        else:
            best_multi = np.nan
            multi_realized = np.nan

        valid_event3 = ordered.loc[
            event3.notna()
        ].copy()
        valid_event3["_event3"] = event3.loc[
            event3.notna()
        ]
        if len(valid_event3):
            best_event3_row = valid_event3.sort_values(
                ["_event3", "watch_event_index"],
                ascending=[False, True],
                kind="stable",
            ).iloc[0]
            best_event3 = float(
                best_event3_row["_event3"]
            )
        else:
            best_event3 = np.nan

        first_records.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "realized_base_return_pct": (
                    float(first_value)
                    if pd.notna(first_value)
                    else 0.0
                ),
            }
        )
        multi_records.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "realized_base_return_pct": (
                    float(multi_realized)
                    if best_multi > 0
                    and pd.notna(multi_realized)
                    else 0.0
                ),
            }
        )
        event3_records.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "realized_base_return_pct": (
                    max(0.0, best_event3)
                    if np.isfinite(best_event3)
                    else 0.0
                ),
            }
        )
        episode_rows.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "first_event3_pct": (
                    float(first_value)
                    if pd.notna(first_value)
                    else np.nan
                ),
                "best_multi_event_value_pct": (
                    best_multi
                    if np.isfinite(best_multi)
                    else np.nan
                ),
                "event3_at_best_multi_pct": (
                    float(multi_realized)
                    if pd.notna(multi_realized)
                    else np.nan
                ),
                "best_event3_pct": (
                    best_event3
                    if np.isfinite(best_event3)
                    else np.nan
                ),
                "has_positive_event3": bool(
                    np.isfinite(best_event3)
                    and best_event3 > 0
                ),
            }
        )

    first_df = pd.DataFrame(first_records)
    multi_df = pd.DataFrame(multi_records)
    event3_df = pd.DataFrame(event3_records)
    episodes = pd.DataFrame(episode_rows)

    state_multi = pd.to_numeric(
        admitted["entry_multi_event_value_pct"],
        errors="coerce",
    )
    state_event3 = pd.to_numeric(
        admitted["entry_event3_realized_base_pct"],
        errors="coerce",
    )
    state_corr = _safe_spearman(
        state_multi,
        state_event3,
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "development_only": True,
        "promotion_eligible": False,
        "availability_gate": (
            availability_gate
        ),
        "admitted_episodes": int(
            len(admitted_keys)
        ),
        "state_multi_value_vs_event3_spearman": (
            state_corr
        ),
        "episode_diagnostics": {
            "any_positive_event3_rate": (
                float(
                    episodes[
                        "has_positive_event3"
                    ].mean()
                )
                if len(episodes)
                else None
            ),
            "best_multi_positive_rate": (
                float(
                    pd.to_numeric(
                        episodes[
                            "best_multi_event_value_pct"
                        ],
                        errors="coerce",
                    ).gt(0).mean()
                )
                if len(episodes)
                else None
            ),
            "event3_at_best_multi_mean_pct": (
                float(
                    pd.to_numeric(
                        episodes[
                            "event3_at_best_multi_pct"
                        ],
                        errors="coerce",
                    ).mean()
                )
                if len(episodes)
                else None
            ),
            "event3_at_best_multi_positive_rate": (
                float(
                    pd.to_numeric(
                        episodes[
                            "event3_at_best_multi_pct"
                        ],
                        errors="coerce",
                    ).gt(0).mean()
                )
                if len(episodes)
                else None
            ),
        },
        "policies": {
            "first_event": _policy_metrics(
                first_df
            ),
            "oracle_best_multi_event": (
                _policy_metrics(multi_df)
            ),
            "oracle_best_event3": (
                _policy_metrics(event3_df)
            ),
        },
        "interpretation": {
            "target_misalignment": (
                "oracle_best_event3 is positive while "
                "oracle_best_multi_event remains non-positive "
                "or materially weaker"
            ),
            "timing_learning_problem": (
                "oracle_best_multi_event and oracle_best_event3 "
                "are both positive, but Request241 learned policy "
                "is negative"
            ),
            "upstream_economic_problem": (
                "even oracle_best_event3 is not economically "
                "positive"
            ),
        },
        "next_if_target_misaligned": (
            "train entry utility directly around downside-controlled "
            "event-3 economics rather than average multi-event value"
        ),
        "next_if_timing_learning_problem": (
            "keep the target, improve event-path representation / "
            "sequence timing model"
        ),
        "next_if_upstream_problem": (
            "availability alone is insufficient; add an independent "
            "economic-opportunity head before timing"
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-positions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-positions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fresh-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.fresh_dir,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
