from __future__ import annotations

import pandas as pd

from victory_trader.active_transport_integrated_hierarchy import (
    _observable_prior_metrics,
    active_observation_rows,
    active_observation_rows_with_state,
    explicit_subscription_audit,
)


def _scored_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    day = "2026-04-30"
    for timestamp, challenger_first in [(60_000, False), (120_000, True)]:
        for index in range(40):
            if challenger_first:
                ticker = f"C{index:02d}" if index < 20 else f"I{index - 20:02d}"
            else:
                ticker = f"I{index:02d}" if index < 20 else f"C{index - 20:02d}"
            rows.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "t": timestamp,
                    "market_hazard_probability": float(100 - index),
                    "attention_score": float(100 - index),
                    "attention_rank": index + 1,
                    "return_from_previous_close_pct": 1.0,
                    "minute_body_return_pct": 0.1,
                    "minute_range_pct": 0.2,
                    "log_minute_volume": 1.0,
                    "log_minute_transactions": 1.0,
                    "minute_return_1m_pct": 0.1,
                    "return_accel_1m_pct": 0.0,
                    "volume_ratio_prev1": 1.0,
                    "transactions_ratio_prev1": 1.0,
                    "target_next_cross": 0,
                }
            )
    return pd.DataFrame(rows)


def test_active_observation_rows_rate_limits_and_keeps_current_features():
    active = active_observation_rows(_scored_rows())

    first = active.loc[active["t"].eq(60_000)]
    second = active.loc[active["t"].eq(120_000)]
    first_names = set(first["ticker"].astype(str))
    second_names = set(second["ticker"].astype(str))

    assert len(first_names) == 20
    assert len(second_names) == 20
    assert len(second_names - first_names) == 5
    assert int(second["transport_addition"].sum()) == 5
    assert {f"C{i:02d}" for i in range(5)}.issubset(second_names)

    retained = second.loc[second["ticker"].eq("I00")].iloc[0]
    challenger = second.loc[second["ticker"].eq("C00")].iloc[0]
    assert retained["focus_age_minutes"] == 2
    assert challenger["focus_age_minutes"] == 1
    assert bool(challenger["transport_desired_now"]) is True
    assert retained["stage1_hazard_probability"] == retained[
        "market_hazard_probability"
    ]


def test_observable_prior_metrics_uses_only_causal_exact_prior_rows():
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-04-30",
                "ticker": "A",
                "t": 120_000,
                "runner_cross_now": True,
            },
            {
                "trading_day": "2026-04-30",
                "ticker": "B",
                "t": 120_000,
                "runner_cross_now": True,
            },
        ]
    )
    scored = pd.DataFrame(
        [
            {"trading_day": "2026-04-30", "ticker": "A", "t": 60_000},
        ]
    )
    focus = scored.copy()

    result = _observable_prior_metrics(scored, focus, scan)

    assert result["runner_crossings"] == 2
    assert result["observable_exact_prior_crossings"] == 1
    assert result["focus_captured_observable"] == 1
    assert result["focus_capture_given_observable_rate"] == 1.0



def test_explicit_subscription_state_survives_excess_missing_rows():
    rows = _scored_rows()
    # More than five incumbent rows disappear at once. The selector can replace
    # only five subscriptions, so some missing-row incumbents must remain truly
    # subscribed even though they are not scoreable at this timestamp.
    missing = {f"I{i:02d}" for i in range(12, 20)}
    rows = rows.loc[
        ~(
            rows["ticker"].isin(missing)
            & rows["t"].eq(120_000)
        )
    ].copy()

    _, trace = active_observation_rows_with_state(rows)
    second = trace.loc[trace["t"].eq(120_000)]
    unscoreable = second.loc[
        ~second["scoreable_now"].fillna(False).astype(bool)
    ]

    assert len(unscoreable) == 3
    assert unscoreable["transport_active"].fillna(False).astype(bool).all()


def test_explicit_subscription_audit_counts_selector_state_not_row_reappearance():
    trace = pd.DataFrame(
        [
            {
                "trading_day": "2026-04-30",
                "t": 60_000,
                "ticker": ticker,
                "scoreable_now": True,
            }
            for ticker in ["A", "B", "C"]
        ]
        + [
            {
                "trading_day": "2026-04-30",
                "t": 120_000,
                "ticker": ticker,
                "scoreable_now": ticker != "B",
            }
            for ticker in ["A", "B", "C"]
        ]
        + [
            {
                "trading_day": "2026-04-30",
                "t": 180_000,
                "ticker": ticker,
                "scoreable_now": True,
            }
            for ticker in ["A", "B", "C"]
        ]
    )

    audit = explicit_subscription_audit(trace)

    assert audit["max_post_initial_subscription_additions_per_decision"] == 0
    assert audit["temporarily_unscoreable_subscription_rows"] == 1
