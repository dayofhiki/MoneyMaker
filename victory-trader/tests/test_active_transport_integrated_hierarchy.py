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


def test_active_age_survives_temporary_unscoreable_gap():
    rows = []
    day = "2026-04-30"
    for timestamp in [60_000, 120_000, 180_000]:
        for index in range(20):
            ticker = f"I{index:02d}"
            if ticker == "I00" and timestamp == 120_000:
                continue
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

    active, trace = active_observation_rows_with_state(pd.DataFrame(rows))
    gap = trace.loc[
        trace["t"].eq(120_000) & trace["ticker"].eq("I00")
    ].iloc[0]
    resumed = active.loc[
        active["t"].eq(180_000) & active["ticker"].eq("I00")
    ].iloc[0]

    assert bool(gap["transport_active"]) is True
    assert bool(gap["scoreable_now"]) is False
    assert resumed["focus_age_minutes"] == 3
    assert resumed["watch_run_age_minutes"] == 3
    assert bool(resumed["transport_addition"]) is False


def test_active_transport_can_rank_by_causal_utility_score():
    rows = _scored_rows().copy()
    rows["active_priority"] = rows["market_hazard_probability"]
    first_t = int(rows["t"].min())
    first = rows["t"].eq(first_t)
    rows.loc[first & rows["ticker"].eq("I19"), "active_priority"] = 10_000.0
    rows.loc[first & rows["ticker"].eq("I00"), "active_priority"] = -10_000.0

    active, _ = active_observation_rows_with_state(
        rows,
        score_column="active_priority",
    )
    selected = set(
        active.loc[active["t"].eq(first_t), "ticker"].astype(str)
    )

    assert "I19" in selected
    assert "I00" not in selected


def test_active_desired_set_respects_focus_eligibility_but_retains_existing_subscription():
    rows = _scored_rows().copy()
    rows["active_priority"] = rows["market_hazard_probability"]
    rows["focus_eligible"] = True

    first_t = int(rows["t"].min())
    second_t = int(rows["t"].max())

    # An outside-Focus name has the highest utility but must not enter desired.
    rows.loc[
        rows["t"].eq(first_t) & rows["ticker"].eq("I19"),
        "focus_eligible",
    ] = False
    rows.loc[
        rows["t"].eq(first_t) & rows["ticker"].eq("I19"),
        "active_priority",
    ] = 1_000_000.0

    active, trace = active_observation_rows_with_state(
        rows,
        score_column="active_priority",
        desired_eligible_column="focus_eligible",
    )
    first_active = set(
        active.loc[active["t"].eq(first_t), "ticker"].astype(str)
    )
    assert "I19" not in first_active

    # I00 starts subscribed, then leaves Focus. It can remain active while the
    # five-addition transport budget moves membership gradually.
    rows2 = _scored_rows().copy()
    rows2["active_priority"] = rows2["market_hazard_probability"]
    rows2["focus_eligible"] = True
    rows2.loc[
        rows2["t"].eq(second_t) & rows2["ticker"].eq("I00"),
        "focus_eligible",
    ] = False

    active2, trace2 = active_observation_rows_with_state(
        rows2,
        score_column="active_priority",
        desired_eligible_column="focus_eligible",
    )
    second_trace = trace2.loc[
        trace2["t"].eq(second_t) & trace2["ticker"].eq("I00")
    ]
    if not second_trace.empty:
        assert bool(second_trace.iloc[0]["scoreable_now"]) is True
        assert "I00" in set(
            active2.loc[active2["t"].eq(second_t), "ticker"].astype(str)
        )
