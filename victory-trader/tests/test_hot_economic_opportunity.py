from __future__ import annotations

import pandas as pd
import pytest

from victory_trader.hot_economic_opportunity import (
    CAL_DAYS,
    ENTRY_FEATURES,
    FIT_DAYS,
    _attach_entry_context,
    fit_policy_opportunity_selector,
)
from victory_trader.market_calendar import regular_session_bounds


def test_entry_feature_allowlist_excludes_future_economic_labels():
    forbidden = {
        "oracle_best_base_pct",
        "oracle_best_minute",
        "oracle_base_positive",
        "return_1m_base_net_pct",
        "runner_cross_now",
        "target_next_cross",
    }
    assert forbidden.isdisjoint(ENTRY_FEATURES)


def test_attach_entry_context_adds_price_time_rank_and_transport_state():
    day = "2026-05-14"
    bounds = regular_session_bounds(pd.Timestamp(day).date())
    assert bounds is not None
    first_t = int(bounds[0].timestamp() * 1000) + 60_000

    candidates = pd.DataFrame(
        [
            {
                "trading_day": day,
                "ticker": "AAA",
                "t": first_t,
                "second_rerank_probability": 0.8,
                "transport_desired_now": True,
            },
            {
                "trading_day": day,
                "ticker": "BBB",
                "t": first_t,
                "second_rerank_probability": 0.4,
                "transport_desired_now": False,
            },
        ]
    )
    scan = pd.DataFrame(
        [
            {"trading_day": day, "ticker": "AAA", "t": first_t, "c": 2.0},
            {"trading_day": day, "ticker": "BBB", "t": first_t, "c": 4.0},
        ]
    )

    result = _attach_entry_context(candidates, scan)

    aaa = result.loc[result["ticker"].eq("AAA")].iloc[0]
    bbb = result.loc[result["ticker"].eq("BBB")].iloc[0]
    assert aaa["hot_candidate_rank"] == 1.0
    assert bbb["hot_candidate_rank"] == 2.0
    assert aaa["transport_desired_now_numeric"] == 1
    assert bbb["transport_desired_now_numeric"] == 0
    assert aaa["minutes_since_open"] == 1.0
    assert aaa["minutes_to_close"] > 300
    assert aaa["log_current_price"] < bbb["log_current_price"]


def test_policy_selector_calibrates_threshold_on_all_calibration_rows():
    rows = []
    for i in range(200):
        row = {
            "trading_day": FIT_DAYS[i % len(FIT_DAYS)],
            "oracle_best_base_pct": 1.0 if i >= 100 else -1.0,
        }
        for feature in ENTRY_FEATURES:
            row[feature] = 0.0
        row[ENTRY_FEATURES[0]] = i / 199.0
        rows.append(row)

    for i in range(100):
        row = {
            "trading_day": CAL_DAYS[i % len(CAL_DAYS)],
            # Deliberately hide future labels for the high-score half. A
            # causal policy threshold must still include these feature rows.
            "oracle_best_base_pct": (0.5 if i < 50 else float("nan")),
        }
        for feature in ENTRY_FEATURES:
            row[feature] = 0.0
        row[ENTRY_FEATURES[0]] = i / 99.0
        rows.append(row)

    frame = pd.DataFrame(rows)
    classifier, threshold = fit_policy_opportunity_selector(frame)
    calibration = frame.loc[frame["trading_day"].isin(CAL_DAYS)]
    probabilities = classifier.predict_proba(calibration[ENTRY_FEATURES])[:, 1]
    expected = float(pd.Series(probabilities).quantile(0.75))

    assert threshold == pytest.approx(expected)
