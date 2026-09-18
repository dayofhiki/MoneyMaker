from datetime import date

import numpy as np
import pandas as pd

from victory_trader.state_eight_k_enrichment import (
    EIGHT_K_FEATURES,
    _feature_row,
    fetch_eight_k_history,
)
from victory_trader.state_eight_k_value import (
    eight_k_action_feature_frame,
    select_first_trades,
    success_check,
)


def _row(day, accession, primary):
    return {
        "_filing_day": pd.Timestamp(day).date(),
        "filing_date": day,
        "accession_number": accession,
        "primary_category": primary,
    }


def test_eight_k_features_use_only_strictly_prior_30_days():
    records = [
        _row("2025-12-01", "old", "financial_results"),
        _row("2025-12-20", "a", "capital_and_financing"),
        _row("2026-01-05", "b", "leadership_and_governance"),
        _row("2026-01-09", "c", "capital_and_financing"),
        _row("2026-01-10", "same-day", "risk_events"),
    ]
    features = _feature_row(records, date(2026, 1, 10))

    assert features["eight_k_unique_filings_30d"] == 3.0
    assert features["eight_k_disclosures_30d"] == 3.0
    assert features["eight_k_has_filing_7d"] == 1.0
    assert features["eight_k_latest_filing_age_days"] == 1.0
    assert (
        features["eight_k_primary_capital_and_financing_count_30d"]
        == 2.0
    )
    assert features["eight_k_primary_risk_events_count_30d"] == 0.0


def test_no_filing_is_valid_zero_state():
    features = _feature_row([], date(2026, 1, 10))
    assert features["eight_k_unique_filings_30d"] == 0.0
    assert features["eight_k_disclosures_30d"] == 0.0
    assert features["eight_k_has_filing_7d"] == 0.0
    assert np.isnan(features["eight_k_latest_filing_age_days"])


def test_eight_k_feature_frame_contains_fixed_features():
    frame = pd.DataFrame(
        {
            "return_from_previous_close_pct": [10.0],
            "minutes_since_10pct_cross": [1.0],
            "minutes_from_regular_open": [10.0],
            "active_minute_fraction_15m": [1.0],
            "c": [5.0],
            "previous_close": [4.0],
            **{feature: [1.0] for feature in EIGHT_K_FEATURES},
        }
    )
    features = eight_k_action_feature_frame(frame)
    for column in EIGHT_K_FEATURES:
        assert column in features.columns
        assert features.iloc[0][column] == 1.0


def _trade_row(minute, *, ev, horizon=15, entry_price=5.0):
    row = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": minute * 60_000,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": entry_price,
        "predicted_best_base_ev_pct": 0.0,
        "predicted_best_horizon_min": horizon,
        "predicted_best_eight_k_ev_pct": ev,
        "predicted_best_eight_k_horizon_min": horizon,
    }
    for current in (5, 10, 15):
        row[f"buy_return_{current}m_base_net_return_pct"] = current / 10
        row[f"buy_return_{current}m_pct"] = current / 10 + 1.0
        row[f"buy_return_{current}m_stress_net_return_pct"] = current / 10 - 1.0
    return row


def test_eight_k_policy_enters_first_half_percent_signal():
    frame = pd.DataFrame(
        [
            _trade_row(0, ev=0.49),
            _trade_row(1, ev=0.50, horizon=10),
            _trade_row(2, ev=2.0),
        ]
    )
    trades = select_first_trades(frame, policy="eight_k_ev_cap1")
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000
    assert int(trades.iloc[0]["action_horizon_min"]) == 10


def test_eight_k_first_signal_consumed_when_unfillable():
    frame = pd.DataFrame(
        [
            _trade_row(0, ev=0.75, entry_price=np.nan),
            _trade_row(1, ev=1.50),
        ]
    )
    trades = select_first_trades(frame, policy="eight_k_ev_cap1")
    assert trades.empty


def test_eight_k_success_check_requires_all_rules():
    rows = []
    for month in ("2026-01", "2026-02", "2026-03"):
        rows.extend(
            [
                {
                    "month": month,
                    "policy": "earliest_ev_cap1",
                    "trades": 20,
                    "base_mean_pct": 0.5,
                    "day_balanced_base_mean_pct": 0.5,
                    "base_p05_pct": -10.0,
                    "stress_mean_pct": -1.0,
                },
                {
                    "month": month,
                    "policy": "eight_k_ev_cap1",
                    "trades": 20,
                    "base_mean_pct": 1.0,
                    "day_balanced_base_mean_pct": 1.0,
                    "base_p05_pct": -9.0,
                    "stress_mean_pct": 0.0,
                },
            ]
        )
    details = pd.DataFrame(rows)
    coverage = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02", "2026-03"],
            "query_complete_coverage": [1.0, 1.0, 1.0],
        }
    )
    assert success_check(
        details,
        coverage,
        {"ci_low_pct": 0.1},
    )["all_pass"] is True


def test_eight_k_history_uses_one_market_wide_query_and_maps_tickers():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def eight_k_disclosures_market(
            self,
            *,
            filing_date_gte,
            filing_date_lte,
        ):
            self.calls.append((filing_date_gte, filing_date_lte))
            return [
                {
                    "filing_date": "2026-01-05",
                    "tickers": ["AAA", "AAA.A"],
                    "accession_number": "a",
                    "primary_category": "financial_results",
                },
                {
                    "filing_date": "2026-01-06",
                    "tickers": ["BBB"],
                    "accession_number": "b",
                    "primary_category": "capital_and_financing",
                },
                {
                    "filing_date": "2026-01-07",
                    "tickers": ["NOT_IN_PANEL"],
                    "accession_number": "c",
                    "primary_category": "risk_events",
                },
            ]

    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "trading_day": ["2026-01-10"] * 3,
        }
    )
    client = FakeClient()
    history, queried = fetch_eight_k_history({"2026-01": frame}, client)

    assert len(client.calls) == 1
    assert set(history) == {"AAA", "BBB"}
    assert queried == {"AAA", "BBB", "CCC"}
    assert history["AAA"][0]["_filing_day"] == date(2026, 1, 5)
    assert history["BBB"][0]["_filing_day"] == date(2026, 1, 6)
