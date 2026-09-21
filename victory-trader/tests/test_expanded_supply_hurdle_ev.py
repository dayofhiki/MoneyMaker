import numpy as np
import pandas as pd

from victory_trader.expanded_supply_hurdle_ev import (
    select_policies,
    supply_feature_frame,
)


def _row():
    return {
        "trading_day": "2026-01-05",
        "ticker": "AAA",
        "t": 1,
        "c": 6.0,
        "previous_close": 5.0,
        "entry_price": 6.0,
        "volume_5m": 500_000.0,
        "dollar_volume_5m": 3_000_000.0,
        "active_minute_fraction_15m": 1.0,
        "transactions_5m": 100.0,
        "supply_weighted_shares_outstanding_prior": 10_000_000.0,
        "supply_share_class_shares_outstanding_prior": 8_000_000.0,
        "buy_return_15m_pct": 2.0,
        "buy_return_15m_base_net_return_pct": 1.0,
        "buy_return_15m_stress_net_return_pct": 0.5,
    }


def test_supply_features_use_prior_share_supply_and_current_causal_flow():
    frame = pd.DataFrame([_row()])

    features = supply_feature_frame(frame)
    row = features.iloc[0]

    assert np.isclose(
        row["supply_log_weighted_shares_prior"],
        np.log(10_000_000.0),
    )
    assert np.isclose(
        row["supply_log_share_class_shares_prior"],
        np.log(8_000_000.0),
    )
    assert np.isclose(
        row["supply_log_implied_market_cap_prior"],
        np.log(50_000_000.0),
    )
    assert np.isclose(
        row["supply_weighted_share_turnover_5m"],
        0.05,
    )
    assert np.isclose(
        row["supply_share_class_turnover_5m"],
        0.0625,
    )
    assert np.isclose(
        row["supply_market_cap_turnover_5m"],
        3_000_000.0 / 50_000_000.0,
    )
    assert np.isclose(
        row["supply_share_class_to_weighted_ratio"],
        0.8,
    )


def test_supply_policy_can_differ_from_baseline_without_threshold_tuning():
    scored = pd.DataFrame(
        [
            {
                **_row(),
                "base_hurdle_ev_pct": -0.1,
                "supply_hurdle_ev_pct": 0.2,
            }
        ]
    )

    trades, attempts = select_policies(scored)

    assert set(attempts["policy"]) == {
        "feasible_earliest_15m_cap1",
        "supply_hurdle_ev_15m_cap1",
    }
    assert set(trades["policy"]) == {
        "feasible_earliest_15m_cap1",
        "supply_hurdle_ev_15m_cap1",
    }
    primary = trades.loc[
        trades["policy"].eq("supply_hurdle_ev_15m_cap1")
    ].iloc[0]
    assert primary["selected_decision_score"] == 0.2
