import pandas as pd

from victory_trader.causal_second_risk_compatible_entry import (
    _selection_rate,
    _summarize,
    first_qualifying,
    first_state_baseline,
)


def _rows():
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": 1,
                "state_t": 61,
                "minutes_held": 1.0,
                "entry_score": 0.2,
                "replay_status": "closed",
                "policy_net_return_pct": -1.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": 1,
                "state_t": 121,
                "minutes_held": 2.0,
                "entry_score": 0.8,
                "replay_status": "closed",
                "policy_net_return_pct": 4.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": 1,
                "state_t": 181,
                "minutes_held": 3.0,
                "entry_score": 0.9,
                "replay_status": "closed",
                "policy_net_return_pct": 8.0,
            },
            {
                "trading_day": "2026-06-24",
                "ticker": "BBB",
                "hot_t": 2,
                "state_t": 62,
                "minutes_held": 1.0,
                "entry_score": 0.85,
                "replay_status": "unresolved",
                "policy_net_return_pct": None,
            },
        ]
    )


def test_first_qualifying_enters_once_per_episode():
    frame = _rows()
    selected = first_qualifying(frame, 0.75)
    assert len(selected) == 2
    aaa = selected.loc[selected["ticker"].eq("AAA")].iloc[0]
    assert aaa["minutes_held"] == 2.0
    assert _selection_rate(selected, frame) == 1.0


def test_baseline_uses_first_observable_state():
    baseline = first_state_baseline(_rows())
    assert len(baseline) == 2
    aaa = baseline.loc[baseline["ticker"].eq("AAA")].iloc[0]
    assert aaa["minutes_held"] == 1.0


def test_unresolved_trade_counts_against_coverage():
    selected = first_qualifying(_rows(), 0.75)
    summary = _summarize(selected)
    assert summary["episodes"] == 2
    assert summary["closed"] == 1
    assert summary["closed_coverage"] == 0.5
    assert summary["mean_net_return_pct"] == 4.0
