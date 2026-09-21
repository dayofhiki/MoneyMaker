import pandas as pd

from victory_trader.expanded_execution_feasible_hurdle_ev import (
    MIN_ACTIVE_MINUTE_FRACTION_15M,
    MIN_DOLLAR_VOLUME_5M,
    MIN_TRANSACTIONS_5M,
    execution_feasible_anchors,
    execution_feasible_mask,
)


def _row(
    *,
    ticker="AAA",
    t=1,
    dollar_volume=MIN_DOLLAR_VOLUME_5M,
    transactions=MIN_TRANSACTIONS_5M,
    active=MIN_ACTIVE_MINUTE_FRACTION_15M,
):
    return {
        "trading_day": "2026-01-02",
        "ticker": ticker,
        "t": t,
        "c": 5.0,
        "previous_close": 4.0,
        "entry_price": 5.0,
        "return_from_previous_close_pct": 25.0,
        "minutes_since_10pct_cross": 0.0,
        "active_minute_fraction_15m": active,
        "dollar_volume_5m": dollar_volume,
        "transactions_5m": transactions,
    }


def test_execution_gate_accepts_exact_structural_boundaries():
    frame = pd.DataFrame([_row()])
    mask = execution_feasible_mask(frame)
    assert mask.tolist() == [True]


def test_execution_gate_rejects_each_failed_requirement():
    frame = pd.DataFrame(
        [
            _row(
                ticker="LOW_DV",
                dollar_volume=MIN_DOLLAR_VOLUME_5M - 1,
            ),
            _row(
                ticker="LOW_TX",
                transactions=MIN_TRANSACTIONS_5M - 1,
            ),
            _row(
                ticker="LOW_ACTIVE",
                active=MIN_ACTIVE_MINUTE_FRACTION_15M - 0.01,
            ),
        ]
    )
    assert execution_feasible_mask(frame).tolist() == [
        False,
        False,
        False,
    ]


def test_gate_is_applied_to_first_anchor_without_later_fallthrough():
    frame = pd.DataFrame(
        [
            _row(
                ticker="AAA",
                t=1,
                dollar_volume=MIN_DOLLAR_VOLUME_5M - 1,
            ),
            _row(
                ticker="AAA",
                t=2,
                dollar_volume=MIN_DOLLAR_VOLUME_5M * 10,
            ),
            _row(ticker="BBB", t=1),
        ]
    )

    anchors = execution_feasible_anchors(frame)

    assert anchors["ticker"].tolist() == ["BBB"]
    assert anchors["t"].tolist() == [1]
