import pandas as pd

from victory_trader.state_sequence_enrichment import (
    SEQUENCE_FEATURES,
    enrich_state_sequence,
)


def test_sequence_lags_are_causal_and_ticker_local():
    rows = []
    for ticker, base in (("AAA", 0.0), ("BBB", 100.0)):
        for minute in range(15):
            rows.append(
                {
                    "trading_day": "2026-01-02",
                    "ticker": ticker,
                    "t": minute,
                    "trailing_return_1m_pct": base + minute,
                    "hod_distance_pct": -(base + minute),
                    "regular_vwap_distance_pct": base + minute / 10,
                    "volume_accel_1m_vs_prior20m": 1 + minute,
                    "bar_range_pct": 0.5 + minute / 100,
                    "bar_close_location": 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    out = enrich_state_sequence(frame)

    aaa10 = out.loc[
        out["ticker"].eq("AAA") & out["t"].eq(10)
    ].iloc[0]
    assert aaa10["seq_ret1_lag1"] == 9.0
    assert aaa10["seq_ret1_lag5"] == 5.0
    assert aaa10["seq_ret1_lag8"] == 2.0
    assert pd.isna(aaa10["seq_ret1_lag13"])

    bbb1 = out.loc[
        out["ticker"].eq("BBB") & out["t"].eq(1)
    ].iloc[0]
    assert bbb1["seq_ret1_lag1"] == 100.0
    assert pd.isna(bbb1["seq_ret1_lag2"])
    assert len(SEQUENCE_FEATURES) == 36


def test_future_changes_do_not_change_past_sequence_features():
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-01-02"] * 5,
            "ticker": ["AAA"] * 5,
            "t": list(range(5)),
            "trailing_return_1m_pct": [0, 1, 2, 3, 4],
            "hod_distance_pct": [0, -1, -2, -3, -4],
            "regular_vwap_distance_pct": [0, 1, 2, 3, 4],
            "volume_accel_1m_vs_prior20m": [1, 2, 3, 4, 5],
            "bar_range_pct": [1, 1, 1, 1, 1],
            "bar_close_location": [0.5] * 5,
        }
    )
    first = enrich_state_sequence(frame)
    changed = frame.copy()
    changed.loc[4, "trailing_return_1m_pct"] = 999
    second = enrich_state_sequence(changed)

    cols = ["seq_ret1_lag1", "seq_ret1_lag2", "seq_hod_lag1"]
    pd.testing.assert_series_equal(
        first.loc[3, cols],
        second.loc[3, cols],
        check_names=False,
    )
