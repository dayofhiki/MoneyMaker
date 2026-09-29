import pandas as pd

from victory_trader.second_path_attention_probe import (
    second_path_features,
)


def test_second_features_ignore_decision_and_future_seconds():
    decision_t = 100_000
    base = pd.DataFrame([
        {
            "t": decision_t - 2_000,
            "o": 10.0,
            "h": 10.2,
            "l": 9.9,
            "c": 10.1,
            "v": 100.0,
            "n": 10.0,
        },
        {
            "t": decision_t - 1_000,
            "o": 10.1,
            "h": 10.3,
            "l": 10.0,
            "c": 10.2,
            "v": 110.0,
            "n": 11.0,
        },
    ])
    with_future = pd.concat([
        base,
        pd.DataFrame([
            {
                "t": decision_t,
                "o": 999.0,
                "h": 999.0,
                "l": 999.0,
                "c": 999.0,
                "v": 999999.0,
                "n": 9999.0,
            },
            {
                "t": decision_t + 1_000,
                "o": 888.0,
                "h": 888.0,
                "l": 888.0,
                "c": 888.0,
                "v": 888888.0,
                "n": 8888.0,
            },
        ]),
    ], ignore_index=True)

    left = pd.Series(second_path_features(base, decision_t))
    right = pd.Series(second_path_features(with_future, decision_t))
    pd.testing.assert_series_equal(left, right)
