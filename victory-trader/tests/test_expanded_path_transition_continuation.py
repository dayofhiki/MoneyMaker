import numpy as np
import pandas as pd

from victory_trader.expanded_path_aware_continuation import PATH_FEATURES
from victory_trader.expanded_path_transition_continuation import (
    path_transition_features,
    transition_lag_coverage,
)


def _frame(times):
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-01-02"] * len(times),
            "ticker": ["TEST"] * len(times),
            "t": times,
        }
    )
    for i, column in enumerate(PATH_FEATURES):
        frame[column] = np.arange(len(times), dtype=float) * (i + 1)
    return frame


def test_path_transition_features_use_exact_minute_lags():
    frame = _frame([0, 60_000, 120_000, 180_000])
    transitions = path_transition_features(frame)

    assert np.isclose(
        transitions.loc[3, "path_transition_1m_path_entry_return_pct"],
        1.0,
    )
    assert np.isclose(
        transitions.loc[3, "path_transition_3m_path_entry_return_pct"],
        3.0,
    )
    assert np.isclose(
        transitions.loc[3, "path_transition_1m_path_mfe_pct"],
        2.0,
    )
    assert np.isnan(
        transitions.loc[2, "path_transition_3m_path_entry_return_pct"]
    )


def test_path_transition_features_do_not_compress_missing_minutes():
    frame = _frame([0, 60_000, 180_000])
    transitions = path_transition_features(frame)

    # The previous observed row is two minutes earlier, so it is not a 1m lag.
    assert np.isnan(
        transitions.loc[2, "path_transition_1m_path_entry_return_pct"]
    )
    # shift(2) points to t=0, which is exactly three minutes earlier, not 2m.
    assert np.isnan(
        transitions.loc[2, "path_transition_2m_path_entry_return_pct"]
    )
    assert np.isclose(
        transitions.loc[2, "path_transition_3m_path_entry_return_pct"],
        2.0,
    )


def test_path_transition_features_never_bleed_across_positions():
    frame = pd.concat(
        [
            _frame([0, 60_000]),
            _frame([120_000, 180_000]).assign(ticker="OTHER"),
        ],
        ignore_index=True,
    )
    transitions = path_transition_features(frame)

    assert np.isnan(
        transitions.loc[2, "path_transition_1m_path_entry_return_pct"]
    )
    assert np.isclose(
        transitions.loc[3, "path_transition_1m_path_entry_return_pct"],
        1.0,
    )


def test_transition_lag_coverage_reports_exact_timestamp_coverage():
    frame = _frame([0, 60_000, 180_000])
    coverage = transition_lag_coverage(frame, "2026-01").set_index(
        "lag_minutes"
    )

    assert coverage.loc[1, "exact_lag_rows"] == 1
    assert np.isclose(coverage.loc[1, "exact_lag_coverage"], 1 / 3)
    assert coverage.loc[2, "exact_lag_rows"] == 0
    assert coverage.loc[3, "exact_lag_rows"] == 1
