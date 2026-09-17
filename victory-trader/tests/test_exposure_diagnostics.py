import pandas as pd
import pytest

from victory_trader.exposure_diagnostics import summarize_unresolved_exposure


def test_unresolved_exposure_reports_break_even_and_worst_case_sensitivity():
    frame = pd.DataFrame(
        [
            {
                "threshold_pct": 20.0,
                "entry_price": 10.0,
                "return_5m_base_net_return_pct": 4.0,
            },
            {
                "threshold_pct": 20.0,
                "entry_price": 11.0,
                "return_5m_base_net_return_pct": 2.0,
            },
            {
                "threshold_pct": 20.0,
                "entry_price": 12.0,
                "return_5m_base_net_return_pct": None,
            },
            {
                "threshold_pct": 20.0,
                "entry_price": None,
                "return_5m_base_net_return_pct": None,
            },
        ]
    )

    result = summarize_unresolved_exposure(frame)
    row = result.iloc[0]

    assert row["event_rows"] == 4
    assert row["entered_n"] == 3
    assert row["resolved_n"] == 2
    assert row["unresolved_after_entry_n"] == 1
    assert row["unresolved_after_entry_rate"] == pytest.approx(1 / 3)
    assert row["observed_resolved_mean_pct"] == pytest.approx(3.0)
    assert row["zero_imputed_all_entered_mean_pct"] == pytest.approx(2.0)
    assert row["worst_case_all_entered_mean_pct"] == pytest.approx(-94.0 / 3.0)
    assert row["break_even_unresolved_mean_pct"] == pytest.approx(-6.0)


def test_no_unresolved_entries_has_nan_break_even():
    frame = pd.DataFrame(
        [
            {
                "threshold_pct": 10.0,
                "entry_price": 5.0,
                "return_5m_base_net_return_pct": 1.0,
            }
        ]
    )

    result = summarize_unresolved_exposure(frame)
    assert result.iloc[0]["unresolved_after_entry_n"] == 0
    assert pd.isna(result.iloc[0]["break_even_unresolved_mean_pct"])
