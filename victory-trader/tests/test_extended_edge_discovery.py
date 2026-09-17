import pandas as pd

from victory_trader.extended_edge_discovery import render_extended_edge_discovery


def test_extended_discovery_separates_gross_and_net_targets():
    rows = []
    for day in range(1, 9):
        for i in range(20):
            feature = float(i + 1)
            rows.append(
                {
                    "trading_day": f"2026-03-{day:02d}",
                    "threshold_pct": 10.0,
                    "previous_close": feature,
                    "return_5m_pct": feature * 0.1,
                    "return_5m_base_net_return_pct": feature * 0.1 - 1.0,
                }
            )
    report = render_extended_edge_discovery(pd.DataFrame(rows), horizon_min=5)
    assert "GROSS continuation" in report
    assert "BASE-NET continuation" in report
    assert "previous_close" in report
    assert "execution-cost mechanics" in report
