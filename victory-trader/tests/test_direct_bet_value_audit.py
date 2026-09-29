import pandas as pd

from victory_trader.direct_bet_value_audit import audit


def test_direct_value_audit_reports_ordering_and_positive_rule():
    rows = []
    for index in range(40):
        score = -2.0 + index * 0.1
        realized = score + 0.25
        rows.append({
            "trading_day": f"2026-05-0{5 + (index % 4)}",
            "ticker": f"T{index}",
            "hot_t": index,
            "selected_prehot": True,
            "resolved": True,
            "trade_return_pct": realized,
            "direct_predicted_return_pct": score,
        })
    report = audit(pd.DataFrame(rows))
    assert report["resolved_shortlist_rows"] == 40
    assert report["direct_value_spearman"] > 0.9
    assert report["predicted_positive_resolved_bets"] > 0
    assert (
        report["quartiles_low_to_high"]["q4"]["realized_mean_pct"]
        > report["quartiles_low_to_high"]["q1"]["realized_mean_pct"]
    )
