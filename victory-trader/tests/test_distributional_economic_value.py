import pandas as pd

from victory_trader import distributional_economic_value as r252


def test_request252_contract():
    assert r252.REQUEST_ID == 252
    assert r252.FRACTIONS == (0.02, 0.05, 0.10, 0.15, 0.20)
    assert r252.MIN_WIN_AUC == 0.60
    assert r252.MIN_EV_SPEARMAN == 0.10
    assert r252.MAX_SELECTED_SEVERE_RATE == 0.20


def test_choose_rule_rejects_weak_model_metrics():
    days = [
        "2026-05-11",
        "2026-05-12",
        "2026-05-13",
        "2026-05-14",
        "2026-05-15",
        "2026-05-18",
        "2026-05-19",
        "2026-05-20",
    ]
    rows = []
    for day in days:
        for i in range(100):
            rows.append(
                {
                    "trading_day": day,
                    "fixed_first_watch_value_pct": 1.0 if i >= 80 else -1.0,
                    "distributional_expected_value": float(i),
                }
            )
    frame = pd.DataFrame(rows)
    chosen, results = r252.choose_rule(
        frame,
        win_auc=0.50,
        ev_spearman=0.0,
    )
    assert chosen is None
    assert len(results) == len(r252.FRACTIONS)
