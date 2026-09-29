import pandas as pd

from victory_trader import downside_survivability_admission as r251


def test_request251_contract():
    assert r251.REQUEST_ID == 251
    assert r251.SEVERE_LOSS_THRESHOLD == -2.0
    assert r251.MIN_RISK_AUC == 0.58
    assert r251.MAX_SELECTED_SEVERE_RATE == 0.15
    assert r251.FRACTIONS == (0.02, 0.05, 0.10, 0.15, 0.20)


def test_choose_rule_refuses_weak_risk_head():
    rows = []
    for day in [
        "2026-05-11",
        "2026-05-12",
        "2026-05-13",
        "2026-05-14",
        "2026-05-15",
        "2026-05-18",
        "2026-05-19",
        "2026-05-20",
    ]:
        for i in range(50):
            rows.append(
                {
                    "trading_day": day,
                    "fixed_first_watch_value_pct": 1.0 if i >= 40 else -1.0,
                    "risk_adjusted_opportunity": i / 50,
                    "joint_percentile_baseline": i / 50,
                }
            )
    frame = pd.DataFrame(rows)
    chosen, candidate, baseline = r251.choose_rule(frame, risk_auc=0.50)
    assert chosen is None
    assert len(candidate) == len(r251.FRACTIONS)
    assert len(baseline) == len(r251.FRACTIONS)
