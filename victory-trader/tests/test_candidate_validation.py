import json

import pandas as pd

from victory_trader.candidate_score import fit_candidate_rule
from victory_trader.candidate_validation import (
    evaluate_external_month,
    load_candidate_rule,
    render_external_validation,
)


def _frame(month: int, start_day: int = 1, days: int = 12) -> pd.DataFrame:
    rows = []
    horizons = (1, 2, 5, 10, 15, 30, 60)
    for day in range(start_day, start_day + days):
        for i in range(70):
            low_is_good = float(i)
            row = {
                "trading_day": f"2026-{month:02d}-{day:02d}",
                "threshold_pct": 10.0 if i < 50 else 20.0,
                "ticker": f"T{i:03d}",
                "entry_price": 5.0 + i * 0.01,
                "prior_15m_low_rebound_pct": low_is_good,
                "volatility_15m_pct": low_is_good + 0.1,
                "signal_bar_range_pct": low_is_good + 0.2,
                "trailing_return_5m_pct": low_is_good + 0.3,
                "dollar_volume_5m": 2_000_000.0 + (i % 7) * 10_000,
                "transactions_5m": 2_000.0 + (i % 5) * 10,
                "active_minute_fraction_15m": 0.9 + (i % 3) * 0.02,
            }
            for horizon in horizons:
                gross = 2.0 - low_is_good * 0.02 + horizon * 0.001
                row[f"return_{horizon}m_pct"] = gross
                row[f"return_{horizon}m_base_net_return_pct"] = gross - 0.5
                row[f"return_{horizon}m_stress_net_return_pct"] = gross - 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_load_candidate_rule_round_trip(tmp_path):
    march = _frame(3, days=16)
    rule = fit_candidate_rule(march)
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(rule.to_dict()), encoding="utf-8")
    loaded = load_candidate_rule(path)
    assert loaded.train_dates == rule.train_dates
    assert loaded.holdout_dates == rule.holdout_dates
    assert loaded.thresholds.keys() == rule.thresholds.keys()


def test_external_validation_uses_frozen_rule_without_refit():
    march = _frame(3, days=16)
    february = _frame(2, days=12)
    rule = fit_candidate_rule(march)
    evaluation = evaluate_external_month(february, rule)
    combined_10 = evaluation.loc[
        evaluation["horizon_min"].eq(10)
        & evaluation["stage"].eq("candidate_combined")
    ].iloc[0]
    assert combined_10["selected_executable_n"] > 0
    assert combined_10["gross_matched_lift_pct"] > 0
    assert combined_10["base_matched_lift_pct"] > 0


def test_external_validation_report_marks_rule_frozen():
    march = _frame(3, days=16)
    february = _frame(2, days=12)
    rule = fit_candidate_rule(march)
    report, evaluation, thresholds = render_external_validation(february, rule)
    assert "rule_status=FROZEN" in report
    assert "2026-02-" in report
    assert not evaluation.empty
    assert not thresholds.empty
