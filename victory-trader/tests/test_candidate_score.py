import pandas as pd

from victory_trader.candidate_score import (
    ALPHA_FEATURES,
    LIQUIDITY_FEATURES,
    apply_candidate_rule,
    evaluate_candidate_rule,
    fit_candidate_rule,
)


def _frame() -> pd.DataFrame:
    rows = []
    horizons = (1, 2, 5, 10, 15, 30, 60)
    for day in range(1, 21):
        for i in range(80):
            low_is_good = float(i)
            row = {
                "trading_day": f"2026-03-{day:02d}",
                "threshold_pct": 10.0 if i < 60 else 20.0,
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


def test_candidate_rule_is_fit_only_on_early_dates_and_is_fixed():
    frame = _frame()
    rule = fit_candidate_rule(frame)
    assert rule.train_dates[-1] < rule.holdout_dates[0]
    assert "10.0" in rule.thresholds
    params = rule.thresholds["10.0"]
    assert set(params["liquidity_cuts"]) == set(LIQUIDITY_FEATURES)
    assert set(params["alpha_references"]) == set(ALPHA_FEATURES)


def test_candidate_score_prefers_low_alpha_features_after_liquidity_gate():
    frame = _frame()
    rule = fit_candidate_rule(frame)
    scored = apply_candidate_rule(frame, rule)
    holdout = scored.loc[scored["trading_day"].isin(rule.holdout_dates)]
    candidates = holdout.loc[holdout["candidate_v1"]]
    assert not candidates.empty
    assert candidates["alpha_score_v1"].mean() > holdout["alpha_score_v1"].mean()
    for feature in ALPHA_FEATURES:
        assert candidates[feature].mean() < holdout[feature].mean()


def test_candidate_evaluation_reports_all_horizons_and_stages():
    frame = _frame()
    rule = fit_candidate_rule(frame)
    evaluation = evaluate_candidate_rule(frame, rule)
    assert set(evaluation["horizon_min"]) == {1, 2, 5, 10, 15, 30, 60}
    assert set(evaluation["stage"]) == {
        "baseline",
        "liquidity_only",
        "alpha_only",
        "candidate_combined",
    }
    combined_10 = evaluation.loc[
        evaluation["horizon_min"].eq(10)
        & evaluation["stage"].eq("candidate_combined")
    ].iloc[0]
    assert combined_10["gross_matched_lift_pct"] > 0
    assert combined_10["base_matched_lift_pct"] > 0
