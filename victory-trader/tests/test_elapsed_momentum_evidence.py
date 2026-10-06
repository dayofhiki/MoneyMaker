import json

import numpy as np
import pandas as pd
import pytest

from victory_trader.elapsed_momentum_evidence import (
    ARMS, DIRECTION, EVAL_DAYS, PACKAGES, SUPPORT, complete_mask, contexts_for,
    expanding_score, gate, paired_intervals, report, score_fit, temporal_features,
)
from victory_trader.feasible_upside_observability import CEILING
from victory_trader.lagged_minute_context import CROSSFIT_DAYS
from victory_trader.pending_exit_integrity import session_limits


def history(seconds=(0, 68, 99), prices=(10., 10., 11.)):
    opening, closing = session_limits(EVAL_DAYS[0])
    bars = pd.DataFrame({"t": opening+np.array(seconds)*1000, "c": prices,
                         "o": prices, "h": prices, "l": prices, "v": 100., "n": 2.})
    source = pd.DataFrame({"trading_day": [EVAL_DAYS[0]], "ticker": ["X"], "decision_t": [opening+100000]})
    return source, bars, {(EVAL_DAYS[0], "X"): (bars, opening, closing)}


def test_stale_baseline_single_print_and_actual_elapsed_rate_are_distinct():
    source, _, contexts = history()
    f = temporal_features(source, contexts).iloc[0]
    assert f.evidence_previous_interprint_s == 31
    assert f.evidence_baseline_age_s_10s == 31
    assert f.evidence_internal_span_s_10s == 0
    assert f.evidence_maximum_interprint_s_10s == 31
    assert np.isnan(f.evidence_internal_log_rate_10s)
    assert f.evidence_baseline_log_rate_10s == pytest.approx(100*np.log(1.1)/31)
    assert np.isnan(f.evidence_baseline_age_s_120s)


def test_two_print_direction_and_acceleration_use_actual_spans():
    source, _, contexts = history((0, 59, 88, 94, 99), (8., 9., 10., 10.5, 11.))
    f = temporal_features(source, contexts).iloc[0]
    assert f.evidence_internal_span_s_10s == 5
    assert f.evidence_internal_log_rate_10s == pytest.approx(100*np.log(11/10.5)/5)
    assert f.evidence_baseline_age_s_10s == 11
    assert f.evidence_baseline_age_s_30s == 40
    assert f.evidence_acceleration_10_30 == pytest.approx(100*np.log(1.1)/11-100*np.log(11/9)/40)


def test_current_incomplete_second_and_future_path_cannot_change_features():
    source, bars, contexts = history()
    before = temporal_features(source, contexts)
    future = bars.iloc[-1:].assign(t=int(source.decision_t.iloc[0]), c=999., o=999., h=999., l=999.)
    longer = pd.concat([bars, future], ignore_index=True)
    key = (EVAL_DAYS[0], "X")
    after = temporal_features(source.assign(future_label=999), {key: (longer, contexts[key][1], contexts[key][2])})
    pd.testing.assert_frame_equal(before[list(SUPPORT+DIRECTION)], after[list(SUPPORT+DIRECTION)])


def test_missing_history_is_explicit_and_invalid_scope_is_rejected():
    source, bars, contexts = history((99,), (11.,))
    f = temporal_features(source, contexts).iloc[0]
    assert np.isnan(f.evidence_previous_interprint_s)
    assert all(np.isnan(f[name]) for name in DIRECTION)
    assert f.evidence_internal_span_s_10s == 0
    with pytest.raises(ValueError, match="missing raw history"):
        temporal_features(source, {})
    with pytest.raises(ValueError, match="sorted unique"):
        temporal_features(source, {(EVAL_DAYS[0], "X"): (pd.concat([bars, bars]), 0, 0)})
    raw = bars.assign(trading_day="2026-06-15", ticker="X")
    with pytest.raises(ValueError, match="fixed May dates"):
        contexts_for(raw, EVAL_DAYS, {("2026-06-15", "X")})
    with pytest.raises(ValueError, match="identity scope"):
        contexts_for(bars.assign(trading_day=EVAL_DAYS[0], ticker="Y"), EVAL_DAYS, {(EVAL_DAYS[0], "X")})


def model_rows(days):
    rows = []
    columns = tuple(dict.fromkeys(name for features in PACKAGES.values() for name in features))
    for d, day in enumerate(days):
        for i in range(4):
            row = dict.fromkeys(columns, float(i+d/10))
            row.update(trading_day=day, ticker=f"X{i}", hot_t=d*100000, decision_t=d*100000+1000,
                       observation_available=True, label_complete=True, missing_reason=None)
            row[CEILING] = 6. if i == 3 else -1.
            rows.append(row)
    return pd.DataFrame(rows)


def test_expanding_fits_exclude_current_later_labels_and_preprocessing():
    original, evaluation = model_rows(CROSSFIT_DAYS), model_rows(EVAL_DAYS)
    before, audit = expanding_score(original, evaluation)
    changed = evaluation.copy()
    changed.loc[changed.trading_day.ge(EVAL_DAYS[3]), CEILING] = 20.
    changed.loc[changed.trading_day.gt(EVAL_DAYS[3]), list(SUPPORT)] = 1e10
    after, other = expanding_score(original, changed)
    columns = [f"{a}_p_net_5" for a in ARMS]
    pd.testing.assert_frame_equal(before.loc[before.trading_day.le(EVAL_DAYS[3]), columns],
                                  after.loc[after.trading_day.le(EVAL_DAYS[3]), columns])
    assert audit[EVAL_DAYS[3]] == other[EVAL_DAYS[3]]
    for day, fold in audit.items():
        assert all(fit_day < day for fit_day in fold["fit_days"])
        expected = len(original)+4*EVAL_DAYS.index(day)
        assert fold["models"]["T"]["train_episodes"] == expected
        assert fold["models"]["T"]["weight_sum"] == pytest.approx(expected)


def test_incomplete_cases_keep_scores_unobserved_keep_nan_and_do_not_train():
    original, evaluation = model_rows(CROSSFIT_DAYS), model_rows(EVAL_DAYS)
    evaluation.loc[0, ["label_complete", "missing_reason", CEILING]] = [False, "no_fill", np.nan]
    evaluation.loc[1, ["observation_available", "label_complete", "missing_reason", CEILING]] = [False, False, "no_print", np.nan]
    scored, audit = expanding_score(original, evaluation)
    assert scored.loc[0, [f"{a}_p_net_5" for a in ARMS]].notna().all()
    assert scored.loc[1, [f"{a}_p_net_5" for a in ARMS]].isna().all()
    assert audit[EVAL_DAYS[1]]["earlier_broad_first_episodes"] == 2
    assert not complete_mask(scored).iloc[:2].any()
    with pytest.raises(ValueError, match="fixed May evaluation"):
        expanding_score(original, evaluation.assign(trading_day="2026-06-15"))


def test_packages_preserve_nominal_comparator_and_separate_support_control():
    from victory_trader.clock_momentum_increment import MOMENTUM, SIGNAL
    assert PACKAGES["M"] == MOMENTUM
    assert len(SUPPORT) == 10 and len(DIRECTION) == 7
    assert not set(SIGNAL+DIRECTION) & set(PACKAGES["K"])
    assert set(PACKAGES["M"]) <= set(PACKAGES["T"])
    assert len(PACKAGES["T"]) == 45
    assert all(len(v) == len(set(v)) for v in PACKAGES.values())


def test_zero_coverage_and_rank_draws_do_not_create_fake_success():
    original, evaluation = model_rows(CROSSFIT_DAYS), model_rows(EVAL_DAYS)
    evaluation["observation_available"] = False
    evaluation["label_complete"] = False
    evaluation[CEILING] = np.nan
    evaluation["missing_reason"] = "no_print"
    scored, _ = score_fit(original, evaluation)
    result = report(scored)
    assert result["complete_coverage"] == 0 and not all(gate(result).values())
    json.dumps(result, allow_nan=False)
    one_class = model_rows([EVAL_DAYS[0]])
    one_class[CEILING] = 0.
    scored, _ = score_fit(original, one_class)
    intervals = paired_intervals(scored, draws=5)
    for cluster in intervals.values():
        for comparison in cluster["comparisons"].values():
            assert comparison["within_day_auc_ci95"] is None
            assert comparison["auc_valid_draws"] == 0
            assert comparison["brier_valid_draws"] == 5
