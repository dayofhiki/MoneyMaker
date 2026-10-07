import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from victory_trader import state_evolution_momentum as exp
from victory_trader.clock_momentum_increment import within_day_auc


def trajectory_fixture():
    rows = []
    for day, ticker in (("2026-05-05", "A"), ("2026-05-05", "B")):
        for j, t in enumerate((1000, 11000, 41000, 80000)):
            rows.append({"trading_day": day, "ticker": ticker, "hot_t": 0, "decision_t": t,
                         **{c: float(j) for c in exp.LAG_INPUTS}, "label_complete": j != 1,
                         "observation_available": True, exp.CEILING: float(j)})
    return pd.DataFrame(rows)


def test_sparse_lag_is_last_completed_at_or_before_cutoff_and_resets():
    result = exp.trajectory_features(trajectory_fixture())
    assert result.trajectory_lag_age_s.tolist() == pytest.approx([np.nan, np.nan, 30., 39.]*2, nan_ok=True)
    assert result[exp.DELTAS[0]].tolist() == pytest.approx([np.nan, np.nan, 1., 1.]*2, nan_ok=True)
    assert result.trajectory_elapsed_s.tolist() == [0., 10., 40., 79.]*2
    assert result.trajectory_log_observations.tolist() == pytest.approx(list(np.log1p([1, 2, 3, 4]))*2)


def test_future_values_and_outcomes_cannot_change_prior_trajectory():
    frame = trajectory_fixture()
    before = exp.trajectory_features(frame)
    changed = frame.copy()
    changed.loc[changed.decision_t.eq(80000), list(exp.LAG_INPUTS)] = 1e9
    changed[exp.CEILING] = -1e9
    changed.label_complete = False
    after = exp.trajectory_features(changed)
    cols = list(exp.HISTORY+exp.DELTAS)
    pd.testing.assert_frame_equal(before.loc[before.decision_t.lt(80000), cols], after.loc[after.decision_t.lt(80000), cols])


def test_censored_states_are_part_of_observed_history():
    result = exp.trajectory_features(trajectory_fixture())
    # The exact41s state's lag11s is censored; censor filtering before history
    # would select1s instead, changing both age and delta.
    assert result.loc[result.decision_t.eq(41000), exp.DELTAS[0]].eq(1.).all()
    assert result.loc[result.decision_t.eq(41000), exp.HISTORY[-1]].eq(30.).all()


def test_duplicate_clock_sealed_date_and_window_rejected():
    frame = trajectory_fixture()
    with pytest.raises(ValueError):
        exp.trajectory_features(pd.concat([frame, frame.iloc[:1]]))
    for field, value in (("trading_day", "2026-06-01"), ("decision_t", 300001)):
        changed = frame.copy()
        changed.loc[0, field] = value
        with pytest.raises(ValueError):
            exp.trajectory_features(changed)


def score_fixture():
    frame = pd.DataFrame({"trading_day": ["2026-05-11"]*6+["2026-05-12"]*3,
        "ticker": ["A", "A", "A", "B", "B", "C", "D", "D", "E"],
        "hot_t": [0]*9, "decision_t": [1, 2, 3, 1, 2, 1, 1, 2, 1],
        "observation_available": True, "label_complete": True,
        exp.CEILING: [6., 0., 6., 0., 6., 0., 0., 6., 0.]})
    for a in exp.ARMS:
        frame[f"{a}_p_net_5"] = [.9, .2, .7, .7, .8, .4, .3, .8, .3]
        frame[f"{a}_prior_net_5"] = .1
    return frame


def brute_between(frame, arm, weights):
    y, p = frame[exp.CEILING].ge(5).to_numpy(), frame[f"{arm}_p_net_5"].to_numpy()
    keys = list(map(tuple, frame[exp.KEYS].to_numpy()))
    numerator = denominator = 0.
    for i in np.flatnonzero(y):
        for j in np.flatnonzero(~y):
            if frame.trading_day.iloc[i] == frame.trading_day.iloc[j] and keys[i] != keys[j]:
                w = weights[i]*weights[j]
                denominator += w
                numerator += w*((p[i] > p[j])+.5*(p[i] == p[j]))
    return numerator/denominator if denominator else None


def test_fast_weighted_ranks_ap_and_between_episode_pair_exclusion():
    frame = score_fixture()
    cache = exp.RankCache(frame, ("C",))
    result = cache.evaluate()["C"]
    y, p = frame[exp.CEILING].ge(5), frame.C_p_net_5
    assert result["auc"] == pytest.approx(roc_auc_score(y, p, sample_weight=cache.w))
    assert result["ap"] == pytest.approx(average_precision_score(y, p, sample_weight=cache.w))
    assert result[exp.METRICS[0]] == pytest.approx(brute_between(frame, "C", cache.w))
    assert result["within_episode_auc"] == pytest.approx(1.)


def test_single_snapshot_matches_original_same_day_auc():
    frame = score_fixture().groupby(exp.KEYS, sort=False).head(1)
    assert exp.metrics(frame, ("C",))["C"][exp.METRICS[0]] == pytest.approx(within_day_auc(frame, "C"))


def test_whole_cluster_multiplicities_use_original_weights():
    frame = score_fixture()
    cache = exp.RankCache(frame, ("C",))
    m = np.array([2., 2., 2., 0., 0., 1., 3., 3., 2.])
    result = cache.evaluate(m)["C"]
    assert result[exp.METRICS[0]] == pytest.approx(brute_between(frame, "C", cache.w*m))
    with pytest.raises(ValueError, match="whole episodes"):
        cache.evaluate(np.array([1., 2., 1., 1., 1., 1., 1., 1., 1.]))


def test_constant_first_scores_have_no_within_episode_ranking():
    frame = exp.broadcast_first(score_fixture())
    assert exp.metrics(frame, ("C0",))["C0"]["within_episode_auc"] == pytest.approx(.5)
    # First clock, including incomplete first outcome, sets the broadcast score.
    incomplete = score_fixture()
    incomplete.loc[0, "label_complete"] = False
    result = exp.broadcast_first(incomplete)
    assert result.loc[result.ticker.eq("A"), "C0_p_net_5"].eq(.9).all()


def test_bootstrap_reports_valid_draws_with_all_correlated_states():
    result = exp.intervals(exp.broadcast_first(score_fixture()), draws=10)
    for scheme in ("day", "ticker_day"):
        assert result[scheme]["draws"] == 10
        assert result[scheme]["comparisons"]["T_minus_S"]["brier_valid_draws"] == 10
        assert result[scheme]["comparisons"]["T_minus_S"]["brier_ci95"] == [0., 0.]


def test_chronological_excludes_held_and_future_days(monkeypatch):
    parts = []
    for day in exp.CROSSFIT_DAYS:
        frame = trajectory_fixture().iloc[:4].copy()
        frame.trading_day = day
        parts.append(frame)
    states = exp.trajectory_features(pd.concat(parts, ignore_index=True))
    def fake_score(train, held):
        assert train.trading_day.max() < held.trading_day.min()
        assert exp.complete_mask(train).all()
        result = held.copy()
        for a in exp.ARMS:
            result[f"{a}_p_net_5"] = .1
            result[f"{a}_prior_net_5"] = .1
        return result, {"fit_days": sorted(train.trading_day.unique())}
    monkeypatch.setattr(exp, "score", fake_score)
    scored, folds = exp.chronological(states)
    assert len(scored) == 12 and set(folds) == set(exp.CROSSFIT_DAYS[1:])


def test_first_replay_checks_all_labels_and_missingness():
    states = score_fixture().iloc[:6].copy()
    for c in exp.CLOCK_FEATURES:
        states[c] = np.nan
    states["label_entry_price"] = 10.
    saved = states.sort_values("decision_t").groupby(exp.KEYS, sort=False).head(1)
    exp.verify_first(states, saved)
    for column in (exp.CEILING, "label_entry_price", "decision_t"):
        changed = states.copy()
        changed.loc[0, column] += 1
        with pytest.raises(AssertionError):
            exp.verify_first(changed, saved)
