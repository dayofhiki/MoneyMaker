import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from victory_trader import within_episode_momentum as exp
from victory_trader.state_evolution_momentum import metrics as previous_metrics


def fixture():
    rows = []
    for day, ticker, labels in (("2026-05-05", "A", [6., 0., 6.]),
                                ("2026-05-05", "B", [0., 6.]),
                                ("2026-05-06", "C", [6., 0.]),
                                ("2026-05-06", "D", [0., 0.])):
        for j, label in enumerate(labels):
            rows.append({"trading_day": day, "ticker": ticker, "hot_t": 0, "decision_t": (j+1)*1000,
                "observation_available": True, "label_complete": True, exp.CEILING: label,
                **{f: float(j)+(label >= 5)*2 for f in exp.PACKAGES["W"]}})
    return pd.DataFrame(rows)


def test_all_pairs_preserve_episode_boundaries_and_independent_mass():
    frame = fixture()
    pairs, ledger = exp.pair_manifest(frame)
    exp.verify_pairs(frame, pairs)
    assert len(pairs) == 4 and int(ledger.mixed_label.sum()) == 3
    assert pairs.pair_weight.sum() == pytest.approx(3)
    day_mass = pairs.groupby("trading_day").pair_weight.sum()
    assert day_mass.iloc[0] == pytest.approx(day_mass.iloc[1])
    ep_mass = pairs.groupby(exp.KEYS).pair_weight.sum()
    assert ep_mass.loc[("2026-05-05", "A", 0)] == pytest.approx(ep_mass.loc[("2026-05-05", "B", 0)])


def test_censored_endpoint_never_becomes_a_negative_training_pair():
    frame = fixture()
    frame.loc[1, "label_complete"] = False
    frame.loc[1, exp.CEILING] = np.nan
    pairs, ledger = exp.pair_manifest(frame)
    assert 1 not in pairs.negative_index.to_numpy()
    assert ledger.loc[ledger.ticker.eq("A"), "mixed_label"].eq(False).all()


def test_scope_duplicate_and_cross_episode_pairs_rejected():
    frame = fixture()
    changed = frame.copy()
    changed.loc[0, "trading_day"] = "2026-06-01"
    with pytest.raises(ValueError):
        exp.pair_manifest(changed)
    with pytest.raises(ValueError):
        exp.pair_manifest(pd.concat([frame, frame.iloc[:1]]))
    pairs, _ = exp.pair_manifest(frame)
    pairs.loc[0, "negative_index"] = 3
    with pytest.raises(ValueError, match="boundary"):
        exp.verify_pairs(frame, pairs)


def test_single_class_training_is_explicitly_unscored():
    frame = fixture()
    frame[exp.CEILING] = 0.
    pairs, ledger = exp.pair_manifest(frame)
    model, transform, audit = exp.fit_pairwise(frame, pairs, exp.PACKAGES["V"])
    assert model is None and audit["no_pair_fit"] and not ledger.mixed_label.any()
    assert transform.apply(frame).shape[0] == len(frame)


def test_pairwise_fit_has_no_intercept_and_finite_relative_scores():
    frame = fixture()
    pairs, _ = exp.pair_manifest(frame)
    model, transform, audit = exp.fit_pairwise(frame, pairs, exp.PACKAGES["V"])
    assert model.fit_intercept is False and audit["intercept"] == 0.
    assert audit["effective_weight_sum"] == pytest.approx(3)
    assert audit["transform_episodes"] == 4
    assert np.isfinite(model.decision_function(transform.apply(frame))).all()


def test_matched_state_classifier_uses_exact_pair_endpoint_marginals():
    frame = fixture()
    pairs, _ = exp.pair_manifest(frame)
    _, transform, pair_audit = exp.fit_pairwise(frame, pairs, exp.PACKAGES["W"])
    model, audit = exp.fit_matched_classifier(frame, pairs, transform)
    assert audit["effective_weight_sum"] == pytest.approx(pair_audit["effective_weight_sum"])
    assert audit["positive_weight"] == pytest.approx(1.5)
    assert audit["negative_weight"] == pytest.approx(1.5)
    assert audit["mixed_training_episodes"] == 3 and model.fit_intercept is False
    assert audit["preprocessing"] == pair_audit["preprocessing"]


def test_inference_needs_only_current_state_and_does_not_refit():
    frame = fixture()
    pairs, _ = exp.pair_manifest(frame)
    model, transform, audit = exp.fit_pairwise(frame, pairs, exp.PACKAGES["W"])
    first = frame.iloc[:1]
    before = model.decision_function(transform.apply(first))
    changed = pd.concat([first, frame.iloc[1:].assign(**{f: 1e9 for f in exp.PACKAGES["W"]})])
    after = model.decision_function(transform.apply(changed.iloc[:1]))
    np.testing.assert_array_equal(before, after)
    assert transform.audit() == audit["preprocessing"]


def test_official_training_transform_is_copied_and_wrong_scope_rejected():
    frame = fixture()
    pairs, _ = exp.pair_manifest(frame)
    _, transform, _ = exp.fit_pairwise(frame, pairs, exp.PACKAGES["W"])
    saved = transform.audit()
    _, copied, _ = exp.fit_pairwise(frame, pairs, exp.PACKAGES["W"], saved)
    assert copied.audit() == saved
    changed = dict(saved)
    changed["mean"] = [m+10 for m in saved["mean"]]
    with pytest.raises(ValueError, match="training scope"):
        exp.fit_pairwise(frame, pairs, exp.PACKAGES["W"], changed)


def scored_fixture():
    frame = fixture().copy()
    frame["trading_day"] = frame.trading_day.replace({"2026-05-05": "2026-05-11", "2026-05-06": "2026-05-12"})
    values = np.array([8., -2., 4., 1., 8., 8., -2., -2., -2.])
    for arm in ("U", "V", "W", "Z", "S", "T"):
        frame[f"{arm}_score"] = values
    return exp.broadcast_first(frame)


def test_relative_scores_rank_without_probability_bounds_or_brier():
    frame = scored_fixture()
    cache = exp.RankCache(frame)
    result = cache.evaluate()["W"]
    y = frame[exp.CEILING].ge(5)
    assert result["auc"] == pytest.approx(roc_auc_score(y, frame.W_score, sample_weight=cache.w))
    assert result["ap"] == pytest.approx(average_precision_score(y, frame.W_score, sample_weight=cache.w))
    assert result["within_episode_auc"] == pytest.approx(1.)
    assert "brier" not in result
    assert exp.metrics(frame)["W0"]["within_episode_auc"] == pytest.approx(.5)


def test_saved_probability_rank_baseline_replays_previous_evaluator():
    frame = scored_fixture()
    for a in ("S", "T"):
        frame[f"{a}_p_net_5"] = (frame[f"{a}_score"]+2)/11
        frame[f"{a}_score"] = frame[f"{a}_p_net_5"]
        frame[f"{a}_prior_net_5"] = .1
    prior, current = previous_metrics(frame, ("S", "T")), exp.metrics(frame)
    for a in ("S", "T"):
        for metric in exp.METRICS:
            assert current[a][metric] == pytest.approx(prior[a][metric])


def test_cluster_multiplicity_cannot_resample_individual_states():
    frame = scored_fixture()
    cache = exp.RankCache(frame)
    m = np.ones(len(frame))
    m[0] = 2
    with pytest.raises(ValueError, match="whole episodes"):
        cache.evaluate(m)
    result = exp.intervals(frame, draws=10)
    assert result["day"]["comparisons"]["W_minus_V"]["within_episode_auc_ci95"] == [0., 0.]


def test_chronological_excludes_current_and_future_days(monkeypatch):
    frame = fixture()
    parts = []
    for day in exp.CROSSFIT_DAYS:
        held = frame.iloc[:3].copy()
        held.trading_day = day
        held["S_p_net_5"] = .1
        held["T_p_net_5"] = .1
        parts.append(held)
    training = pd.concat(parts, ignore_index=True)
    def fake_score(train, pairs, held):
        assert train.trading_day.max() < held.trading_day.min()
        assert set(pairs.trading_day).issubset(set(train.trading_day))
        result = held.copy()
        for a in ("U", "V", "W", "Z", "S", "T"):
            result[f"{a}_score"] = .1
        return result, {"fit_days": sorted(train.trading_day.unique())}
    monkeypatch.setattr(exp, "score", fake_score)
    scored, folds = exp.chronological(training, training)
    assert len(scored) == 9 and set(folds) == set(exp.CROSSFIT_DAYS[1:])
