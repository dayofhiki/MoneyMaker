import numpy as np
import pandas as pd
import pytest

from victory_trader import nearby_state_momentum as exp
from victory_trader.compact_momentum_representation import fit_transform, independent_weights
from victory_trader.within_episode_momentum import PACKAGES, pair_manifest


def fixture():
    rows = []
    for day, ticker, clocks, labels in (("2026-05-05", "A", [1000, 31000, 31001], [6., 0., 0.]),
                                       ("2026-05-05", "B", [1000, 2000], [0., 6.]),
                                       ("2026-05-06", "C", [1000, 2000], [6., 0.]),
                                       ("2026-05-06", "D", [1000, 50000], [6., 0.])):
        for clock, label in zip(clocks, labels):
            rows.append({"trading_day": day, "ticker": ticker, "hot_t": 0, "decision_t": clock,
                "observation_available": True, "label_complete": True, exp.CEILING: label,
                **{f: clock/1000+(label >= 5) for f in PACKAGES["W"]}})
    frame = pd.DataFrame(rows)
    for arm in exp.ARMS:
        frame[f"{arm}_score"] = frame[exp.CEILING]
    return frame


def test_exact_proximity_keeps_every_eligible_pair_and_all_identities():
    frame = fixture()
    identities = frame[exp.KEYS].drop_duplicates()
    extra = identities.iloc[:1].assign(ticker="UNOBSERVED")
    pairs, ledger = exp.local_manifest(frame, pd.concat([identities, extra]))
    assert len(pairs) == 3 and len(ledger) == 5
    assert 1 in pairs.negative_index.to_numpy() and 2 not in pairs.negative_index.to_numpy()
    assert pairs.absolute_gap_ms.max() == 30000
    assert not ledger.loc[ledger.ticker.isin(["D", "UNOBSERVED"]), "local_evaluable"].any()


def test_local_pair_weights_equal_eligible_day_and_episode_mass():
    frame = fixture()
    pairs, _ = exp.local_manifest(frame, frame[exp.KEYS].drop_duplicates())
    assert pairs.pair_weight.sum() == pytest.approx(1)
    assert pairs.groupby("trading_day").pair_weight.sum().to_list() == pytest.approx([.5, .5])
    assert pairs.loc[pairs.ticker.eq("A"), "pair_weight"].sum() == pytest.approx(.25)


def test_censoring_and_no_local_pairs_are_explicit():
    frame = fixture()
    frame.loc[1, "label_complete"] = False
    pairs, ledger = exp.local_manifest(frame, frame[exp.KEYS].drop_duplicates())
    assert 1 not in pairs.negative_index.to_numpy()
    assert not ledger.loc[ledger.ticker.eq("A"), "local_evaluable"].any()
    frame[exp.CEILING] = 0.
    pairs, ledger = exp.local_manifest(frame, frame[exp.KEYS].drop_duplicates())
    assert pairs.empty and not ledger.local_evaluable.any()
    assert exp.local_metrics(frame, pairs)["W"]["local_pair_rank"] is None


def test_duplicate_states_or_identities_are_rejected():
    frame = fixture()
    ids = frame[exp.KEYS].drop_duplicates()
    with pytest.raises(ValueError, match="states"):
        exp.local_manifest(pd.concat([frame, frame.iloc[:1]]), ids)
    with pytest.raises(ValueError, match="identities"):
        exp.local_manifest(frame, pd.concat([ids, ids.iloc[:1]]))


def test_ties_and_direction_are_not_probabilities():
    frame = fixture()
    pairs, _ = exp.local_manifest(frame, frame[exp.KEYS].drop_duplicates())
    frame["A_score"] = 10.
    frame["U_score"] = -frame[exp.CEILING]
    result = exp.local_metrics(frame, pairs)
    assert result["W"]["local_pair_rank"] == 1.
    assert result["A"]["local_pair_rank"] == .5
    assert result["U"]["local_pair_rank"] == 0.


def test_local_bootstrap_requires_whole_episodes():
    frame = fixture()
    pairs, _ = exp.local_manifest(frame, frame[exp.KEYS].drop_duplicates())
    cache = exp.LocalCache(frame, pairs)
    m = np.ones(len(frame))
    m[0] = 2.
    with pytest.raises(ValueError, match="whole-episode"):
        cache.evaluate(m)
    m[:3] = 0
    assert cache.evaluate(m)["W"]["local_evaluable_episodes"] == 2
    report = exp.intervals(frame, pairs, draws=10)
    assert report["day"]["comparisons"]["W_minus_A"]["local_pair_rank_ci95"] == [0., 0.]


def test_subset_transform_exactly_preserves_raw_and_missing_columns():
    frame = fixture()
    frame.loc[0, exp.HISTORY[-1]] = np.nan
    full = fit_transform(frame, PACKAGES["W"], independent_weights(frame))
    sliced = exp.subset_transform(full.audit())
    indices = [full.features.index(f) for f in exp.HISTORY]
    np.testing.assert_array_equal(sliced.apply(frame), full.apply(frame)[:, indices+[i+len(full.features) for i in indices]])


def test_elapsed_fit_is_fixed_and_current_state_only():
    frame = fixture()
    pairs, _ = pair_manifest(frame)
    full = fit_transform(frame, PACKAGES["W"], independent_weights(frame))
    model, transform, audit = exp.fit_elapsed(frame, pairs, full.audit())
    assert audit["effective_weight_sum"] == pytest.approx(4)
    assert audit["preprocessing"]["raw_features"] == list(exp.HISTORY)
    assert not model.fit_intercept
    score = exp.score_elapsed(frame, model, transform)
    changed = frame.copy()
    changed.loc[1:, list(exp.HISTORY)] = 1e9
    altered = exp.score_elapsed(changed, model, transform)
    assert score.A_score.iloc[0] == altered.A_score.iloc[0]
    assert score.groupby(exp.KEYS).A0_score.nunique().eq(1).all()


def test_wrong_training_transform_and_sealed_scope_are_rejected():
    frame = fixture()
    pairs, _ = pair_manifest(frame)
    audit = fit_transform(frame, PACKAGES["W"], independent_weights(frame)).audit()
    audit["mean"] = [v+10 for v in audit["mean"]]
    with pytest.raises(ValueError, match="training scope"):
        exp.fit_elapsed(frame, pairs, audit)
    frame.loc[0, "trading_day"] = "2026-06-01"
    with pytest.raises(ValueError, match="training scope"):
        exp.fit_elapsed(frame, pairs, audit)


def test_single_class_elapsed_fit_stays_unscored():
    frame = fixture()
    frame[exp.CEILING] = 0.
    pairs, _ = pair_manifest(frame)
    audit = fit_transform(frame, PACKAGES["W"], independent_weights(frame)).audit()
    model, transform, fit = exp.fit_elapsed(frame, pairs, audit)
    assert model is None and fit["no_pair_fit"]
    scored = exp.score_elapsed(frame, model, transform)
    assert scored.A_score.isna().all() and len(scored) == len(frame)
