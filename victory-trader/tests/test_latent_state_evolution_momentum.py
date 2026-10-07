import numpy as np
import pandas as pd
import pytest

from victory_trader import latent_state_evolution_momentum as exp
from victory_trader.compact_momentum_representation import fit_transform, independent_weights
from victory_trader.nearby_state_momentum import local_manifest


def fixture(labels=(0., 6., 0., 6.)):
    rows = []
    for day, ticker in (("2026-05-05", "A"), ("2026-05-05", "B"), ("2026-05-06", "C")):
        for j, label in enumerate(labels):
            rows.append({"trading_day": day, "ticker": ticker, "hot_t": 0, "decision_t": (j+1)*1000,
                "observation_available": True, "label_complete": True, exp.CEILING: label,
                **{f: float(j) for f in set().union(*exp.PACKAGES.values()) if f != exp.GAP}})
    return exp.causal_states(pd.DataFrame(rows))


def initial_audit(frame, constant=.2):
    features = exp.PREVIOUS_PACKAGES["G"]
    transform = fit_transform(frame, features, independent_weights(frame))
    return {"fit_days": ["2026-05-05", "2026-05-06"], "preprocessing": transform.audit(),
        "single_class_fallback": True, "prior": constant}


def held(frame):
    result = frame.copy()
    result.trading_day = result.trading_day.replace({"2026-05-05": "2026-05-11", "2026-05-06": "2026-05-12"})
    return result


def test_actual_gaps_preserve_order_and_use_censored_observations():
    frame = fixture().iloc[:4].copy()
    frame.decision_t = [1000, 2000, 62000, 302000]
    frame.loc[1, "label_complete"] = False
    shuffled = frame.iloc[[3, 1, 0, 2]]
    result = exp.causal_states(shuffled)
    assert result.index.tolist() == shuffled.index.tolist()
    assert pd.isna(result.loc[0, exp.GAP])
    np.testing.assert_allclose(result.loc[[1, 2, 3], exp.GAP], np.log1p([1, 60, 240]))
    assert result.loc[2, "state_previous_t"] == 2000


def test_training_transitions_do_not_bridge_incomplete_original_endpoint():
    frame = fixture().iloc[:4].copy()
    frame.loc[1, "label_complete"] = False
    pairs, ledger = exp.transition_manifest(frame)
    assert pairs[["previous_index", "current_index"]].to_numpy().tolist() == [[2, 3]]
    assert pairs[["previous_label", "current_label"]].to_numpy().tolist() == [[0, 1]]
    assert ledger.iloc[0][["observed_states", "complete_states", "transitions"]].tolist() == [4, 3, 1]


def test_shared_transform_and_conditional_episode_mass_are_independent():
    frame = fixture()
    pairs, _ = exp.transition_manifest(frame)
    bundles, audits = exp.fit_transitions(frame, pairs)
    for arm in exp.PACKAGES:
        assert bundles[arm][0][1] is bundles[arm][1][1]
        assert audits[arm]["transform_states"] == 9
        assert audits[arm]["transform_episodes"] == 3
        assert audits[arm]["transform_weight_sum"] == pytest.approx(3)
        for previous in ("0", "1"):
            assert audits[arm]["conditional"][previous]["weight_sum"] == pytest.approx(3)
        expected = fit_transform(frame.loc[pairs.current_index], exp.PACKAGES[arm], independent_weights(frame.loc[pairs.current_index]))
        assert audits[arm]["preprocessing"] == expected.audit()


def test_prediction_only_recursion_and_common_initial_probabilities():
    frame = fixture()
    pairs, _ = exp.transition_manifest(frame)
    bundles, _ = exp.fit_transitions(frame, pairs)
    for arm in exp.PACKAGES:
        for previous, q in ((0, .1), (1, .8)):
            _, transform, _ = bundles[arm][previous]
            bundles[arm][previous] = (None, transform, q)
    result, _ = exp.infer(held(frame), bundles, initial_audit(frame))
    expected = [.2]
    for _ in range(3):
        expected.append((1-expected[-1])*.1+expected[-1]*.8)
    for arm in (*exp.PACKAGES, "K"):
        np.testing.assert_allclose(result.loc[:3, f"R324_{arm}_p_net_5"], expected)
    assert result.R324_B_p_net_5.eq(.2).all() and result.R324_B0_p_net_5.eq(.2).all()


def test_inference_needs_no_outcomes_and_label_censor_mutations_do_not_change_it():
    frame = fixture(labels=(0., 0., 6., 6., 0.))
    pairs, _ = exp.transition_manifest(frame)
    bundles, _ = exp.fit_transitions(frame, pairs)
    evaluation, audit = held(frame), initial_audit(frame)
    original, _ = exp.infer(evaluation, bundles, audit)
    evaluation[exp.CEILING] = np.nan
    evaluation.label_complete = False
    evaluation.observation_available = False
    mutated, _ = exp.infer(evaluation, bundles, audit)
    causal_only, _ = exp.infer(evaluation.drop(columns=[exp.CEILING, "label_complete", "observation_available"]), bundles, audit)
    columns = [f"R324_{a}_p_net_5" for a in exp.ARMS]
    pd.testing.assert_frame_equal(original[columns], mutated[columns])
    pd.testing.assert_frame_equal(original[columns], causal_only[columns])


def test_future_features_and_observation_truncation_do_not_change_present():
    frame = fixture(labels=(0., 0., 6., 6., 0.))
    pairs, _ = exp.transition_manifest(frame)
    bundles, _ = exp.fit_transitions(frame, pairs)
    assert bundles["F"][0][0] is not None and bundles["F"][1][0] is not None
    evaluation, audit = held(frame), initial_audit(frame)
    original, _ = exp.infer(evaluation, bundles, audit)
    future = evaluation.copy()
    future.loc[future.decision_t.gt(2000), list(set().union(*exp.PACKAGES.values()))] = 1e9
    mutated, _ = exp.infer(future, bundles, audit)
    prefix, _ = exp.infer(evaluation.loc[evaluation.decision_t.le(2000)], bundles, audit)
    columns = [f"R324_{a}_p_net_5" for a in exp.ARMS]
    pd.testing.assert_frame_equal(original.loc[prefix.index, columns], mutated.loc[prefix.index, columns])
    pd.testing.assert_frame_equal(original.loc[prefix.index, columns], prefix[columns])


def test_no_support_keeps_rows_and_initial_then_leaves_recursion_missing():
    frame = fixture(labels=(0., 0., 0.))
    pairs, _ = exp.transition_manifest(frame)
    bundles, fits = exp.fit_transitions(frame, pairs)
    result, audit = exp.infer(held(frame), bundles, initial_audit(frame))
    assert len(result) == len(frame)
    for arm in (*exp.PACKAGES, "K"):
        assert audit["unscored_observations"][arm] == 6
        for _, group in result.groupby(exp.KEYS):
            assert group[f"R324_{arm}_p_net_5"].iloc[0] == .2
            assert group[f"R324_{arm}_p_net_5"].iloc[1:].isna().all()
    assert fits["F"]["conditional"]["1"]["no_support"]
    assert fits["F"]["conditional"]["0"]["single_class_fallback"]


def test_single_class_conditionals_use_own_priors_without_classifier():
    frame = fixture()
    pairs, _ = exp.transition_manifest(frame)
    bundles, fits = exp.fit_transitions(frame, pairs)
    assert bundles["F"][0][0] is None and bundles["F"][0][2] == 1.
    assert bundles["F"][1][0] is None and bundles["F"][1][2] == 0.
    assert fits["F"]["conditional"]["0"]["single_class_fallback"]


def test_initial_scope_and_saved_score_replay_are_enforced():
    frame = fixture()
    audit = initial_audit(frame)
    with pytest.raises(ValueError, match="strictly preceding"):
        exp.frozen_initial(frame, audit)
    evaluation = held(frame)
    evaluation["G_p_net_5"] = .3
    with pytest.raises(ValueError, match="replay changed"):
        exp.frozen_initial(evaluation, audit)
    evaluation.G_p_net_5 = .2
    _, error = exp.frozen_initial(evaluation, audit)
    assert error == 0.


@pytest.mark.parametrize("corruption", ["duplicate_clock", "sealed_day", "manifest"])
def test_invalid_scope_and_transition_manifest_are_rejected(corruption):
    frame = fixture()
    if corruption == "duplicate_clock":
        frame.loc[1, "decision_t"] = frame.loc[0, "decision_t"]
        with pytest.raises(ValueError, match="unique"):
            exp.causal_states(frame)
    elif corruption == "sealed_day":
        frame.loc[0, "trading_day"] = "2026-06-01"
        with pytest.raises(ValueError, match="dates"):
            exp.causal_states(frame)
    else:
        pairs, _ = exp.transition_manifest(frame)
        pairs.loc[0, "current_index"] = 9
        with pytest.raises((ValueError, AssertionError)):
            exp.fit_transitions(frame, pairs)


def test_report_preserves_original_local_pairs_and_fixed_weight_bootstrap():
    frame = held(fixture())
    for arm in exp.ARMS:
        frame[f"R324_{arm}_p_net_5"] = np.where(frame[exp.CEILING].ge(5), .8, .2)
        frame[f"R324_{arm}_prior_net_5"] = .2
    pairs, _ = local_manifest(frame, frame[exp.KEYS].drop_duplicates())
    result = exp.report_metrics(frame, pairs)
    assert result["F"]["local_pair_rank"] == result["F"]["within_episode_auc"] == 1.
    assert result["F"]["local_evaluable_episodes"] == 3
    intervals = exp.intervals(frame, pairs, draws=10)
    assert intervals["day"]["comparisons"]["F_minus_B"]["local_pair_rank_ci95"] == [0., 0.]
