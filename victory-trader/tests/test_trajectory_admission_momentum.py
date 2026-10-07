import numpy as np
import pandas as pd
import pytest

from victory_trader import trajectory_admission_momentum as exp
from victory_trader.compact_momentum_representation import fit_transform, independent_weights


def fixture(day="2026-05-05"):
    rows = []
    for ticker in ("A", "B"):
        for j, clock in enumerate((1000, 10000, 31000, 45000)):
            rows.append({"trading_day": day, "ticker": ticker, "hot_t": 0, "decision_t": clock,
                "observation_available": True, "label_complete": True, exp.CEILING: 6. if j % 2 else 0.,
                **{f: float(j)+(ticker == "B") for f in exp.PREVIOUS_PACKAGES["T"]}})
    return pd.DataFrame(rows)


def head_fixture():
    training, held = fixture(), fixture("2026-05-06")
    heads = {}
    for arm, features in (("W", exp.PREVIOUS_PACKAGES["T"]), ("A", exp.PREVIOUS_PACKAGES["G"][-3:])):
        transform = fit_transform(training, features, independent_weights(training))
        coefficients = np.arange(2*len(features), dtype=float)/100
        heads[arm] = {"fit_days": ["2026-05-05"], "preprocessing": transform.audit(), "coefficients": coefficients.tolist(), "intercept": 0.}
        held[f"{arm}_score"] = transform.apply(held)@coefficients
    return training, held, heads


def test_head_normalization_replays_past_fit_and_uses_no_held_statistics():
    train, held, heads = head_fixture()
    original, audit = exp.normalized_head(train, held, heads["W"], "W")
    changed = held.copy()
    changed.loc[1:, list(exp.PREVIOUS_PACKAGES["T"])] = 1e9
    transform = exp.frozen_transform(heads["W"]["preprocessing"])
    changed.W_score = transform.apply(changed)@heads["W"]["coefficients"]
    altered, changed_audit = exp.normalized_head(train, changed, heads["W"], "W")
    assert audit == changed_audit and original[0] == altered[0]
    assert audit["fit_days"] == ["2026-05-05"] and audit["training_episodes"] == 2


def test_own_day_fit_or_wrong_saved_score_is_rejected():
    train, held, heads = head_fixture()
    wrong = dict(heads["W"], fit_days=["2026-05-06"])
    with pytest.raises(ValueError, match="preceding"):
        exp.normalized_head(train, held, wrong, "W")
    held.W_score += 1
    with pytest.raises(ValueError, match="replay"):
        exp.normalized_head(train, held, heads["W"], "W")


def test_first_and_latest_30s_lag_are_causal_without_forward_fill():
    frame = fixture()
    values = np.arange(len(frame), dtype=float)
    result = exp.causal_changes(frame, {"W": values, "A": -values})
    assert result.bridge_lag_decision_t.iloc[:2].isna().all()
    assert result.bridge_lag_decision_t.iloc[2] == 1000
    assert result.bridge_lag_decision_t.iloc[3] == 10000
    assert result.bridge_W_from_first.iloc[2] == 2
    assert result.bridge_W_lag30_change.iloc[3] == 2
    assert result.groupby(exp.KEYS).bridge_first_decision_t.nunique().eq(1).all()


def test_future_mutation_does_not_change_earlier_trajectory():
    frame = fixture()
    scores = np.arange(len(frame), dtype=float)
    before = exp.causal_changes(frame, {"W": scores, "A": scores})
    changed = scores.copy()
    changed[2:4] = 1e9
    after = exp.causal_changes(frame, {"W": changed, "A": changed})
    pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])


def test_censored_observation_remains_in_causal_lag_history():
    frame = fixture()
    frame.loc[1, "label_complete"] = False
    scores = np.arange(len(frame), dtype=float)
    result = exp.causal_changes(frame, {"W": scores, "A": scores})
    assert len(result) == len(frame) and not result.label_complete.iloc[1]
    assert result.bridge_lag_decision_t.iloc[3] == frame.decision_t.iloc[1]


def test_duplicate_and_missing_normalized_scores_rejected():
    frame = fixture()
    scores = np.arange(len(frame), dtype=float)
    with pytest.raises(ValueError, match="unique"):
        exp.causal_changes(pd.concat([frame, frame.iloc[:1]]), {"W": scores, "A": scores})
    scores[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        exp.causal_changes(frame, {"W": scores, "A": scores})


def test_meta_heads_share_support_mass_and_keep_first_controls():
    train, held, heads = head_fixture()
    meta, _ = exp.enrich(train, held, heads)
    evaluation = meta.copy()
    evaluation.trading_day = "2026-05-11"
    meta.loc[1, "label_complete"] = False
    scored, audits = exp.fit_admission(meta, evaluation)
    assert all(a["train_rows"] == 7 and a["train_episodes"] == 2 and a["weight_sum"] == pytest.approx(2) for a in audits.values())
    assert all(a["fit_days"] == ["2026-05-06"] and a["seed"] == exp.SEED for a in audits.values())
    assert audits["B"]["preprocessing"]["raw_features"] == list(exp.PACKAGES["B"])
    assert audits["C"]["preprocessing"]["raw_features"] == list(exp.PACKAGES["C"])
    assert scored.groupby(exp.KEYS).B0_p_net_5.nunique().eq(1).all()
    assert exp.metrics(scored, exp.ARMS)["B0"]["within_episode_auc"] == .5
    intervals = exp.intervals(scored, draws=10)
    assert intervals["day"]["draws"] == 10


def test_meta_fit_rejects_own_day_or_unregistered_training():
    train, held, heads = head_fixture()
    meta, _ = exp.enrich(train, held, heads)
    with pytest.raises(ValueError, match="preceding"):
        exp.fit_admission(meta, meta)
    future = meta.copy()
    future.trading_day = "2026-05-11"
    meta.trading_day = "2026-05-05"
    with pytest.raises(ValueError, match="preceding"):
        exp.fit_admission(meta, future)
