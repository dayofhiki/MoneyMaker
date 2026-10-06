import json

import numpy as np
import pandas as pd
import pytest

from victory_trader import gross_net_factorization as experiment
from victory_trader.compact_momentum_representation import independent_weights
from victory_trader.feasible_upside_observability import CEILING
from victory_trader.frozen_momentum_may_transport import EVAL_DAYS
from victory_trader.pending_exit_integrity import session_limits
from victory_trader.preentry_momentum_observability import clock_features, entry_labels


def bars(opens=(10., 10., 9., 12., 12.), closes=(10., 9., 12., 12., 12.)):
    return pd.DataFrame({"t": np.array([0, 1000, 2000, 3000, 5000]), "o": opens, "c": closes,
                         "h": np.maximum(opens, closes), "l": np.minimum(opens, closes), "v": 100., "n": 1.})


def test_stop_and_cost_drop_are_nested_on_same_pending_fill_schedule():
    raw = bars()
    stopped = entry_labels(raw, 1000, 4000)
    stages = experiment.stage_labels(raw, 1000, 4000)
    # Terminal submission at4s remains pending and fills at5s, not prior close.
    assert stages["stage_whole_complete"]
    assert stages["stage_fill_gross_pct"] == pytest.approx(20.)
    assert stages["stage_whole_net_pct"] > 5
    assert stopped["label_stop_t"] == 2000 and stopped[CEILING] < 5
    assert stages["stage_whole_net_pct"] == stopped["label_whole_max_net_pct"]
    raw = bars(opens=(10., 10., 10.51, 10.51, 10.51), closes=(10.,)*5)
    stages = experiment.stage_labels(raw, 1000, 4000)
    assert stages["stage_fill_gross_pct"] >= 5 and stages["stage_whole_net_pct"] < 5


def test_repricing_can_gain_or_lose_without_using_observed_close_as_entry():
    expensive = bars(opens=(10., 11., 11., 11., 11.), closes=(10., 11., 11., 11., 11.))
    cheap = bars(opens=(10., 9., 10., 10., 10.), closes=(10., 9., 10., 10., 10.))
    lost = experiment.stage_labels(expensive, 1000, 4000)
    gained = experiment.stage_labels(cheap, 1000, 4000)
    assert lost["stage_reference_gross_pct"] >= 5 and lost["stage_fill_gross_pct"] < 5
    assert gained["stage_reference_gross_pct"] < 5 and gained["stage_fill_gross_pct"] >= 5


def test_final_complete_whole_censored_and_unobserved_rows_are_retained():
    day = EVAL_DAYS[0]
    opening, closing = session_limits(day)
    raw = bars().iloc[:4].copy()
    raw["t"] += opening
    state = {"trading_day": day, "ticker": "X", "hot_t": opening,
             "decision_t": opening+1000, "observation_available": True,
             **entry_labels(raw, opening+1000, opening+3600000)}
    missing = state | {"ticker": "Z", "decision_t": np.nan, "observation_available": False,
                       "label_complete": False, CEILING: np.nan, "label_whole_max_net_pct": np.nan}
    result = experiment.attach_stages(pd.DataFrame([state, missing]), {(day, "X"): (raw, opening, closing)})
    assert len(result) == 2 and result.label_complete.iloc[0]
    assert not result.stage_whole_complete.any()
    assert not experiment.common_mask(result).any()
    assert not experiment.stage_labels(raw.iloc[:0], opening, closing)["stage_whole_complete"]


def training():
    # Several seconds per episode intentionally test conditional weight preservation.
    return pd.DataFrame({"trading_day": ["2026-05-05"]*4+["2026-05-06"]*2,
                         "ticker": ["X", "X", "X", "Y", "Z", "W"], "hot_t": 0,
                         "x": [0., 1., 2., 3., 4., 5.], "label_complete": True,
                         "stage_whole_complete": True, "stage_reference_gross_pct": [6., 6., 6., 0., 6., 0.],
                         "stage_fill_gross_pct": [6., 0., 6., 0., 6., 0.],
                         "stage_whole_net_pct": [6., 0., 6., 0., 0., 0.], CEILING: [6., 0., 6., 0., 0., 0.]})


def test_conditional_fit_preserves_actual_original_weights(monkeypatch):
    source = training()
    gross = source.stage_fill_gross_pct.ge(5).to_numpy()
    expected = independent_weights(source)[gross]
    real = experiment.LogisticRegression
    recorded = []

    class RecordingLogistic:
        def __init__(self, **kwargs):
            self.inner = real(**kwargs)

        def fit(self, x, y, sample_weight):
            recorded.append(sample_weight.copy())
            self.inner.fit(x, y, sample_weight=sample_weight)
            self.n_iter_, self.intercept_, self.coef_ = self.inner.n_iter_, self.inner.intercept_, self.inner.coef_
            return self

    monkeypatch.setattr(experiment, "LogisticRegression", RecordingLogistic)
    _, _, prior, audit = experiment.fit_conditional(source, ("x",))
    np.testing.assert_array_equal(recorded[0], expected)
    assert audit["weight_sum"] == pytest.approx(expected.sum())
    assert prior == pytest.approx(np.average(source.loc[gross, CEILING].ge(5), weights=expected))
    assert not np.allclose(expected, independent_weights(source.loc[gross]))


def test_product_scoring_ignores_evaluation_outcomes_and_preserves_prior_identity(monkeypatch):
    monkeypatch.setattr(experiment, "MOMENTUM", ("x",))
    source = training()
    held = source.assign(observation_available=True).copy()
    held.loc[5, "observation_available"] = False
    first, audit = experiment.score_heads(source, held)
    poisoned = held.copy()
    poisoned[[CEILING, *experiment.STAGES]] = 99999.
    second, _ = experiment.score_heads(source, poisoned)
    fields = [f"{a}_p_net_5" for a in (*experiment.ARMS, "G", "H")]
    pd.testing.assert_frame_equal(first[fields], second[fields])
    assert first.loc[5, fields].isna().all()
    np.testing.assert_allclose(first.P_p_net_5, first.G_p_net_5*first.H_p_net_5, equal_nan=True)
    assert audit["prior_product_error"] < 1e-12
    assert first.P_prior_net_5.iloc[0] == pytest.approx(first.D_prior_net_5.iloc[0])


def test_empty_and_single_class_conditional_support_have_explicit_fallback():
    source = training()
    source["stage_fill_gross_pct"] = 0.
    source[CEILING] = 0.
    model, transform, prior, audit = experiment.fit_conditional(source, ("x",))
    assert model is transform is None and prior == 0 and audit["empty_support_fallback"]
    source["stage_fill_gross_pct"] = 6.
    model, transform, prior, audit = experiment.fit_conditional(source, ("x",))
    assert model is None and transform is not None and prior == 0 and audit["single_class_fallback"]


def test_causal_inputs_unchanged_by_future_price_and_empty_coverage_fails(monkeypatch):
    day = EVAL_DAYS[0]
    opening, closing = session_limits(day)
    raw = bars()
    raw["t"] += opening
    source = pd.DataFrame([{"trading_day": day, "ticker": "X", "hot_t": opening,
                            "decision_t": opening+1000, "previous_close": 10.}])
    first = clock_features(source, {(day, "X"): (raw, opening, closing)})
    changed = raw.copy()
    changed.loc[changed.t.ge(opening+1000), ["o", "c", "h", "l", "v", "n"]] = 999.
    second = clock_features(source, {(day, "X"): (changed, opening, closing)})
    pd.testing.assert_frame_equal(first, second)
    monkeypatch.setattr(experiment, "MOMENTUM", ("x",))
    train = training()
    held = train.assign(observation_available=False, label_complete=False, missing_reason="absent")
    scored, _ = experiment.score_heads(train, held)
    scored[CEILING] = np.nan
    result = experiment.report(scored, draws=5)
    assert result["sampled_episodes"] == 6 and result["labeled_episodes"] == 0
    assert not experiment.gate(result)["complete_coverage_at_least90pct"]
    assert not experiment.gate(result)["ticker_day_P_minus_D_within_day_gain_ci_positive"]
    json.dumps(result, allow_nan=False)


def test_stage_counts_expose_nonadditive_repricing_and_separate_cost_stop():
    source = training().iloc[:4].copy()
    source["stage_reference_gross_pct"] = [6., 0., 6., 0.]
    source["stage_fill_gross_pct"] = [0., 6., 6., 6.]
    source["stage_whole_net_pct"] = [0., 0., 6., 6.]
    source[CEILING] = [0., 0., 0., 6.]
    result = experiment.stage_report(source)
    assert result["repricing_lost_rows"] == 1 and result["repricing_gained_rows"] == 2
    assert result["cost_lost_rows"] == 1 and result["stop_lost_rows"] == 1
