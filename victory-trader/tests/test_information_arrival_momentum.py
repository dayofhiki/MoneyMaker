import json

import numpy as np
import pandas as pd
import pytest

from victory_trader.feasible_upside_observability import CEILING
from victory_trader.frozen_momentum_may_transport import EVAL_DAYS
from victory_trader.information_arrival_momentum import evidence_observation, later_states, paired_report
from victory_trader.pending_exit_integrity import session_limits


def bars(seconds=(0, 60, 75, 81)):
    opening, closing = session_limits(EVAL_DAYS[0])
    raw = pd.DataFrame({"t": opening+np.array(seconds)*1000, "o": 5., "c": 5., "h": 5., "l": 5., "v": 100., "n": 1.})
    return raw, opening, closing


def test_selection_is_timestamp_only_and_window_boundary_is_explicit():
    raw, opening, closing = bars()
    assert evidence_observation(raw, opening+60000, closing) == opening+82000
    raw[["o", "c", "h", "l", "v", "n"]] = 99999.
    assert evidence_observation(raw, opening+60000, closing) == opening+82000
    raw, opening, closing = bars((60, 70))
    assert evidence_observation(raw, opening+60000, closing) is None
    raw, opening, closing = bars((60, 69))
    assert evidence_observation(raw, opening+60000, closing) == opening+70000


def test_evidence_can_be_first_and_never_requires_future_three_prints():
    raw, opening, closing = bars((55, 60))
    assert evidence_observation(raw, opening+60000, closing) == opening+61000
    raw, opening, closing = bars((60, 61, 360))
    assert evidence_observation(raw.iloc[:2], opening+60000, closing) == opening+62000
    raw, opening, closing = bars((60, 359, 360))
    assert evidence_observation(raw, opening+60000, closing) is None
    assert evidence_observation(raw.iloc[:0], opening+60000, closing) is None


def test_label_incompleteness_does_not_skip_first_evidence_state():
    raw, opening, closing = bars((60, 61))
    cohort = pd.DataFrame([{"trading_day": d, "ticker": "X", "hot_t": session_limits(d)[0]+60000} for d in EVAL_DAYS])
    frame = later_states(cohort, {(EVAL_DAYS[0], "X"): (raw, opening, closing)})
    assert frame.decision_t.iloc[0] == opening+62000
    assert frame.observation_available.iloc[0] and not frame.label_complete.iloc[0]
    assert frame.missing_reason.iloc[0] == "incomplete_entry_or_terminal_fill_label"
    assert not frame.observation_available.iloc[1:].any()
    with pytest.raises(ValueError, match="fixed original May cohort"):
        later_states(cohort.assign(trading_day="2026-06-15"), {})


def scored():
    return pd.DataFrame({"trading_day": [EVAL_DAYS[0]]*4, "ticker": list("ABCD"), "hot_t": 0,
                         "decision_t": 1000., "observation_available": True, "label_complete": True,
                         "missing_reason": None, CEILING: [6., -1., -1., -1.],
                         "M_p_net_5": [.8, .2, .1, .1], "M_prior_net_5": .1})


def test_paired_target_transition_and_lost_unavailable_winner_are_reported():
    original = scored()
    evidence = original.copy()
    evidence["decision_t"] = 2000.
    evidence.loc[0, ["observation_available", "label_complete", "decision_t", CEILING, "M_p_net_5"]] = [False, False, np.nan, np.nan, np.nan]
    evidence.loc[1, CEILING] = 7.
    result = paired_report(original, evidence, draws=5)
    assert result["paired_complete"] == 3
    assert result["original_positive_total"] == 1
    assert result["original_positive_no_later_observation"] == 1
    assert result["paired_positive_table"]["original_0_evidence_1"] == 1
    assert result["paired_arms"]["original"]["auc"] is None
    assert result["paired_intervals"]["ticker_day"]["auc_ci95"] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(AssertionError):
        paired_report(original, evidence.iloc[::-1], draws=5)


def test_same_clock_has_identical_metrics_and_no_information_gain():
    original = scored()
    result = paired_report(original, original, draws=5)
    assert result["same_clock_observed"] == 4 and result["later_clock_observed"] == 0
    assert result["paired_arms"]["original"] == result["paired_arms"]["evidence"]
    assert result["paired_intervals"]["ticker_day"]["brier_ci95"] == [0., 0.]
