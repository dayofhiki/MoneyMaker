import numpy as np
import pandas as pd
import pytest

from victory_trader.feasible_upside_observability import (
    ADVANTAGE, CEILING, build_ceiling_labels, crossfit_diagnostics,
    economics_report, first_states, fit_classifier, validate_source,
)
from victory_trader.frozen_entry_second_hold_exit import KEYS, net
from victory_trader.preentry_context_hold_value import ALL_FEATURES


def fixture(stop=3000, closing=4000000):
    frame = pd.DataFrame({name: [0., 0., 0.] for name in ALL_FEATURES})
    frame = frame.assign(trading_day="2026-05-05", ticker="T", hot_t=0,
                         entry_t=0, entry_price=10., decision_t=[1000, 2000, 3000],
                         risk_reachable_hold=[True, True, False] if np.isfinite(stop) else True,
                         forced_stop_t=stop, exit_now_pct=[net(10, 10), net(10, 11), net(10, 9)])
    bars = pd.DataFrame({"t": [1000, 2000, 3000, 3601000], "o": [10., 11., 9., 20.]})
    return frame, {("2026-05-05", "T", 0): (bars, 0, closing)}


def test_absorbing_stop_prevents_rebound_and_suffix_is_correct():
    source, contexts = fixture()
    labeled, episodes = build_ceiling_labels(source, contexts)
    assert episodes.reachable_max_base.iloc[0] == pytest.approx(net(10, 11))
    assert episodes.whole_path_max_base.iloc[0] == pytest.approx(net(10, 20))
    assert labeled[CEILING].iloc[0] == labeled[CEILING].iloc[1] == pytest.approx(net(10, 11))
    assert labeled[ADVANTAGE].iloc[1] == pytest.approx(0)
    assert np.isnan(labeled[CEILING].iloc[2])
    pd.testing.assert_frame_equal(source[list(ALL_FEATURES)], labeled[list(ALL_FEATURES)])


def test_stop_cap_tie_and_pending_fill_remain_allowed():
    source, contexts = fixture(stop=3600000)
    source.loc[2, "decision_t"] = 3600000
    contexts[("2026-05-05", "T", 0)][0].loc[2, "t"] = 3600500
    labeled, episodes = build_ceiling_labels(source, contexts)
    assert episodes.terminal_submission_t.iloc[0] == 3600000
    assert episodes.terminal_fill_t.iloc[0] == 3600500
    assert labeled[CEILING].iloc[0] == pytest.approx(net(10, 11))


def test_long_pending_stop_rebound_is_fill_not_new_hold_action():
    source, contexts = fixture()
    bars = contexts[("2026-05-05", "T", 0)][0].drop(index=2)
    contexts[("2026-05-05", "T", 0)] = (bars, 0, 4000000)
    labeled, episodes = build_ceiling_labels(source, contexts)
    assert episodes.terminal_fill_t.iloc[0] == 3601000
    assert labeled[CEILING].iloc[0] == pytest.approx(net(10, 20))
    assert np.isnan(labeled[CEILING].iloc[2])


def test_missing_terminal_censors_instead_of_labeling_loser():
    source, contexts = fixture(stop=np.nan)
    bars = contexts[("2026-05-05", "T", 0)][0].iloc[:3].copy()
    contexts[("2026-05-05", "T", 0)] = (bars, 0, 4000000)
    labeled, episodes = build_ceiling_labels(source, contexts)
    assert not episodes.reachable_complete.iloc[0]
    assert labeled[CEILING].isna().all()
    assert np.isnan(episodes.reachable_max_base.iloc[0])
    decisions = source.iloc[:1][KEYS].assign(resolved=False, base_net_return_pct=np.nan, stop_cost_only_breach=False)
    report, pair = economics_report(episodes, decisions)
    assert pair.diagnostic_category.iloc[0] == "censored"
    assert report["oracle_mean_net_pct"] is None


def test_future_mutation_changes_labels_only():
    source, contexts = fixture()
    old, _ = build_ceiling_labels(source, contexts)
    contexts[("2026-05-05", "T", 0)][0].loc[1, "o"] = 12.
    new, _ = build_ceiling_labels(source, contexts)
    assert new[CEILING].iloc[0] > old[CEILING].iloc[0]
    pd.testing.assert_frame_equal(new[list(ALL_FEATURES)], old[list(ALL_FEATURES)])


def test_no_eligible_first_snapshot_is_never_replaced_by_later_state():
    source, _ = fixture()
    source.loc[0, "risk_reachable_hold"] = False
    assert not first_states(source).risk_reachable_hold.any()


def test_single_class_uses_only_training_prevalence():
    source, _ = fixture()
    source[CEILING] = 3.
    fitted, p = fit_classifier(source, 5, "snapshot", 1)
    assert fitted is None and p == 0
    fitted, p = fit_classifier(source, 0, "states", 1)
    assert fitted is None and p == 1


def test_held_day_labels_cannot_change_its_predictions(monkeypatch):
    pieces = []
    for day in ("2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08"):
        f, _ = fixture(stop=np.nan)
        f["trading_day"], f[CEILING], f[ADVANTAGE] = day, 3., 1.
        pieces.append(f)
    source = pd.concat(pieces, ignore_index=True)
    calls = []
    def fit(train, level, kind, seed):
        calls.append(set(train.trading_day))
        return None, float(train[CEILING].ge(level).mean())
    monkeypatch.setattr("victory_trader.feasible_upside_observability.fit_classifier", fit)
    monkeypatch.setattr("victory_trader.feasible_upside_observability.model_fit", lambda f,*args: float(f[ADVANTAGE].mean()))
    monkeypatch.setattr("victory_trader.feasible_upside_observability.weighted_offset_model", lambda f,m: (m, {}))
    monkeypatch.setattr("victory_trader.feasible_upside_observability._predict", lambda f,m: np.full(len(f),m))
    before, snapshots, _ = crossfit_diagnostics(source)
    source.loc[source.trading_day.eq("2026-05-05"), [CEILING, ADVANTAGE]] = 1e12
    after, changed_snapshots, _ = crossfit_diagnostics(source)
    columns = ["p_net_0", "p_net_5", "p_net_10", "p_net_20", "predicted_max_exit_advantage_pct"]
    pd.testing.assert_frame_equal(before.loc[before.trading_day.eq("2026-05-05"), columns], after.loc[after.trading_day.eq("2026-05-05"), columns])
    assert snapshots.p_net_5.iloc[0] == changed_snapshots.p_net_5.iloc[0]
    assert calls[0] == {"2026-05-06", "2026-05-07", "2026-05-08"}


@pytest.mark.parametrize("change", ["future_states", "future_raw", "duplicate"])
def test_invalid_source_rejected(change):
    frames = []
    for i in range(86):
        frame, _ = fixture()
        frame["ticker"] = str(i)
        frame["trading_day"] = ("2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08")[i%4]
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    raw = pd.DataFrame({"trading_day": ["2026-05-05"]})
    if change == "future_states":
        source.loc[0, "trading_day"] = "2026-06-15"
    elif change == "future_raw":
        raw.loc[0, "trading_day"] = "2026-06-15"
    else:
        source = pd.concat([source, source.iloc[:1]])
    with pytest.raises(ValueError):
        validate_source(source, raw)


def test_ceiling_cannot_be_less_than_actual_return():
    source, contexts = fixture()
    _, episodes = build_ceiling_labels(source, contexts)
    decisions = source.iloc[:1][KEYS].assign(resolved=True, base_net_return_pct=50., stop_cost_only_breach=False)
    with pytest.raises(ValueError, match="ceiling below"):
        economics_report(episodes, decisions)
