import numpy as np
import pandas as pd
import pytest

from victory_trader.execution_costs import DEFAULT_EXECUTION_SCENARIOS
from victory_trader.feasible_upside_observability import CEILING
from victory_trader.frozen_entry_second_hold_exit import KEYS, net
from victory_trader.hierarchical_crack_entry_controller import EVENT_FEATURES
from victory_trader.local_turn_entry import TURN_FEATURES
from victory_trader.preentry_momentum_observability import (
    CLOCK_FEATURES, STATIC_FEATURES, baseline_columns, clock_features,
    collect_raw, crossfit, entry_labels, snapshots, vector_net,
)


def bars():
    t = np.arange(0, 300000, 1000)
    c = 10+np.arange(len(t))*.001
    return pd.DataFrame({"t": t, "o": c, "h": c, "l": c, "c": c, "v": 10., "n": 2.})


def features():
    f = pd.DataFrame({name: [0., 0.] for name in (*STATIC_FEATURES, *EVENT_FEATURES, *TURN_FEATURES)})
    return f.assign(trading_day="2026-05-05", ticker="T", hot_t=100000, decision_t=[121000, 151000])


def test_completed_seconds_only_and_future_mutation_invariance():
    f, b = features(), bars()
    before = clock_features(f, {("2026-05-05", "T"): (b, 0, 400000)})
    changed = b.copy()
    changed.loc[changed.t.ge(121000), ["o", "h", "l", "c", "v", "n"]] = 1e6
    after = clock_features(f, {("2026-05-05", "T"): (changed, 0, 400000)})
    pd.testing.assert_series_equal(before.loc[0, list(CLOCK_FEATURES)], after.loc[0, list(CLOCK_FEATURES)])
    assert before.momentum_return_120s.iloc[0] > 0
    assert f.hot_t.iloc[0] > 1000  # Long-window history is pre-HOT too.


def test_sparse_activity_does_not_fake_ten_observations():
    f, b = features().iloc[:1], bars()
    b = b.loc[b.t.isin([0, 90000, 115000, 120000])]
    out = clock_features(f, {("2026-05-05", "T"): (b, 0, 400000)})
    assert out.momentum_activity_fraction_10s.iloc[0] == pytest.approx(.2)
    assert out.momentum_gap_s.iloc[0] == 0
    assert np.isnan(out.momentum_volume_rate_ratio_10s.iloc[0])


def test_liquidity_rates_and_direction_are_causal_proxies():
    f, b = features().iloc[:1], bars()
    b.loc[b.t.ge(111000) & b.t.le(120000), "v"] = 30.
    out = clock_features(f, {("2026-05-05", "T"): (b, 0, 400000)})
    assert out.momentum_volume_rate_ratio_10s.iloc[0] == pytest.approx(3)
    assert out.momentum_signed_volume_proxy_10s.iloc[0] == pytest.approx(1)
    assert out.momentum_efficiency_10s.iloc[0] == pytest.approx(1)


def test_cost_estimate_uses_last_completed_close_not_future_fill():
    f, b = features().iloc[:1], bars()
    b.loc[b.t.le(120000), ["o", "h", "l", "c"]] = .2
    b.loc[b.t.ge(121000), ["o", "h", "l", "c"]] = 10.
    out = clock_features(f, {("2026-05-05", "T"): (b, 0, 400000)})
    assert out.known_cost_drag_pct.iloc[0] == pytest.approx(-net(.2, .2))
    assert out.known_cost_drag_pct.iloc[0] > 2.5


@pytest.mark.parametrize("scenario", DEFAULT_EXECUTION_SCENARIOS)
def test_vector_costs_match_shared_execution_rules(scenario):
    for entry in (.1, 1., 10.):
        prices = np.array([entry*.9, entry, entry*1.1])
        assert vector_net(entry, prices, scenario) == pytest.approx([net(entry, p, scenario) for p in prices])


def label_bars():
    return pd.DataFrame({"t": [1000, 2000, 3000, 5000, 7000],
                         "o": [10., 10.1, 9., 20., 20.],
                         "c": [10., 9., 9., 20., 20.]})


def test_stop_is_absorbing_and_later_rebound_never_becomes_reachable():
    labels = entry_labels(label_bars(), 1000, 6000)
    assert labels[CEILING] == pytest.approx(net(10, 10.1))
    assert labels["label_stop_t"] == 3000
    assert labels["label_whole_max_net_pct"] > 90


def test_stop_cap_tie_and_pending_rebound_fill():
    b = label_bars().drop(index=2)
    labels = entry_labels(b, 1000, 3000)
    assert labels["label_stop_t"] == 3000
    assert labels[CEILING] == pytest.approx(net(10, 20))


def test_missing_terminal_is_censored_not_a_negative():
    labels = entry_labels(label_bars().iloc[:3], 1000, 6000)
    # This path stops at3000 and has a fill; use no-stop closes to test cap.
    b = label_bars().iloc[:3].copy()
    b["c"] = 10.
    labels = entry_labels(b, 1000, 6000)
    assert not labels["label_complete"]
    assert np.isnan(labels[CEILING])


def test_future_and_learned_columns_cannot_enter_allowlist():
    f = features().assign(predicted_enter_utility_pct=100., future_return_15s_pct=10., oracle_best_base_pct=20., execution_price=30.)
    selected = baseline_columns(f)
    assert not set(selected) & {"predicted_enter_utility_pct", "future_return_15s_pct", "oracle_best_base_pct", "execution_price"}
    with pytest.raises(ValueError, match="forbidden"):
        baseline_columns(f.assign(current_predicted_target=1.))


def test_snapshot_is_fixed_clock_recheck_and_never_best_future_state():
    f = features().iloc[:1].copy()
    f = pd.concat([f.assign(decision_t=t) for t in (1000, 10000, 35000, 60000)], ignore_index=True)
    f[CEILING] = [1., 100., -100., 5.]
    out = snapshots(f, 20)
    assert out.decision_t.iloc[0] == 35000
    assert out[CEILING].iloc[0] == -100


def test_missing_raw_retained_as_missing_and_future_raw_refused():
    f = features()
    b = bars().assign(trading_day="2026-05-05", ticker="OTHER")
    contexts, audit, _ = collect_raw(f, b, fetch_missing=False)
    assert contexts == {}
    assert audit["uncovered_pairs"] == [["2026-05-05", "T"]]
    with pytest.raises(ValueError, match="dates forbidden"):
        collect_raw(f, b.assign(trading_day="2026-06-15"), fetch_missing=False)


def test_outer_day_future_labels_do_not_change_its_fitted_scores(monkeypatch):
    f = features()
    frame = pd.concat([f.assign(trading_day=day, **{CEILING: 3., "label_complete": True, "entry_utility_fixed_pct": 1.})
                       for day in ("2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08")], ignore_index=True)
    for name in CLOCK_FEATURES:
        frame[name] = 0.
    calls = []
    def fit(train, features, level, seed):
        calls.append(set(train.trading_day))
        return None, float(train[CEILING].ge(level).mean())
    monkeypatch.setattr("victory_trader.preentry_momentum_observability.fit_probability", fit)
    monkeypatch.setattr("victory_trader.preentry_momentum_observability._fit", lambda f,*args,**kwargs: float(f.entry_utility_fixed_pct.mean()))
    monkeypatch.setattr("victory_trader.preentry_momentum_observability._predict", lambda f,m: np.full(len(f),m))
    before, _ = crossfit(frame, STATIC_FEATURES)
    frame.loc[frame.trading_day.eq("2026-05-05"), [CEILING, "entry_utility_fixed_pct"]] = 1e12
    after, _ = crossfit(frame, STATIC_FEATURES)
    columns = ["legacy_utility_score", "B_p_net_5", "C_p_net_5"]
    pd.testing.assert_frame_equal(before.loc[before.trading_day.eq("2026-05-05"), columns], after.loc[after.trading_day.eq("2026-05-05"), columns])
    assert calls[0] == {"2026-05-06", "2026-05-07", "2026-05-08"}
    assert len(before[KEYS].drop_duplicates()) == 4
