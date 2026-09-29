import numpy as np
import pandas as pd
import pytest

from victory_trader import chronological_action_value as av


def source(prices=(10, 9, 11, 12), minutes=None, day="2026-05-05"):
    minutes = minutes or list(range(1, len(prices) + 1))
    return pd.DataFrame(
        [
            {
                "trading_day": day,
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": minute * 60000,
                "log_current_close": np.log(p),
                "exit_reference_open": p,
                "minutes_to_close": 100 - minute,
            }
            for minute, p in zip(minutes, prices)
        ]
    )


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.repeat(self.value, len(x))


def policy(enter=1.0, wait=0.0, hold=-1.0):
    return av.Policy(
        dict(zip(av.TARGETS, map(Constant, (enter, wait, hold)))),
        ("elapsed_minutes",),
        (),
        {},
    )


def test_future_suffix_cannot_change_feature_prefix():
    a = source()
    b = a.copy()
    b.loc[2:, ["log_current_close", "exit_reference_open"]] = [np.log(999), 999]
    pd.testing.assert_frame_equal(
        av.build_states(a).iloc[:2], av.build_states(b).iloc[:2]
    )


def test_fill_and_oracle_poison_do_not_change_actions_or_features():
    a = source()
    b = a.copy()
    b["exit_reference_open"] = 99999.0
    b["remaining_option_value_pct"] = 999999.0
    x, y = av.build_states(a), av.build_states(b)
    pd.testing.assert_frame_equal(
        x.drop(columns="execution_open"), y.drop(columns="execution_open")
    )
    for xx, yy in zip(av.actions(x, policy()), av.actions(y, policy())):
        np.testing.assert_array_equal(xx, yy)


def test_missing_close_never_uses_future_fill():
    a = source()
    a.loc[0, "log_current_close"] = np.nan
    x = av.build_states(a)
    assert np.isnan(x.loc[0, "log_current_price"])
    assert not x.loc[0, "can_enter"]


def test_targets_follow_frozen_policy_not_best_future_price():
    result = av.rollout(av.build_states(source((10, 9, 100))), None)
    assert result.loc[0, "enter_value"] == pytest.approx(av.net(10, 9))
    assert result.loc[0, "enter_value"] < 0
    assert result.loc[0, "wait_value"] == pytest.approx(av.net(9, 100))


def test_hold_target_excludes_repeated_buy_cost():
    result = av.rollout(av.build_states(source((10, 10))), None)
    assert result.loc[0, "hold_advantage"] == pytest.approx(0)
    assert result.loc[0, "enter_value"] < 0


def test_sparse_events_are_kept_and_clock_deadline_remains():
    states = av.build_states(source((10, 11, 12), [1, 4, 8]))
    assert list(states.event_index) == [1, 2, 3]
    assert list(states.can_enter) == [True, True, False]
    assert states.loc[1, "gap_minutes"] == 3


def test_missing_fill_does_not_remove_state_and_is_unresolved():
    s = source()
    s.loc[0, "exit_reference_open"] = np.nan
    result = av.rollout(av.build_states(s), policy())
    episode = av.episode_results(result).iloc[0]
    assert len(result) == 4
    assert episode.entered and episode.unresolved and np.isnan(episode.net_pct)


def test_last_historical_row_is_not_automatic_exit():
    states = av.build_states(source((10, 11, 12)))
    result = av.episode_results(av.rollout(states, policy(hold=1))).iloc[0]
    assert result.entered and result.unresolved


def test_precommitted_deadline_exits_even_with_hold_signal():
    states = av.build_states(source((10, 11, 12), [1, 2, 29]))
    result = av.episode_results(av.rollout(states, policy(hold=1))).iloc[0]
    assert not result.unresolved
    assert result.exit_t == 29 * 60000
    assert result.net_pct == pytest.approx(av.net(10, 12))


def test_no_trade_is_cash_but_unresolved_trade_poisons_day():
    states = av.build_states(source())
    cash = av.episode_results(av.rollout(states, policy(enter=-1, wait=-1)))
    unresolved = av.episode_results(av.rollout(states, policy(hold=1)))
    assert av.metrics(cash, ["2026-05-05"])["day_balanced_episode_mean_pct"] == 0
    failed = av.metrics(unresolved, ["2026-05-05"])
    assert failed["day_balanced_episode_mean_pct"] is None
    assert failed["unresolved_trades"] == 1


def test_chronological_target_teacher_never_sees_later_dates(monkeypatch):
    states = av.build_states(
        pd.concat([source(day=d) for d in av.FIT_DAYS], ignore_index=True)
    )
    seen = []

    def fake_fit(labeled, **kwargs):
        seen.append(sorted(labeled.trading_day.unique()))
        p = policy()
        p.training_days = tuple(seen[-1])
        return p

    monkeypatch.setattr(av, "fit_policy", fake_fit)
    teacher, student, targets = av.chronological_fit(states)
    assert seen == [av.FIT_DAYS[:3], av.FIT_DAYS[3:]]
    assert max(teacher.training_days) < min(targets.trading_day)
    assert student.training_days == tuple(av.FIT_DAYS[3:])


def test_duplicate_states_rejected():
    s = source()
    with pytest.raises(ValueError, match="duplicate"):
        av.build_states(pd.concat([s, s.iloc[[0]]]))
