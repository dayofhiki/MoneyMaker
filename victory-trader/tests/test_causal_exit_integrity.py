import pandas as pd
import pytest

from victory_trader.pullback_trailing_exit import apply_exit_rule


def replay(path, hold=10):
    episodes = pd.DataFrame([dict(
        trading_day='2026-05-05', ticker='TEST', hot_t=0,
        entered=True, entry_t=60_000, entry_price=10.0, path=path,
    )])
    return apply_exit_rule(
        episodes, stop_loss_pct=-4, trail_pct=3,
        max_hold_minutes=hold, policy_name='audit',
    ).iloc[0]


def test_truncated_path_cannot_sell_at_last_known_price():
    row = replay([(120_000, 10.1), (180_000, 10.2)])
    assert not row.resolved
    assert pd.isna(row.economic_return_pct)
    assert row.exit_reason == 'missing_deadline_state'


def test_later_price_cannot_fill_missing_deadline():
    row = replay([(120_000, 10.1), (720_000, 10.2)])
    assert not row.resolved
    assert pd.isna(row.exit_t)


@pytest.mark.parametrize('hold,deadline,reason', [
    (10, 660_000, 'max_hold'), (30, 1_800_000, 'terminal_cap'),
])
def test_exact_known_deadline_resolves(hold, deadline, reason):
    row = replay([(120_000, 10.1), (deadline, 10.2)], hold)
    assert row.resolved
    assert row.exit_t == deadline
    assert row.exit_reason == reason


def test_causal_stop_does_not_depend_on_future_path():
    prefix = [(120_000, 10.5), (180_000, 10.0)]
    left = replay(prefix)
    right = replay(prefix + [(240_000, 12.0), (660_000, 13.0)])
    assert left.resolved and right.resolved
    assert left.exit_t == right.exit_t == 180_000
    assert left.trade_return_pct == right.trade_return_pct
