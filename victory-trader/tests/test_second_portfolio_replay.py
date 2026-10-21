"""Account conservation and causal admission across one-second paths."""

import math

import pytest

from victory_trader.causal_second_execution import RiskRule, SecondBar
from victory_trader.execution_costs import ExecutionScenario
from victory_trader.second_portfolio_replay import EntryAttempt, replay_portfolio

T = 1_800_000_000_000
ZERO = ExecutionScenario("zero", 0, 0)
RULE = RiskRule(stop_pct=3, take_pct=10, trail_retrace_pct=7)


def bar(second, price, *, high=None, low=None, close=None):
    return SecondBar(
        T + second * 1_000,
        price,
        price if high is None else high,
        price if low is None else low,
        price if close is None else close,
    )


def attempt(ticker="A", second=0, close_second=30):
    return EntryAttempt(ticker, T + second * 1_000, 10, T + close_second * 1_000, RULE)


def test_gap_stop_uses_resume_open_and_marks_drawdown():
    paths = {"A": [
        bar(0, 10),
        bar(1, 9.8, high=9.8, low=9.5, close=9.6),
        bar(2, 8),
    ]}
    result = replay_portfolio([attempt()], paths, ZERO, initial_cash=1_000)
    assert result.records[0]["status"] == "closed"
    assert result.summary["ending_cash"] == pytest.approx(960)
    assert result.summary["realized_pnl"] == pytest.approx(-40)
    assert result.summary["observed_mark_drawdown_pct"] >= 4
    assert result.summary["ending_marked_equity"] == pytest.approx(960)


def test_same_ticker_reentry_is_blocked_before_exit():
    paths = {"A": [
        bar(0, 10),
        bar(1, 9.8, low=9.5, close=9.6),
        bar(2, 9),
        bar(3, 10),
    ]}
    result = replay_portfolio(
        [attempt(), attempt(second=1)], paths, ZERO, initial_cash=1_000,
    )
    assert result.records[0]["status"] == "closed"
    assert result.records[1]["status"] == "blocked_open_position"
    assert result.summary["blocked"] == 1


def test_missing_entry_releases_reservation_before_next_decision():
    paths = {"A": [], "B": [
        bar(6, 10), bar(7, 9.8, low=9.5, close=9.6), bar(8, 10),
    ]}
    result = replay_portfolio(
        [attempt("A"), attempt("B", second=6)], paths, ZERO,
        initial_cash=1_000,
    )
    assert result.records[0]["status"] == "entry_unavailable"
    assert result.records[1]["status"] == "closed"
    assert result.summary["ending_cash"] == pytest.approx(1_000)
    assert not result.summary["complete_coverage"]


def test_partial_take_cash_release_uses_whole_shares():
    paths = {"A": [
        bar(0, 10),
        bar(1, 10.5, high=11.1, low=10.4, close=11),
        bar(2, 11, high=11.1, low=10.9, close=11),
        bar(3, 11.5, high=12, low=11.4, close=12),
        bar(4, 11.6, high=11.6, low=11.1, close=11.3),
        bar(5, 11),
    ]}
    result = replay_portfolio([attempt()], paths, ZERO, initial_cash=1_000)
    assert result.records[0]["shares"] == 20
    assert result.records[0]["status"] == "closed"
    assert result.records[0]["net_return_pct"] == pytest.approx(10)
    assert result.summary["ending_cash"] == pytest.approx(1_020)
    assert all(math.isfinite(float(row["marked_equity"])) for row in result.equity_curve)


def test_session_close_cannot_fill_at_next_day_reference():
    paths = {"A": [
        bar(0, 10), bar(1, 9.8, low=9.5, close=9.6), bar(4, 7),
    ]}
    result = replay_portfolio(
        [attempt(close_second=3)], paths, ZERO, initial_cash=1_000,
    )
    assert result.records[0]["status"] == "open_unresolved"
    assert result.summary["unresolved"] == 1
    assert result.summary["ending_cash"] == pytest.approx(800)
    assert result.summary["ending_marked_equity"] == pytest.approx(992)


def test_entry_gap_does_not_resize_order_after_observing_open():
    paths = {"A": [bar(0, 20)], "B": [bar(0, 10)]}
    result = replay_portfolio(
        [attempt("A"), attempt("B")], paths, ZERO,
        initial_cash=1_000, max_positions=2,
        risk_fraction=0.5, max_allocation_fraction=0.5,
    )
    assert result.records[0]["status"] == "entry_open_unaffordable"
    assert result.records[1]["status"] == "open_unresolved"
    assert result.records[1]["shares"] == 50
    assert result.summary["ending_cash"] == pytest.approx(500)


def test_ambiguous_same_second_remains_open_and_blocks_profit_claim():
    paths = {"A": [bar(0, 10), bar(1, 10, high=11.2, low=9.5)]}
    result = replay_portfolio([attempt()], paths, ZERO, initial_cash=1_000)
    assert result.records[0]["status"] == "ambiguous_open_unresolved"
    assert result.summary["unresolved"] == 1
    assert not result.summary["complete_coverage"]


def test_future_exit_path_cannot_change_simultaneous_admission():
    base = {"A": [bar(0, 10), bar(1, 10)], "B": [bar(0, 10)]}
    for future in (bar(2, 20), bar(2, 5)):
        paths = {**base, "A": [*base["A"], future]}
        result = replay_portfolio(
            [attempt("A"), attempt("B")], paths, ZERO,
            initial_cash=1_000, max_positions=1,
        )
        assert result.records[0]["entry_t"] == T
        assert result.records[1]["status"] == "blocked_capacity"

