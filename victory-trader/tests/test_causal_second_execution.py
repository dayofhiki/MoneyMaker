"""Contract tests for temporal ordering and conservative barrier replay."""

import unittest

from victory_trader.causal_second_execution import (
    RiskRule,
    SecondBar,
    replay_long,
    risk_sized_whole_shares,
)
from victory_trader.execution_costs import ExecutionScenario

ZERO = ExecutionScenario("zero", 0, 0)
RULE = RiskRule(stop_pct=3, take_pct=10, trail_retrace_pct=7)
T = 1_800_000_000_000


def bar(second, o, h=None, low=None, c=None, halted=False):
    return SecondBar(
        T + second * 1_000,
        o,
        o if h is None else h,
        o if low is None else low,
        o if c is None else c,
        halted,
    )


class CausalSecondExecutionTests(unittest.TestCase):
    def test_entry_latency_and_expiry(self):
        path = [bar(0, 10), bar(1, 11), bar(2, 12)]
        result = replay_long(path, decision_t=T, rule=RULE, scenario=ZERO, latency_ms=2_000)
        self.assertEqual(result.entry.fill_t, T + 2_000)
        self.assertEqual(result.entry.modeled_price, 12)
        expired = replay_long([bar(7, 10)], decision_t=T, rule=RULE, scenario=ZERO)
        self.assertEqual(expired.status, "entry_unavailable")
        self.assertEqual(expired.reason, "entry_order_expired")

    def test_stop_crosses_within_second_then_rebounds_but_fills_at_next_open(self):
        path = [bar(0, 10), bar(1, 10, h=10.2, low=9.5, c=9.9), bar(2, 8.5)]
        result = replay_long(path, decision_t=T, rule=RULE, scenario=ZERO)
        self.assertEqual(result.status, "closed")
        self.assertEqual(result.exits[0].trigger_t, T + 2_000)
        self.assertEqual(result.exits[0].fill_t, T + 2_000)
        self.assertEqual(result.exits[0].reference_price, 8.5)
        self.assertAlmostEqual(result.net_return_pct, -15)

    def test_exit_latency_waits_for_later_open_after_completed_trigger(self):
        path = [
            bar(0, 10),
            bar(1, 10),
            bar(2, 10),
            bar(3, 10, h=10, low=9.5, c=9.9),
            bar(4, 9.8),
            bar(5, 9.7),
            bar(6, 9),
        ]
        result = replay_long(
            path, decision_t=T, rule=RULE, scenario=ZERO, latency_ms=2_000,
        )
        self.assertEqual(result.entry.fill_t, T + 2_000)
        self.assertEqual(result.exits[0].trigger_t, T + 4_000)
        self.assertEqual(result.exits[0].fill_t, T + 6_000)

    def test_simultaneous_stop_and_take_is_unresolved_order(self):
        result = replay_long(
            [bar(0, 10), bar(1, 10, h=11.2, low=9.5, c=10)],
            decision_t=T, rule=RULE, scenario=ZERO,
        )
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.net_return_pct)
        self.assertEqual(result.exits, ())

    def test_partial_take_and_proportional_trailing_exit(self):
        path = [
            bar(0, 10),
            bar(1, 10.5, h=11.1, low=10.4, c=11),
            bar(2, 11, h=11.1, low=10.9, c=11),
            bar(3, 11.5, h=12, low=11.4, c=12),
            bar(4, 11.6, h=11.6, low=11.1, c=11.3),
            bar(5, 11),
        ]
        result = replay_long(path, decision_t=T, rule=RULE, scenario=ZERO, quantity=10)
        self.assertEqual(result.status, "closed")
        self.assertEqual([fill.reason for fill in result.exits], ["partial_take", "trailing"])
        self.assertEqual([fill.quantity for fill in result.exits], [5, 5])
        self.assertAlmostEqual(result.net_return_pct, 10)

    def test_halt_blocks_stop_fill_until_resume_gap(self):
        path = [
            bar(0, 10),
            bar(1, 9.8, h=9.8, low=9.5, c=9.6),
            bar(2, 0, halted=True),
            bar(3, 0, halted=True),
            bar(4, 7),
        ]
        result = replay_long(path, decision_t=T, rule=RULE, scenario=ZERO)
        self.assertEqual(result.status, "closed")
        self.assertEqual(result.exits[0].fill_t, T + 4_000)
        self.assertAlmostEqual(result.net_return_pct, -30)

    def test_missing_seconds_do_not_create_exit(self):
        result = replay_long([bar(0, 10), bar(4, 10)], decision_t=T, rule=RULE, scenario=ZERO)
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(result.exits, ())

    def test_pending_stop_without_resumed_open_is_unresolved(self):
        result = replay_long([bar(0, 10, low=9)], decision_t=T, rule=RULE, scenario=ZERO)
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(result.reason, "stop_unfilled")
        self.assertIsNone(result.net_return_pct)

    def test_risk_sizing_is_whole_share_and_respects_caps(self):
        shares = risk_sized_whole_shares(
            equity=10_000, cash=10_000, entry_reference=10,
            rule=RULE, scenario=ZERO,
        )
        self.assertEqual(shares, 200)  # 20% capital cap binds before 1% risk cap.
        self.assertEqual(
            risk_sized_whole_shares(
                equity=10_000, cash=10_000, entry_reference=10,
                rule=RULE, scenario=ZERO, max_liquidity_shares=25,
            ), 25,
        )
        narrow = risk_sized_whole_shares(
            equity=10_000, cash=10_000, entry_reference=10,
            rule=RULE, scenario=ZERO, max_allocation_fraction=1,
        )
        wide = risk_sized_whole_shares(
            equity=10_000, cash=10_000, entry_reference=10,
            rule=RiskRule(stop_pct=7, take_pct=10, trail_retrace_pct=7),
            scenario=ZERO, max_allocation_fraction=1,
        )
        self.assertGreater(narrow, wide)


if __name__ == "__main__":
    unittest.main()

