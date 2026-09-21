import unittest

import pandas as pd

from victory_trader.execution_costs import ExecutionScenario
from victory_trader.position_ledger import MINUTE, replay

ZERO = ExecutionScenario('zero', 0, 0)
T = 1_800_000_000_000


def attempts(times=(T,), tickers=('A',), prices=(10,)):
    return pd.DataFrame({'ticker': tickers, 'decision_t': times, 'entry_price': prices})


def bars(times, prices, tickers=None):
    return pd.DataFrame({'ticker': tickers or ['A'] * len(times), 't': times,
                         'o': prices, 'c': prices})


class LedgerTests(unittest.TestCase):
    def test_exact_exit_and_cash_conservation(self):
        r, s, _ = replay(attempts(), bars([T + 10 * MINUTE], [11]), ZERO)
        self.assertAlmostEqual(s['ending_cash'], 10100)
        self.assertEqual(s['closed'], 1)
        self.assertEqual(r.iloc[0].exit_t, T + 11 * MINUTE)

    def test_gap_exits_at_later_open_never_stale_close(self):
        r, s, _ = replay(attempts(), bars([T + 9 * MINUTE, T + 20 * MINUTE], [12, 8]), ZERO)
        self.assertAlmostEqual(s['realized_pnl'], -200)
        self.assertEqual(s['delayed_exits'], 1)
        self.assertEqual(r.iloc[0].exit_t, T + 20 * MINUTE)

    def test_unresolved_capital_blocks_later_entry_across_days(self):
        a = attempts((T, T + 1440 * MINUTE), ('A', 'B'), (10, 10))
        r, s, _ = replay(a, bars([T + 9 * MINUTE], [8]), ZERO, initial_cash=1000)
        self.assertEqual(s['blocked_cash'], 1)
        self.assertEqual(s['committed_capital'], 1000)
        self.assertEqual(s['ending_cash'], 0)
        self.assertEqual(s['realized_pnl'], 0)
        self.assertAlmostEqual(s['ending_stale_marked_equity'], 800)
        self.assertFalse(s['complete_accounting'])

    def test_same_ticker_blocked_until_exit(self):
        a = attempts((T, T + 5 * MINUTE), ('A', 'A'), (10, 10))
        _, s, _ = replay(a, bars([T + 10 * MINUTE], [11]), ZERO)
        self.assertEqual(s['blocked_position'], 1)

    def test_missing_entry_is_explicit_assumption(self):
        _, s, _ = replay(attempts(prices=(float('nan'),)), bars([], []), ZERO)
        self.assertEqual(s['missing_entry_references'], 1)
        self.assertEqual(s['ending_cash'], 10000)
        self.assertFalse(s['complete_accounting'])

    def test_future_price_does_not_change_acceptance(self):
        a = attempts((T, T), ('A', 'B'), (10, 10))
        for prices in ([100, 1], [1, 100]):
            _, s, _ = replay(a, bars([T + 10 * MINUTE] * 2, prices, ['A', 'B']),
                             ZERO, initial_cash=1000)
            self.assertEqual(s['accepted'], 1)
            self.assertEqual(s['blocked_cash'], 1)

    def test_friction_reduces_cash(self):
        _, s, _ = replay(attempts(), bars([T + 10 * MINUTE], [10]),
                         ExecutionScenario('base', 25, 25, 1))
        self.assertLess(s['ending_cash'], 10000)

    def test_duplicate_bars_rejected(self):
        with self.assertRaises(ValueError):
            replay(attempts(), bars([T, T], [10, 10]), ZERO)

    def test_attempt_horizon_controls_scheduled_exit(self):
        a = attempts()
        a['action_horizon_min'] = 5
        r, s, _ = replay(a, bars([T + 5 * MINUTE], [11]), ZERO)
        self.assertEqual(s['closed'], 1)
        self.assertEqual(
            r.iloc[0].scheduled_exit_bar_t,
            T + 5 * MINUTE,
        )


if __name__ == '__main__':
    unittest.main()
