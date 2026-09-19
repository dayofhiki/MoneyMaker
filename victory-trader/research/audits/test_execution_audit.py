import unittest
from execution_audit import audit


def fixture():
    details = '\n'.join(f'{m} raw_same_fit_cap1 8 0.5 -0.5 0.2' for m in ('a', 'b', 'c'))
    paths = '\n'.join(f'{m} raw_same_fit_cap1 10 2' for m in ('a', 'b', 'c'))
    return ('=== details ===\nmonth policy trades gross_mean_pct base_mean_pct decision_score_mean\n'
            + details + '\n=== paths ===\nmonth policy buy_attempts unevaluated_attempts\n' + paths)


class AuditTests(unittest.TestCase):
    def test_arithmetic(self):
        row = audit(fixture())[0]
        self.assertEqual(row['modeled_friction_pp'], 1)
        self.assertEqual(row['unevaluated_fraction'], 0.2)
        self.assertEqual(row['selected_prediction_error_pp'], 0.7)
        self.assertEqual(row['missing_mean_needed_for_attempt_mean_zero_pct'], 2)

    def test_reject_count_mismatch(self):
        with self.assertRaises(ValueError):
            audit(fixture().replace('10 2', '10 1'))

    def test_reject_incomplete_months(self):
        with self.assertRaises(ValueError):
            audit(fixture().replace('c raw_same_fit_cap1 8 0.5 -0.5 0.2', ''))


if __name__ == '__main__':
    unittest.main()
