"""Verify exception scope on tiny fixtures, without running any scientific model."""
from pathlib import Path
import sys
import unittest
import subprocess
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gray_whale_corridor.release.idw_acceptance import APPROVED, KEYS, FREQUENCY, check_frequency, check_route_invariants, ROUTE_INVARIANTS


class IDWAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame([dict(zip(KEYS, key), x=1., **{FREQUENCY: pair[1]})
                                      for key, pair in APPROVED.items()])
        self.actual = self.reference.copy()
        self.actual[FREQUENCY] = [pair[0] for pair in APPROVED.values()]

    def test_approved_profile_preserves_both_inputs(self):
        a, b = self.actual.copy(deep=True), self.reference.copy(deep=True)
        result = check_frequency(a, b, pd, np)
        self.assertTrue(result['passed'])
        self.assertTrue(result['known_difference_profile_used'])
        self.assertFalse(result['exact_frequency_match'])
        self.assertIn('differences', json.loads(json.dumps(result)))
        pd.testing.assert_frame_equal(a, self.actual, check_exact=True)
        pd.testing.assert_frame_equal(b, self.reference, check_exact=True)

    def test_exact_profile_needs_no_exception(self):
        self.assertTrue(check_frequency(self.reference, self.reference, pd, np)['exact_frequency_match'])

    def test_wrong_year_rejected(self):
        self.actual['target_year'] = 2012
        self.reference['target_year'] = 2012
        with self.assertRaises(AssertionError):
            check_frequency(self.actual, self.reference, pd, np)

    def test_wrong_magnitude_rejected(self):
        self.actual.loc[0, FREQUENCY] += .000625
        with self.assertRaises(AssertionError):
            check_frequency(self.actual, self.reference, pd, np)

    def test_one_cell_only_rejected(self):
        self.actual.loc[0, FREQUENCY] = self.reference.loc[0, FREQUENCY]
        with self.assertRaises(AssertionError):
            check_frequency(self.actual, self.reference, pd, np)

    def test_extra_cell_rejected(self):
        a = self.actual.iloc[[0]].copy()
        b = self.reference.iloc[[0]].copy()
        a['grid_id'] = b['grid_id'] = '(99, 99)'
        with self.assertRaises(AssertionError):
            check_frequency(pd.concat([self.actual, a]), pd.concat([self.reference, b]), pd, np)

    def test_other_field_change_rejected(self):
        self.actual.loc[0, 'x'] += 1
        with self.assertRaises(AssertionError):
            check_frequency(self.actual, self.reference, pd, np)

    def test_frequent_membership_change_rejected(self):
        self.actual.loc[0, FREQUENCY] = .06
        with self.assertRaises(AssertionError):
            check_frequency(self.actual, self.reference, pd, np)

    def test_duplicate_or_missing_keys_rejected(self):
        for frame in (pd.concat([self.actual, self.actual.iloc[[0]]]), self.actual.iloc[[0]]):
            with self.assertRaises(AssertionError):
                check_frequency(frame, self.reference, pd, np)

    def test_centroid_change_rejected(self):
        row = dict(setting='k8_p1', scenario='prey_only', target_year=2011,
                   **{key: 1. for key in ROUTE_INVARIANTS})
        reference = pd.DataFrame([row])
        actual = reference.copy()
        check_route_invariants(actual, reference, pd)
        actual.loc[0, 'centroid_x'] += 1e-9
        with self.assertRaises(AssertionError):
            check_route_invariants(actual, reference, pd)


def run_isolated_case(name):
    result = subprocess.run([sys.executable, '-B', __file__, 'IDWAcceptanceTests.' + name],
                            capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)


def load_tests(loader, tests, pattern):
    if __name__ == '__main__':
        return tests
    # Numeric fixtures must not pollute the read-only entry's no-scientific-import tests.
    cases = []
    for name in loader.getTestCaseNames(IDWAcceptanceTests):
        def case(test_name=name):
            run_isolated_case(test_name)
        case.__name__ = name
        cases.append(unittest.FunctionTestCase(case, description=name))
    return unittest.TestSuite(cases)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    unittest.main()
