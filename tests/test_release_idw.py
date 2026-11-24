"""Read-only contract tests for the archived-checkpoint IDW entry."""
import json
from pathlib import Path
import sys
import unittest

SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / 'src'))
from gray_whale_corridor.release import idw_v8
from gray_whale_corridor.release.paths import TaskPaths, load_policy


class IDWEntryTests(unittest.TestCase):
    def test_fixed_v8_settings(self):
        scope = json.loads((SOFTWARE / 'config/ncc_v8/scope.json').read_text())
        actual = {(v['neighbors'], v['power']) for v in idw_v8.SETTINGS.values()}
        self.assertEqual(actual, {(v['k'], v['p']) for v in scope['robustness']['IDW']})
        self.assertEqual(actual, {(8, 2), (8, 1), (8, 3), (4, 2)})

    def test_only_archived_inputs(self):
        task = load_policy()['tasks']['sensitivity_idw']
        self.assertFalse(task['hold_reasons'])
        self.assertIn('two-cell frequency profile', task['acceptance_rule'])
        self.assertEqual(sum(k.startswith('prey_') for k in task['aliases']), 9)
        self.assertEqual(sum(k.startswith('ice_') for k in task['aliases']), 9)
        self.assertEqual(sum(k.startswith('grid_') for k in task['aliases']), 18)
        self.assertFalse(any('data/raw/prey' in p or 'station_year_values' in p for p in task['inputs']))
        context = TaskPaths('sensitivity_idw', SOFTWARE.parent / 'dataset', 'read_only_idw_test')
        self.assertEqual(len(context.snapshot()), len(task['inputs']))
        self.assertFalse(context.task_root.exists())

    def test_benthic_source_pending_is_separate_from_execution(self):
        task = load_policy()['tasks']['prey_processing']
        self.assertFalse(task['hold_reasons'])
        self.assertIn('pending', task['source_access_status'])
        self.assertIn('archived model-ready surfaces are included', task['source_access_status'])
        self.assertFalse(any('doi' in name.lower() for name in task['aliases']))


if __name__ == '__main__':
    unittest.main()
