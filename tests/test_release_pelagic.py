"""Pelagic release scope and path checks; no model execution."""
import json
from pathlib import Path
import sys
import unittest

SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / 'src'))
from gray_whale_corridor.release import pelagic_v8 as adapter
from gray_whale_corridor.release.paths import TaskPaths, load_policy


class PelagicEntryTests(unittest.TestCase):
    def test_archived_inputs_only(self):
        task = load_policy()['tasks']['pelagic_support']
        self.assertFalse(task['hold_reasons'])
        self.assertEqual(len(task['inputs']), 171)
        self.assertFalse(any(path.startswith('data/raw/') for path in task['inputs']))
        self.assertFalse(any('phase8u' in path.lower() for path in task['inputs']))
        self.assertEqual(sum('/corridor_sensitivity/lcp/' in p for p in task['inputs']), 108)
        self.assertEqual(sum('/corridor_sensitivity/surfaces/' in p for p in task['inputs']), 18)
        self.assertEqual(sum(p.endswith('.npz') for p in task['inputs']), 2)

    def test_saved_design_and_scope(self):
        scope = json.loads((SOFTWARE / 'config/ncc_v8/scope.json').read_text())['pelagic']
        self.assertEqual(list(adapter.YEARS), scope['observed_years'])
        self.assertEqual(sum(map(len, adapter.DESIGN.values())) * 2 * 4, 72)
        self.assertEqual(scope['primary_beta'], .25)
        self.assertEqual(scope['stored_prediction_draws'], 200)
        self.assertEqual(scope['used_corridor_draws'], 100)
        self.assertEqual(scope['prediction_seed'], 8675309)
        self.assertFalse(scope['temporal_extrapolation'])

    def test_registered_inputs_without_outputs(self):
        context = TaskPaths('pelagic_support', SOFTWARE.parent / 'dataset', 'pelagic_read_only_test')
        self.assertEqual(len(context.snapshot()), 171)
        self.assertFalse(context.task_root.exists())
        self.assertTrue(context.input('early_surface_qa').is_file())
        self.assertTrue(context.input('model_integration_gates').is_file())


if __name__ == '__main__':
    unittest.main()
