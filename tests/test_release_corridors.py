"""Input/entry controls for archived-input corridor reproduction."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / 'src'))
from gray_whale_corridor.release import corridors_v8 as adapter
from gray_whale_corridor.release.paths import TaskPaths, load_policy


class CorridorEntryTests(unittest.TestCase):
    def test_primary_energy_scope_and_no_raw_provider_inputs(self):
        policy = load_policy()
        primary = policy['tasks']['primary_corridors']['aliases']
        self.assertEqual(sum(k.startswith('surface_') for k in primary), 28)
        self.assertEqual(sum(k.startswith('grid_') for k in primary), 28)
        self.assertEqual(sum(k.startswith('paths_') for k in primary), 4)
        self.assertEqual(sum(len(v) for v in adapter.PERIODS.values()), 28)
        for task in ('primary_corridors', 'sensitivity_ice', 'sensitivity_vessel'):
            self.assertFalse(policy['tasks'][task]['hold_reasons'])
            self.assertFalse(any(p.startswith('data/raw/prey/') or 'station_year_values' in p
                                 for p in policy['tasks'][task]['inputs']))
            context = TaskPaths(task, SOFTWARE.parent / 'dataset', 'read_only_contract_test')
            self.assertEqual(len(context.snapshot()), len(context.contract['inputs']))
            self.assertFalse(context.task_root.exists())

    def test_sensitivity_requires_baseline_acceptance(self):
        with patch.object(Path, 'is_file', return_value=False):
            with self.assertRaisesRegex(PermissionError, 'Primary corridor acceptance'):
                adapter.require_primary_acceptance()

    def test_station_processing_is_not_raw_conversion(self):
        policy = load_policy()
        task = policy['tasks']['prey_processing']
        self.assertFalse(task['hold_reasons'])
        self.assertIn('NOT provider-raw-to-station', task['starting_layer'])

    def test_scientific_scope_not_replaced_by_adapter_defaults(self):
        scope = json.loads((SOFTWARE / 'config/ncc_v8/scope.json').read_text())
        self.assertEqual(adapter.VARIABLE, scope['primary']['prey_variable'])
        self.assertEqual(scope['ice']['sensitivity_joint_anchors'], [[.1, .45], [.15, .55], [.2, .65]])
        self.assertEqual(scope['vessel']['alpha_sensitivity'], [.25, .5, .75])


if __name__ == '__main__':
    unittest.main()
