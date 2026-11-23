"""Late-season release boundary tests; no scientific run or output creation."""
import json
from pathlib import Path
import sys
import unittest

SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / 'src'))
from gray_whale_corridor.release import late_season_v8 as adapter
from gray_whale_corridor.release.paths import TaskPaths, load_policy


class LateSeasonEntryTests(unittest.TestCase):
    def test_archived_environment_and_exact_reference_inventory(self):
        task = load_policy()['tasks']['late_season']
        self.assertFalse(task['hold_reasons'])
        aliases = task['aliases']
        for kind in ('utility', 'corridor', 'paths'):
            self.assertEqual(sum(key.startswith('reference_' + kind + '_') for key in aliases), 28)
        self.assertEqual(sum(key.startswith('ice_') for key in aliases), 9)
        self.assertEqual(sum(key.startswith('vessel_') for key in aliases), 5)
        self.assertFalse(any(path.startswith(('data/raw/prey/', 'data/raw/ice/', 'data/raw/ship/')) for path in task['inputs']))
        self.assertFalse(any('phase8u' in path.lower() for path in task['inputs']))

    def test_original_asamm_csv_export_order(self):
        aliases = load_policy()['tasks']['late_season']['aliases']
        self.assertEqual(len(adapter.ASAMM_GROUPS), 4)
        paths = [aliases['asamm_' + group] for group in adapter.ASAMM_GROUPS]
        self.assertEqual(paths, sorted(paths))
        self.assertTrue(all('/_Output__' in path for path in paths))
        self.assertEqual(len(adapter.VALIDATION_KEYS), 6)
        self.assertEqual(adapter.MAPPING_DISTANCE_ATOL_KM, 1e-10)

    def test_registered_inputs_readable_without_outputs(self):
        context = TaskPaths('late_season', SOFTWARE.parent / 'dataset', 'late_read_only_test')
        self.assertEqual(len(context.snapshot()), len(context.contract['inputs']))
        self.assertFalse(context.task_root.exists())
        scope = json.loads((SOFTWARE / 'config/ncc_v8/scope.json').read_text())
        config = json.loads(context.input('reference_config').read_text())
        self.assertEqual(config['months'], scope['late_months'])
        self.assertEqual(config['vessel_primary_log_q99'], scope['vessel']['fixed_log_q99'])
        self.assertNotEqual(config['vessel_primary_log_q99'], config['vessel_sensitivity_log_q99'])


if __name__ == '__main__':
    unittest.main()
