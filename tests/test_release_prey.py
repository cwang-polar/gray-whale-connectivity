"""Scope, source/execution separation and unchanged scientific functions."""
import ast
import json
from pathlib import Path
import sys
import subprocess
import textwrap
import unittest

SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE/'src'))
from gray_whale_corridor.release import prey_v8
from gray_whale_corridor.release.paths import TaskPaths, load_policy


class PreyEntryTests(unittest.TestCase):
    def test_boolean_comparison_is_exact_without_subtraction(self):
        # Keep scientific imports out of the entry-control test process.
        code = textwrap.dedent('''
            import sys
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch
            sys.path.insert(0, str(Path(sys.argv[1])/'src'))
            import numpy as np
            import pandas as pd
            from gray_whale_corridor.release import corridors_v8, prey_v8
            frame = pd.DataFrame({'grid_id':['a','b'], 'supported':[True,False], 'value':[1.,2.]})
            paths = SimpleNamespace(output=lambda name: name)
            checks = []
            with patch.object(corridors_v8,'save'), patch.object(prey_v8,'save'), patch.object(pd.DataFrame,'to_csv'):
                prey_v8.compare_frames(paths,'bool-check',frame,frame.copy(),['grid_id'],pd,np,checks)
                assert checks[0]['passed'] and checks[0]['boolean_fields_checked_exactly'] == ['supported']
                assert frame['supported'].dtype == bool
                changed = frame.copy()
                changed.loc[0,'supported'] = False
                try:
                    prey_v8.compare_frames(paths,'bool-check',changed,frame,['grid_id'],pd,np,[])
                except RuntimeError as error:
                    assert 'Frozen V8 mismatch' in str(error)
                else:
                    raise AssertionError('Changed boolean accepted')
        ''')
        result = subprocess.run([sys.executable,'-B','-c',code,str(SOFTWARE)], capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_exact_energy_scope(self):
        scope = json.loads((SOFTWARE/'config/ncc_v8/scope.json').read_text())
        self.assertEqual(prey_v8.VARIABLE, scope['primary']['prey_variable'])
        self.assertEqual(prey_v8.YEARS, tuple(range(2011,2020)))

    def test_checkpoints_are_explicit_and_not_doi_gated(self):
        task = load_policy()['tasks']['prey_processing']
        self.assertEqual(len(task['inputs']),15)
        self.assertFalse(task['hold_reasons'])
        self.assertIn('station_totals', task['aliases'])
        self.assertIn('station_matches', task['aliases'])
        self.assertFalse(any('doi' in alias.lower() for alias in task['aliases']))
        context = TaskPaths('prey_processing', SOFTWARE.parent/'dataset', 'read_only_prey_contract')
        self.assertEqual(len(context.snapshot()),15)
        self.assertFalse(context.task_root.exists())

    def test_no_historical_main_energy_reconversion_or_corridor_dispatch(self):
        tree = ast.parse((SOFTWARE/'src/gray_whale_corridor/release/prey_v8.py').read_text())
        names = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for function in ('full.load_full_trailing_inputs','full.build_surfaces','full.apply_global_scaling_full',
                         'full.write_surfaces_full','full.summarize_full'):
            self.assertIn(function,names)
        self.assertFalse(any(n.endswith('.main') or 'run_paths' in n or 'convert_energy' in n for n in names))


if __name__ == '__main__':
    unittest.main()
