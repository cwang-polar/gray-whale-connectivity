"""Preparation entry contracts; never execute a scientific pipeline."""
import ast
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import os
from types import SimpleNamespace

SOFTWARE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE / 'src'))
from gray_whale_corridor.release.ierp_v8 import EXPECTED_COUNTS, OUTPUTS
from gray_whale_corridor.release.paths import TaskPaths, load_policy
from gray_whale_corridor.release.runtime import ALLOWED, PREPARATION_ALLOWED
from gray_whale_corridor.release.runtime import AccessGuard
from gray_whale_corridor.release.ierp_inputs import check_outputs, prepared_files, registry


class IerpPreparationTests(unittest.TestCase):
    def test_separate_entry_and_scope(self):
        self.assertNotIn('prepare_ierp', ALLOWED)
        self.assertEqual(len(ALLOWED), 35)
        self.assertIn('prepare_ierp', PREPARATION_ALLOWED)
        self.assertEqual(EXPECTED_COUNTS, {2017: 38, 2018: 28})
        self.assertEqual(len(OUTPUTS), 3)
        self.assertEqual(len(load_policy()['tasks']), 35)

    def test_inputs_and_no_fallback(self):
        paths = TaskPaths('prepare_ierp', SOFTWARE.parent / 'dataset', 'ierp_static_test',
                          policy_file='ierp_input_preparation.json')
        self.assertEqual(len(paths.contract['inputs']), 5)
        self.assertFalse(paths.task_root.exists())
        self.assertFalse(paths.contract['hold_reasons'])
        for alias in ('grid', 'pelagic_2017', 'pelagic_2018'):
            self.assertTrue(paths.input(alias).is_file())
        for alias in ('glorys', 'ierp_csv'):
            relative = paths.contract['aliases'][alias]
            self.assertEqual(paths.policy['files'][relative]['role'], 'external_acquisition_local_validation_only')
            with patch.object(Path, 'is_file', return_value=False):
                with self.assertRaisesRegex(FileNotFoundError, 'Missing V8 input; no fallback'):
                    paths.input(alias)
        self.assertFalse(any(key.startswith('reference_') for key in paths.contract['aliases']))
        self.assertEqual(len(paths.forbidden_read_paths), 3)

    def test_acceptance_registry_checksum_and_rejection(self):
        self.assertEqual(len(registry()['outputs']), 3)
        with patch('gray_whale_corridor.release.ierp_inputs.digest', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'acceptance metadata changed'):
                registry()

    def test_no_fit_path_or_plot_calls(self):
        tree = ast.parse((SOFTWARE / 'src/gray_whale_corridor/release/ierp_v8.py').read_text())
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {'fit', 'nested_spatial_cv', 'bootstrap_surface', 'build_early_surfaces', 'run_central', 'savefig'})
        self.assertTrue({'parse_asgard_abundance', 'match_to_grid', 'load_model_data', 'sample_glorys', 'load_observed_support'} <= calls)

    def test_accepts_only_registered_exact_output_profiles(self):
        records = registry()['outputs']
        outputs = {key: Path(spec['filename']) for key,spec in records.items()}
        for profile in ('original_reference', 'approved_projection_roundoff'):
            hashes = {v['filename']: v['accepted_sha256'][profile] for v in records.values()}
            with patch.object(Path, 'is_file', return_value=True), patch(
                    'gray_whale_corridor.release.ierp_inputs.digest', side_effect=lambda p: hashes[p.name]), patch(
                    'gray_whale_corridor.release.ierp_inputs.registry', return_value={'outputs': records}):
                self.assertTrue(all(v['accepted_profile'] == profile for v in check_outputs(outputs).values()))
                hashes[records['s11']['filename']] = '0' * 64
                with self.assertRaises(ValueError):
                    check_outputs(outputs)

    def test_rejects_mixed_profiles_and_changed_schema(self):
        records = registry()['outputs']
        outputs = {key: Path(spec['filename']) for key,spec in records.items()}
        hashes = {v['filename']: v['accepted_sha256']['original_reference'] for v in records.values()}
        hashes[records['s11']['filename']] = records['s11']['accepted_sha256']['approved_projection_roundoff']
        with patch.object(Path, 'is_file', return_value=True), patch(
                'gray_whale_corridor.release.ierp_inputs.digest', side_effect=lambda p: hashes[p.name]), patch(
                'gray_whale_corridor.release.ierp_inputs.registry', return_value={'outputs': records}):
            with self.assertRaises(ValueError):
                check_outputs(outputs)
            with self.assertRaises(ValueError):
                check_outputs({'records': outputs['records']})

    def test_prepared_run_ids_and_task_scope(self):
        with self.assertRaises(ValueError):
            prepared_files(SOFTWARE.parent / 'runs', '../escape')
        with self.assertRaises(ValueError):
            TaskPaths('Fig5', SOFTWARE.parent / 'dataset', 'ierp_task_scope_test', prepared_ierp_run='anything')
        with self.assertRaises(ValueError):
            TaskPaths('S11', SOFTWARE.parent / 'dataset', 'ierp_missing_receipt_test', prepared_ierp_run='no_such_preparation')

    def test_guard_denies_old_observation_tables(self):
        context = TaskPaths('prepare_ierp', SOFTWARE.parent / 'dataset', 'ierp_guard_test',
                            policy_file='ierp_input_preparation.json')
        guard = AccessGuard(context)
        for path in context.forbidden_read_paths:
            with self.assertRaises(PermissionError):
                guard('open', (str(path), 'r', os.O_RDONLY))

    def test_prepared_inputs_are_read_only_and_scoped(self):
        source = SOFTWARE.parent / 'runs/other/prepare_ierp/source_data/observations.csv'
        context = SimpleNamespace(data_root=SOFTWARE.parent/'dataset', run_root=SOFTWARE.parent/'runs/test',
            task_root=SOFTWARE.parent/'runs/test/S11', prepared_read_paths={source}, forbidden_read_paths=set())
        guard = AccessGuard(context)
        guard('open', (str(source), 'r', os.O_RDONLY))
        with self.assertRaises(PermissionError):
            guard('open', (str(source), 'w', os.O_WRONLY))
        with self.assertRaises(PermissionError):
            guard('open', (str(source.parent/'other.csv'), 'r', os.O_RDONLY))


if __name__ == '__main__':
    unittest.main()
