"""Prepare study June IERP observation tables, without fitting or drawing."""
import importlib

from .archived_figures import bound_attributes
from .control import digest
from .corridors_v8 import prepare, save
from .figure_adapters import plotting_imports
from .ierp_inputs import check_outputs

EXPECTED_COUNTS = {2017: 38, 2018: 28}
OUTPUTS = {
    'records': 'phase8s_early_model_records_with_glorys.csv',
    'points': 'phase8s_observation_points_source_data_v1.csv',
    's11': 'c5_pelagic_observations_source_data_v2.csv',
}


def compare_table(actual_path, reference_path, pd, np):
    actual = pd.read_csv(actual_path, dtype={'grid_id': str})
    expected = pd.read_csv(reference_path, dtype={'grid_id': str})
    result = {'output': actual_path.name, 'rows': len(actual), 'reference_rows': len(expected),
              'sha256': digest(actual_path), 'reference_sha256': digest(reference_path),
              'numeric_rtol': 1e-12, 'numeric_atol': 1e-12, 'differences': {}}
    result['byte_identical'] = result['sha256'] == result['reference_sha256']
    result['schema_and_rows_match'] = list(actual.columns) == list(expected.columns) and len(actual) == len(expected)
    if not result['schema_and_rows_match']:
        result['passed'] = False
        return result
    keys = ['source_id', 'year', 'month', 'grid_id']
    result['record_order_matches'] = actual[keys].equals(expected[keys])
    valid = result['record_order_matches']
    for column in actual:
        a, b = actual[column], expected[column]
        missing_equal = a.isna().equals(b.isna())
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same = np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-12, atol=1e-12, equal_nan=True)
            exact = a.eq(b) | (a.isna() & b.isna())
            if not exact.all():
                result['differences'][column] = {'changed_rows': int((~exact).sum()),
                    'outside_tolerance_rows': int((~same).sum()),
                    'max_absolute_difference': float((a-b).abs().max())}
            valid = valid and missing_equal and bool(same.all())
        else:
            same = a.fillna('').astype(str).eq(b.fillna('').astype(str))
            if not same.all():
                result['differences'][column] = {'changed_rows': int((~same).sum())}
            valid = valid and missing_equal and bool(same.all())
    result['passed'] = bool(valid)
    return result


def prepare_ierp(paths):
    before = prepare(paths, 'prepare_ierp')
    source = paths.output('source_data')
    with plotting_imports(paths):
        import numpy as np
        import pandas as pd
        import xarray as xr
        origin = importlib.import_module('audit_phase8n_comprehensive_euphausiid_integration')
        model = importlib.import_module('audit_phase8s_observation_informed_pelagic_support')
        figure = importlib.import_module('build_si_c5_pelagic_support_evidence_v1')
        assert origin.GRID_MATCH_LIMIT_KM == 15
        with bound_attributes(origin, {'RAW': paths.input('ierp_csv').parents[1], 'GRID': paths.input('grid')}):
            observations = origin.parse_asgard_abundance()
            matched, gridded = origin.match_to_grid(observations)
        counts = gridded.groupby('year').size().to_dict()
        assert counts == EXPECTED_COUNTS, counts
        assert gridded.month.eq(6).all() and not gridded.duplicated(['year', 'grid_id']).any()
        matched.to_csv(source / 'ierp_june_matched_source_records.csv', index=False)
        evidence = source / 'ierp_june_grid_evidence.csv'
        gridded.to_csv(evidence, index=False)
        with bound_attributes(model, {'EVIDENCE': evidence, 'TABLES': source}):
            _, early, late = model.load_model_data()
        assert late.empty
        with xr.open_dataset(paths.input('glorys')) as dataset:
            early, qa = model.sample_glorys(early, dataset)
        qa.to_csv(source / 'ierp_glorys_sampling_qa.csv', index=False)
        outputs = {key: source / name for key, name in OUTPUTS.items()}
        # Preserve historical CSV handoffs, including their original precision.
        early.to_csv(outputs['records'], index=False)
        pd.read_csv(outputs['records'], dtype={'grid_id': str}).to_csv(outputs['points'], index=False)
        surfaces = {year: pd.read_csv(paths.input(f'pelagic_{year}'), dtype={'grid_id': str}) for year in EXPECTED_COUNTS}
        with bound_attributes(figure, {'OBSERVATIONS_PATH': outputs['points']}):
            figure.load_observed_support(surfaces).to_csv(outputs['s11'], index=False)
        checks = check_outputs(outputs)
        save(paths, 'qa/output_identity_checks.json', checks)
        assert paths.snapshot() == before
        original = all(item['accepted_profile'] == 'original_reference' for item in checks.values())
        save(paths, 'qa/acceptance.json', {
            'status': 'passed_exact_original_output' if original else 'passed_with_documented_auxiliary_distance_exception',
            'task': 'prepare_ierp', 'output_identity_checks_passed': len(checks),
            'all_three_byte_identical_to_original': original, 'observation_cells_by_year': counts,
            'inputs': before, 'outputs': checks,
            'source_rows_by_year': observations.groupby('year').size().to_dict(),
            'matched_source_rows_by_year': matched.loc[matched.grid_match_within_15km].groupby('year').size().to_dict(),
            'starting_layer': 'Original Gonzalez INPUT CSV and original GLORYS NetCDF; archived grid and support-scaling anchors',
            'legacy_observation_tables_read': False,
            'fitting_CV_bootstrap_interpolation_domain_paths_figures_run': False,
            'predictor_extraction_run': True,
            'reference_files_replaced': False,
            'sample_parameter_and_taxon_rules_changed': False,
            'remaining': ['This does not validate upstream model fitting or prediction generation.',
                'Raw external inputs are acquisition dependencies, not automatic redistribution permission.'],
        })
        print(f'IERP preparation: three verified output identities; exact-original={original}.', flush=True)
