"""study energy surfaces from existing station totals and coordinate matches."""
import importlib
import json

from .archived_figures import bound_attributes
from .corridors_v8 import VARIABLE, compare_frames as compare_numeric_frames, finish, prepare, save
from .figure_adapters import plotting_imports

YEARS = tuple(range(2011, 2020))


def compare_frames(paths, label, actual, expected, keys, pd, np, checks):
    """Represent booleans as 0/1 only in comparison copies, never in outputs."""
    a, b = actual.copy(), expected.copy()
    flags = []
    for field in set(a.columns) & set(b.columns):
        if pd.api.types.is_bool_dtype(a[field]) and pd.api.types.is_bool_dtype(b[field]):
            flags.append(field)
            a[field], b[field] = a[field].astype('uint8'), b[field].astype('uint8')
    compare_numeric_frames(paths, label, a, b, keys, pd, np, checks)
    # The existing tolerance cannot admit a changed 0/1 value; verify explicitly too.
    for field in flags:
        np.testing.assert_array_equal(actual.sort_values(keys)[field], expected.sort_values(keys)[field])
    checks[-1]['boolean_fields_checked_exactly'] = sorted(flags)
    save(paths, 'qa/comparisons.json', checks)


def run_prey_surfaces(paths):
    before = prepare(paths, 'prey_processing')
    checks = []
    with plotting_imports(paths):
        import numpy as np
        import pandas as pd
        full = importlib.import_module('build_phase8b2_brian_amphipod_prey_surfaces_full_trailing_c3_v2')
        base = full.base
        assert tuple(full.TARGET_YEARS) == tuple(base.TARGET_YEARS) == YEARS
        assert (base.IDW_K, base.IDW_POWER, base.IDW_MAX_DISTANCE_KM) == (8, 2.0, 250.0)
        assert base.WINDOW_YEARS == 5 and base.IDW_LONG_DISTANCE_QA_KM == 150.0
        assert [full.full_trailing_window_for_target(y) for y in YEARS] == [(y-4, y) for y in YEARS]
        energy = {VARIABLE: base.VARIABLES[VARIABLE]}
        assert energy[VARIABLE]['station_col'] == 'total_amphipod_kJm2_frozen'
        output = paths.output('source_data')
        # Original variable loops are independent; run only study's primary energy surface.
        with bound_attributes(full, {
            'GRID': paths.input('grid'), 'C3_DOMAIN': paths.input('domain'),
            'STATION_TOTALS': paths.input('station_totals'), 'UNIQUE_MATCH': paths.input('station_matches'),
            'OUT': output, 'VARIABLES': energy,
        }), bound_attributes(base, {'OUT': output, 'VARIABLES': energy}):
            grid, stations, _ = full.load_full_trailing_inputs()
            if len(grid) != 2839 or grid.grid_id.duplicated().any():
                raise ValueError('Unexpected V8 grid')
            stations.to_csv(output/'selected_station_year_inputs.csv', index=False)
            surfaces, support = full.build_surfaces(grid, stations)
            scaling = full.apply_global_scaling_full(surfaces)
            full.write_surfaces_full(surfaces)
            summary = full.summarize_full(surfaces, support)
        for year in YEARS:
            name = f'amphipod_prey_surface_{VARIABLE}_target{year}_full_trailing_C3_v2.csv'
            actual = pd.read_csv(output/name)
            expected = pd.read_csv(paths.input(f'reference_{year}'))
            compare_frames(paths, f'energy_surface_{year}', actual, expected, ['grid_id'], pd, np, checks)
            for field in ('observed_target_year_flag','observed_window_flag','interpolated_flag',
                          'supported_for_surface','reliability_flag','idw_neighbor_count'):
                np.testing.assert_array_equal(actual[field], expected[field])
            print('Verified primary energy surface:', year, len(actual), 'cells', flush=True)
        expected_summary = pd.read_csv(paths.input('reference_summary'))
        expected_summary = expected_summary.loc[expected_summary.prey_variable.eq(VARIABLE)]
        compare_frames(paths, 'energy_annual_summary', summary, expected_summary,
                       ['prey_variable','target_year'], pd, np, checks)
        expected_scaling = json.loads(paths.input('reference_scaling').read_text())
        assert scaling['normalization_mode'] == expected_scaling['normalization_mode']
        assert scaling['q99_recorded_for_QA_only'] == expected_scaling['q99_recorded_for_QA_only']
        compare_frames(paths, 'energy_scaling', pd.DataFrame([scaling['variables'][VARIABLE]]),
                       pd.DataFrame([expected_scaling['variables'][VARIABLE]]), ['station_col'], pd, np, checks)
        assert len(checks) == 11
        details = {
            'layer': 'Existing station-year energy totals and saved coordinate matches to primary annual energy/utility surfaces',
            'surface_years': list(YEARS), 'surface_rows': sum(len(v) for v in surfaces[VARIABLE].values()),
            'selected_station_year_rows': len(stations),
            'selected_station_year_rows_by_year': stations.groupby('Year').size().to_dict(),
            'observed_cells_by_target_year': summary.set_index('target_year')['observed_window_cell_count'].to_dict(),
            'scaling': scaling['variables'][VARIABLE], 'doi_required_for_execution': False,
            'input_source_documentation_pending': True,
            'source_energy_values_reconverted': False, 'initial_cleaning_matching_aggregation_reconstructed': False,
            'alternate_currencies_recomputed': False, 'corridors_or_figures_recomputed': False,
            'station_inputs_proposed_for_public_rehosting': False,
        }
        save(paths, 'qa/starting_layer.json', details)
    finish(paths, before, checks, details)
