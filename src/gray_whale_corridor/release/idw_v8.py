"""study IDW reproduction from archived observed-cell checkpoints."""
import importlib
import json

from .archived_figures import bound_attributes
from .control import CONFIG
from .corridors_v8 import compare_frames, finish, prepare, require_primary_acceptance, save
from .figure_adapters import plotting_imports
from .late_season_v8 import compare_csv
from .idw_acceptance import check_frequency, check_route_invariants

SETTINGS = {
    'k4_p2': {'neighbors': 4, 'power': 2.0, 'primary': False},
    'k8_p1': {'neighbors': 8, 'power': 1.0, 'primary': False},
    'primary_k8_p2': {'neighbors': 8, 'power': 2.0, 'primary': True},
    'k8_p3': {'neighbors': 8, 'power': 3.0, 'primary': False},
}
OBSERVED_COLUMNS = (
    'grid_id', 'observed_value_window_mean', 'observed_value_window_median',
    'support_station_year_count', 'support_unique_station_count', 'support_year_count',
    'support_years', 'observed_value_target_year_mean', 'target_year_station_count',
)
OUTPUTS = (
    ('p1_idw_spatial_block_predictions.csv.gz', 'reference_cv_predictions', ['setting', 'target_year', 'grid_id']),
    ('p1_idw_spatial_block_metrics.csv', 'reference_cv_metrics', ['setting', 'target_year', 'scope']),
    ('p1_idw_energy_surfaces.csv.gz', 'reference_energy_surfaces', ['setting', 'target_year', 'grid_id']),
    ('p1_idw_frequency_surfaces.csv.gz', 'reference_frequency_surfaces', ['setting', 'scenario', 'target_year', 'grid_id']),
    ('p1_idw_route_metrics.csv', 'reference_route_metrics', ['setting', 'scenario', 'target_year']),
)


def load_observed_checkpoints(paths, module, checks):
    """Recover the saved aggregation output, not a synthetic station table."""
    observed = {}
    reference = module.pd.read_csv(paths.input('reference_cv_predictions'))
    reference = reference.loc[reference.setting.eq('primary_k8_p2')]
    counts = []
    for year in module.YEARS:
        frame = module.pd.read_csv(paths.input(f'prey_{year}'), dtype={'grid_id': str})
        assert frame.target_year.eq(year).all()
        assert frame.window_start.eq(year - 4).all() and frame.window_end.eq(year).all()
        assert len(frame) == 2839 and not frame.grid_id.duplicated().any()
        module.np.testing.assert_array_equal(frame.observed_window_flag, frame.observed_value_window_mean.notna())
        selected = frame.loc[frame.observed_window_flag, list(OBSERVED_COLUMNS)].sort_values('grid_id').reset_index(drop=True)
        expected = reference.loc[reference.target_year.eq(year), ['grid_id', 'observed']]
        actual = selected[['grid_id', 'observed_value_window_mean']].rename(columns={'observed_value_window_mean': 'observed'})
        module.pd.testing.assert_frame_equal(actual, expected.sort_values('grid_id').reset_index(drop=True), check_exact=True)
        compare_frames(paths, f'observed_checkpoint_{year}', actual, expected, ['grid_id'], module.pd, module.np, checks)
        observed[year] = selected
        counts.append({'target_year': year, 'observed_cells': len(selected), 'window_start': year - 4, 'window_end': year})
    module.pd.DataFrame(counts).to_csv(paths.output('qa/observed_checkpoint_inventory.csv'), index=False)
    assert sum(len(value) for value in observed.values()) == 1591
    return observed


def run_idw(paths):
    require_primary_acceptance()
    before = prepare(paths, 'sensitivity_idw')
    checks = []
    with plotting_imports(paths):
        module = importlib.import_module('run_phase8t_primary_robustness_smoke_test')
        scope = json.loads((CONFIG / 'scope.json').read_text())
        assert module.IDW_SETTINGS == SETTINGS
        assert {(v['neighbors'], v['power']) for v in SETTINGS.values()} == {
            (v['k'], v['p']) for v in scope['robustness']['IDW']}
        assert module.YEARS == list(range(2011, 2020))
        assert module.EPSILON == .001 and module.FREQUENT_THRESHOLD == .05
        assert scope['primary']['interpolation']['max_distance_km'] == 250
        grid = module.pd.read_csv(paths.input('grid'), dtype={'grid_id': str})
        observed = load_observed_checkpoints(paths, module, checks)

        def load_aggregates(checkpoints, variable, year):
            assert checkpoints is observed
            assert variable == module.prey_base.VARIABLES[module.PREY_VARIABLE]['station_col']
            return checkpoints[year].copy()

        def stage(name, state, **details):
            assert name == 'p1'
            save(paths, 'qa/stage.json', {'stage': name, 'state': state, **details})

        bindings = {
            'ROOT': paths.data_root, 'GRID': paths.input('grid'), 'ENDPOINTS': paths.input('endpoints'),
            'prey_path': lambda year: paths.input(f'prey_{year}'),
            'ice_path': lambda year: paths.input(f'ice_{year}'),
            'primary_grid_path': lambda scenario, year: paths.input(f'grid_{scenario}_{year}'),
            'primary_route_path': lambda scenario: paths.input('route_' + scenario),
            'CV_OUT': paths.output('source_data'), 'SURFACE_OUT': paths.output('source_data'),
            'LCP_OUT': paths.output('source_data'), 'SUMMARY_OUT': paths.output('source_data'),
            'QA_OUT': paths.output('qa'), 'set_stage': stage,
        }
        # Replace only upstream I/O. The original p1 algorithm, ordering and ties remain intact.
        with bound_attributes(module, bindings), bound_attributes(module.prey_trailing, {
            'load_full_trailing_inputs': lambda: (grid.copy(), observed, None),
        }), bound_attributes(module.prey_base, {'aggregate_observed_cells': load_aggregates}):
            topology = module.load_topology()
            assert len(topology.source_indices) == len(topology.destination_indices) == 40
            module.p1_interpolation(topology)
        for filename, alias, keys in OUTPUTS:
            if alias == 'reference_frequency_surfaces':
                actual = module.pd.read_csv(paths.output('source_data/' + filename))
                reference = module.pd.read_csv(paths.input(alias))
                checks.append(check_frequency(actual, reference, module.pd, module.np))
                save(paths, 'qa/comparisons.json', checks)
            else:
                compare_csv(paths, filename, paths.output('source_data/' + filename), alias, keys, module, checks)
        compare_csv(paths, 'primary_reproduction', paths.output('qa/p1_primary_reproduction_check.csv'),
                    'reference_primary_check', ['scenario', 'target_year'], module, checks)
        routes = module.pd.read_csv(paths.output('source_data/p1_idw_route_metrics.csv'))
        assert len(routes) == 72
        assert routes.successful_path_count.eq(1600).all() and routes.failed_path_count.eq(0).all()
        check_route_invariants(routes, module.pd.read_csv(paths.input('reference_route_metrics')), module.pd)
        figure = importlib.import_module('build_phase8t_robustness_candidates_v1')
        source = figure.range_table(routes.loc[routes.scenario.eq('prey_only')], 'primary_k8_p2',
            ['mean_path_cost', 'corridor_area_cells_freq_gt_0_05'])
        source.to_csv(paths.output('source_data/figure2_idw_k_le_8_source_data_v2.csv'), index=False)
        compare_frames(paths, 'Fig2C_source', source, module.pd.read_csv(paths.input('reference_Fig2')),
                       ['target_year'], module.pd, module.np, checks)
        module.pd.testing.assert_frame_equal(source.reset_index(drop=True),
            module.pd.read_csv(paths.input('reference_Fig2')).reset_index(drop=True), check_dtype=False, check_exact=True)
    exceptions = [check for check in checks if check.get('known_difference_profile_used')]
    finish(paths, before, checks, {
        'status': 'passed_archived_input_reproduction_with_documented_exception' if exceptions else 'passed_archived_input_reproduction',
        'documented_frequency_exceptions': exceptions,
        'frequent_sets_centroids_and_Fig2_values_exact': True,
        'layer': 'Archived five-year observed-cell means to CV, IDW surfaces, alternative corridor solutions and Fig2C source',
        'settings': SETTINGS, 'scenario_year_rows': 72, 'alternative_scenario_years_recomputed': 54,
        'alternative_paths_recomputed': 86400, 'primary_scenario_years_reused_as_in_original': 18,
        'primary_corridor_reuse_requires_previous_independent_acceptance': True,
        'observed_cell_years': 1591, 'CV_prediction_rows': 6364, 'CV_summary_rows': 180,
        'raw_station_aggregation_recomputed': False, 'energy_conversion_recomputed': False,
        'IDW_radius_km': 250, 'scaling': 'q05/q95 pooled over nine rebuilt annual surfaces separately for each setting',
        'plots_rendered': False, 'random_sampling': False})
