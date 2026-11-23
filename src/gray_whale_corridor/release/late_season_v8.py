"""study August-October reproduction using the unchanged Phase 8Q functions."""
import importlib
import json

from .archived_figures import bound_attributes
from .control import CONFIG
from .corridors_v8 import compare_frames, prepare, require_primary_acceptance, save
from .figure_adapters import plotting_imports

ASAMM_GROUPS = ('1979_2011', '2012_2014', '2015_2017', '2018_2021')
MAPPING_DISTANCE_ATOL_KM = 1e-10
VALIDATION_KEYS = {
    'sightings_study_region': ['record_id'],
    'strict_effort_cells': ['year', 'grid_id'],
    'four_scenario_record_metrics': ['scenario', 'record_id'],
    'four_scenario_by_year': ['scenario', 'target_year'],
    'available_period_summary': ['scenario'],
    'common_2015_2019_summary': ['scenario'],
}


def check_config(module, paths):
    scope = json.loads((CONFIG / 'scope.json').read_text())
    original = json.loads(paths.input('reference_config').read_text())
    ship = json.loads(paths.input('ship_scaling').read_text())
    assert module.ALL_YEARS == original['target_years'] == list(range(2011, 2020))
    assert module.VESSEL_YEARS == original['vessel_years'] == list(range(2015, 2020))
    assert module.MONTHS == original['months'] == scope['late_months'] == [8, 9, 10]
    assert original['prey_variable'] == scope['primary']['prey_variable'] == 'total_kJm2_frozen'
    assert module.EPSILON == original['utility_floor_before_resistance'] == scope['graph']['numerical_epsilon']
    assert [module.SIC_LOW, module.SIC_HIGH] == scope['ice']['primary_anchors'] == [.15, .55]
    assert module.A_MIN == original['ice_transform']['minimum_accessibility'] == .05
    assert module.FREQUENT_THRESHOLD == .05
    assert original['vessel_primary_log_q99'] == scope['vessel']['fixed_log_q99'] == ship['log_q99']
    assert original['vessel_alpha'] == scope['vessel']['alpha'] == .5


def compare_csv(paths, label, actual_path, reference_alias, keys, module, checks):
    actual = module.pd.read_csv(actual_path)
    reference = module.pd.read_csv(paths.input(reference_alias))
    if label == 'ASAMM_sightings_study_region':
        column = 'distance_to_nearest_grid_cell_km'
        a, b = actual.sort_values(keys).reset_index(drop=True), reference.sort_values(keys).reset_index(drop=True)
        module.pd.testing.assert_frame_equal(a[keys], b[keys], check_exact=True)
        module.pd.testing.assert_series_equal(a[column], b[column], check_exact=False,
            rtol=0, atol=MAPPING_DISTANCE_ATOL_KM)
        checks.append({'comparison': label + '_mapping_distance', 'rows': len(a), 'passed': True,
            'numeric_atol_km': MAPPING_DISTANCE_ATOL_KM, 'numeric_rtol': 0,
            'max_numeric_difference': {column: float((a[column] - b[column]).abs().max())},
            'reason': 'Sub-micrometre projection arithmetic tolerance only; not a corridor-distance metric. '
                      'Record IDs, geographic coordinates, assigned cells and every spatial metric are checked separately.',
            'source_values_altered': False})
        # Check this auxiliary field separately; all other fields retain the common strict check.
        actual, reference = actual.drop(columns=column), reference.drop(columns=column)
    # Boolean membership is checked exactly; cast only for the numeric QA reporter.
    for column in set(actual.columns).intersection(reference.columns):
        if module.pd.api.types.is_bool_dtype(actual[column]) and module.pd.api.types.is_bool_dtype(reference[column]):
            module.np.testing.assert_array_equal(
                actual.sort_values(keys)[column], reference.sort_values(keys)[column])
            actual[column] = actual[column].astype(int)
            reference[column] = reference[column].astype(int)
    compare_frames(paths, label, actual, reference, keys, module.pd, module.np, checks)


def compare_s5_sources(paths, tree, module, checks):
    figure = importlib.import_module('build_si_d1_aug_oct_corridor_observed_use_v1')
    output = paths.output('source_data/S5')
    output.mkdir()
    with bound_attributes(figure, {
        'COMPARISON_PATH': tree['summaries'] / 'phase8q_aug_oct_scenario_comparisons.csv',
        'ROUTE_PATH': tree['summaries'] / 'phase8q_aug_oct_route_summary.csv',
        'ANNUAL_OBSERVED_PATH': tree['validation'] / 'asamm_aug_oct_four_scenario_by_year.csv',
        'AVAILABLE_OBSERVED_PATH': tree['validation'] / 'asamm_aug_oct_available_period_summary.csv',
        'COMMON_OBSERVED_PATH': tree['validation'] / 'asamm_aug_oct_common_2015_2019_summary.csv',
        'SOURCE_DIR': output,
    }):
        comparisons, _, annual, full, common = figure.load_inputs()
        contrasts = figure.export_corridor_source(comparisons)
        figure.build_observed_source(annual, full, common)
        compare_csv(paths, 'S5_corridor_source', contrasts, 'reference_S5_corridor_contrasts',
                    ['comparison', 'target_year'], module, checks)
        compare_csv(paths, 'S5_observed_source', output / 'gray_whale_si_d1_observed_use_source_data_v1.csv',
                    'reference_S5_observed_use', ['display_period', 'summary_type', 'scenario', 'target_year'], module, checks)


def run_late_season(paths):
    require_primary_acceptance()
    before = prepare(paths, 'late_season')
    checks, routes = [], []
    tree = {name: paths.output('source_data/' + name) for name in ('surfaces', 'lcp', 'summaries', 'validation')}
    for directory in tree.values():
        directory.mkdir()
    with plotting_imports(paths):
        module = importlib.import_module('run_phase8q_aug_oct_late_season_connectivity')
        check_config(module, paths)
        grid = module.pd.read_csv(paths.input('grid'), dtype={'grid_id': str})
        grid = grid.loc[grid['ocean_flag'].eq(1)].reset_index(drop=True)
        endpoints = module.pd.read_csv(paths.input('endpoints'))
        topology = module.lcp_base.build_graph_topology(grid, endpoints)
        assert not topology.warnings, topology.warnings
        assert len(grid) == 2839
        assert len(topology.source_indices) == len(topology.destination_indices) == 40
        scope = module.asamm_scope
        with bound_attributes(module, {
            'ROOT': paths.data_root, 'GRID': paths.input('grid'), 'ENDPOINTS': paths.input('endpoints'),
            'prey_path': lambda year: paths.input(f'prey_{year}'),
            'early_lcp_path': lambda scenario, year: paths.input(f'early_grid_{scenario}_{year}'),
        }), bound_attributes(scope, {
            'ROOT': paths.data_root, 'GRID_PATH': paths.input('grid'), 'DOMAIN_PATH': paths.input('domain'),
            'ASAMM_OUTPUTS': [paths.input('asamm_' + group) for group in ASAMM_GROUPS],
        }):
            for year in module.ALL_YEARS:
                prey = module.load_prey(year, grid)
                ice = module.pd.read_csv(paths.input(f'ice_{year}'), dtype={'grid_id': str})
                vessel = module.pd.read_csv(paths.input(f'vessel_{year}'), dtype={'grid_id': str}) if year in module.VESSEL_YEARS else None
                for scenario, years in module.SCENARIO_YEARS.items():
                    if year not in years:
                        continue
                    surface = module.build_scenario_surface(scenario, year, prey, ice, vessel)
                    stem = f'{scenario}_aug_oct_target{year}_study_region.csv'
                    surface.to_csv(tree['surfaces'] / ('utility_' + stem), index=False)
                    compare_csv(paths, f'{scenario}_{year}_utility', tree['surfaces'] / ('utility_' + stem),
                                f'reference_utility_{scenario}_{year}', ['grid_id'], module, checks)
                    corridor, path_frame, route = module.run_scenario_lcp(scenario, year, surface, topology)
                    module.write_outputs(tree, scenario, year, surface, corridor, path_frame)
                    assert route['successful_path_count'] == 1600 and route['failed_path_count'] == 0
                    compare_csv(paths, f'{scenario}_{year}_corridor', tree['lcp'] / ('corridor_' + stem),
                                f'reference_corridor_{scenario}_{year}', ['grid_id'], module, checks)
                    compare_csv(paths, f'{scenario}_{year}_paths', tree['lcp'] / ('paths_' + stem),
                                f'reference_paths_{scenario}_{year}',
                                ['computational_source_grid_id', 'computational_destination_grid_id'], module, checks)
                    routes.append(route)
                    print('Verified late season:', scenario, year, '1600 path identities and costs', flush=True)
            assert len(routes) == 28
            route_summary = module.pd.DataFrame(routes)
            route_file = tree['summaries'] / 'phase8q_aug_oct_route_summary.csv'
            route_summary.to_csv(route_file, index=False)
            compare_csv(paths, 'route_summary', route_file, 'reference_route_summary',
                        ['scenario', 'target_year'], module, checks)
            module.build_scenario_comparisons(tree, route_summary, module.ALL_YEARS)
            early = module.compare_with_early_season(tree, route_summary, module.ALL_YEARS)
            for name, keys in (('scenario_comparisons', ['comparison', 'target_year']),
                               ('vs_may_july_comparison', ['scenario', 'target_year'])):
                compare_csv(paths, name, tree['summaries'] / f'phase8q_aug_oct_{name}.csv',
                            'reference_' + name, keys, module, checks)
            assert early.loc[early.scenario.eq('prey_only'), 'maximum_absolute_frequency_difference'].max() < 1e-12
            print('Corridors verified; reconstructing August-October ASAMM selection from four CSV exports.', flush=True)
            module.build_asamm_validation(tree, module.ALL_YEARS)
            for name, keys in VALIDATION_KEYS.items():
                compare_csv(paths, 'ASAMM_' + name, tree['validation'] / f'asamm_aug_oct_{name}.csv',
                            'reference_' + name, keys, module, checks)
            qa = json.loads((tree['validation'] / 'asamm_aug_oct_validation_qa.json').read_text())
            if qa != json.loads(paths.input('reference_validation_qa').read_text()):
                raise RuntimeError('Late-season ASAMM QA differs from the frozen reference')
            assert qa['all_year_unique_sighting_records'] == 1843
            assert qa['common_2015_2019_unique_records'] == 1131
            assert qa['all_year_strict_effort_events'] == 42921
            compare_s5_sources(paths, tree, module, checks)
    assert paths.snapshot() == before
    maximum = max(value for result in checks for value in result.get('max_numeric_difference', {}).values())
    save(paths, 'qa/acceptance.json', {
        'status': 'passed_archived_input_reproduction', 'task': 'late_season',
        'starting_layer': 'Archived primary prey utility and August-October ice/vessel modifiers; original ASAMM CSV exports',
        'scenario_years_recomputed': 28, 'paths_recomputed': 44800,
        'comparison_checks_passed': len(checks), 'table_comparisons_passed': len(checks) - 1,
        'separate_auxiliary_field_checks': 1, 'maximum_numeric_difference': maximum,
        'path_sequences_counts_and_memberships_exact': True,
        'ASAMM_record_ids_coordinates_effort_membership_checked': True,
        'ASAMM_QA': qa, 'S5_source_data_reproduced': True,
        'auxiliary_mapping_distance_atol_km': MAPPING_DISTANCE_ATOL_KM,
        'auxiliary_mapping_distance_max_difference_km': next(
            r['max_numeric_difference']['distance_to_nearest_grid_cell_km'] for r in checks
            if r['comparison'] == 'ASAMM_sightings_study_region_mapping_distance'),
        'raw_environment_preprocessing_recomputed': False, 'raw_benthic_processing_recomputed': False,
        'late_season_q99_alternative_executed': False, 'Phase8U_executed': False,
        'figures_redrawn': False, 'scientific_parameters_changed': False, 'input_hashes_unchanged': True,
    })
    print('Late-season acceptance passed:', len(checks), 'checks; 1843 / 1131 sightings.', flush=True)
