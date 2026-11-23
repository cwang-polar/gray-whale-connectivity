"""Frozen study pelagic integration from archived predictions, not a model refit."""
import importlib
import json

from .archived_figures import bound_attributes
from .control import CONFIG
from .corridors_v8 import prepare, require_primary_acceptance, save
from .figure_adapters import plotting_imports
from .late_season_v8 import compare_csv

YEARS = (2017, 2018)
SCENARIOS = ('prey', 'prey_ice', 'prey_vessel', 'prey_ice_vessel')
DESIGN = {'complementary': (0., .1, .25, .5, 1.), 'maximum': (.25, .5), 'additive_capped': (.25, .5)}
CENTRAL_KEYS = ['target_year', 'combination_operator', 'pelagic_strength_beta', 'scenario']
POOLED_KEYS = CENTRAL_KEYS[1:]


def check_inputs(paths, module):
    scope = json.loads((CONFIG / 'scope.json').read_text())
    pelagic = scope['pelagic']
    assert list(YEARS) == pelagic['observed_years']
    assert module.YEARS == YEARS and module.SCENARIOS == SCENARIOS
    assert module.CENTRAL_DESIGN == DESIGN
    assert module.UNCERTAINTY_DRAWS == pelagic['used_corridor_draws'] == 100
    assert module.UNCERTAINTY_OPERATOR == 'complementary'
    assert module.UNCERTAINTY_BETA == pelagic['primary_beta'] == .25
    assert module.EPSILON == scope['graph']['numerical_epsilon'] == .001
    assert module.FREQUENT_THRESHOLD == .05
    module.verify_model_gates()
    qa = module.pd.read_csv(paths.input('early_surface_qa')).set_index('target_year')
    for position, year in enumerate(YEARS):
        base = module.load_year_inputs(year)
        assert len(base) == 2839 and not base.grid_id.duplicated().any()
        assert qa.loc[year, 'observation_grid_cell_count'] == pelagic['observed_grid_counts'][position]
        assert int(base.pelagic_evidence_domain.sum()) == pelagic['supported_grid_counts'][position]
        assert qa.loc[year, 'prediction_domain_cell_count'] == pelagic['supported_grid_counts'][position]
        assert qa.loc[year, 'shared_interpolation_limit_km'] == pelagic['max_prediction_distance_km'] == 60
        assert qa.loc[year, 'model_type'] == 'spatial'
        outside = ~base.pelagic_evidence_domain.astype(bool)
        assert (base.loc[outside, 'P_pelagic_central'] == 0).all()
        with module.np.load(paths.input(f'draws_{year}')) as archive:
            assert archive['support_draws'].shape == (pelagic['stored_prediction_draws'], 2839) == (200, 2839)
            assert archive['support_draws'].dtype == module.np.float32
            assert archive['random_seed'].tolist() == [pelagic['prediction_seed']] == [8675309]
        draws = module.align_bootstrap_draws(year, base)
        assert draws.shape == (100, 2839)
        assert module.np.isfinite(draws).all() and (draws >= 0).all() and (draws <= 1).all()
        assert (draws[:, outside] == 0).all()


def compare_output(paths, module, checks, path, keys):
    compare_csv(paths, path.stem, path, 'reference_' + path.stem, keys, module, checks)


def compare_central(paths, module, checks):
    for operator, betas in DESIGN.items():
        for beta in betas:
            for year in YEARS:
                compare_output(paths, module, checks, module.central_surface_path(operator, beta, year), ['grid_id'])
                for scenario in SCENARIOS:
                    compare_output(paths, module, checks, module.central_grid_path(operator, beta, scenario, year), ['grid_id'])
            for scenario in SCENARIOS:
                path = module.LCP / f'phase8s_{operator}_{module.beta_label(beta)}_{scenario}_path_summary.csv'
                compare_output(paths, module, checks, path, ['target_year', 'source_grid_id', 'destination_grid_id'])
    for folder, name, keys in (
        (module.SUMMARIES, 'central_route_summary', CENTRAL_KEYS),
        (module.SUMMARIES, 'central_utility_surface_qa', ['target_year', 'operator', 'beta']),
        (module.VALIDATION, 'central_asamm_validation_summary', CENTRAL_KEYS),
    ):
        compare_output(paths, module, checks, folder / f'phase8s_{name}.csv', keys)


def summarize_central(paths, module, route, tree, checks):
    summary = importlib.import_module('summarize_phase8s_observation_informed_results')
    with bound_attributes(summary, {'ROOT': paths.data_root, 'OUT': paths.output('source_data'),
        'CORRIDOR': paths.output('source_data'), 'SUMMARIES': tree['summaries'],
        'VALIDATION': tree['validation'], 'TABLES': tree['summaries']}):
        records = summary.build_record_table()
        pooled = summary.pooled_validation(records)
        summary.build_change_tables(route, pooled)
        assert len(records) == 11628 and records.record_id.nunique() == 323
        assert len(pooled) == 36 and pooled.sighting_record_count.eq(323).all()
        assert pooled.year_count.eq(2).all()
        for folder, name, keys in (
            (tree['validation'], 'central_asamm_record_metrics', POOLED_KEYS + ['record_id']),
            (tree['validation'], 'central_asamm_pooled_2017_2018_summary', POOLED_KEYS),
            (tree['validation'], 'central_asamm_change_from_beta0_2017_2018', POOLED_KEYS),
            (tree['summaries'], 'central_corridor_change_from_benthic_baseline', CENTRAL_KEYS),
        ):
            compare_output(paths, module, checks, folder / f'phase8s_{name}.csv', keys)


def s12_source_data(paths, module, tree, checks):
    source = importlib.import_module('plot_phase8s_observation_informed_pelagic_support')
    with bound_attributes(source, {'SUMMARIES': tree['summaries']}):
        route = source.complementary_route_summary()
        bootstrap = source.bootstrap_average_intervals()
    pooled = module.pd.read_csv(tree['validation'] / 'phase8s_central_asamm_pooled_2017_2018_summary.csv')
    validation = pooled.loc[pooled.combination_operator.eq('complementary')].copy()
    outputs = {}
    for name, frame, keys in (
        ('corridor_response', route, ['pelagic_strength_beta', 'scenario']),
        ('bootstrap_average_intervals', bootstrap, ['scenario', 'metric']),
        ('asamm_response', validation, POOLED_KEYS),
    ):
        path = tree['S12'] / f'phase8s_{name}_source_data_v1.csv'
        frame.to_csv(path, index=False)
        outputs[name] = path
        compare_csv(paths, 'S12_upstream_' + name, path, 'reference_plot_' + name, keys, module, checks)
    figure = importlib.import_module('build_si_b4_pelagic_support_response_v1')
    with bound_attributes(figure, {
        'ROOT': paths.task_root, 'OUT_ROOT': tree['S12'], 'SOURCE_OUT': tree['S12'],
        'ROUTE_SOURCE': outputs['corridor_response'],
        'BOOTSTRAP_SOURCE': outputs['bootstrap_average_intervals'],
        'ASAMM_SOURCE': outputs['asamm_response'],
    }):
        figure.save_source_data(*figure.load_inputs())
    for name, keys in (('corridor_response', ['pelagic_strength_beta', 'scenario']),
                       ('bootstrap_intervals', ['scenario', 'metric']), ('asamm_response', POOLED_KEYS)):
        path = tree['S12'] / f'b4_pelagic_{name}_source_data_v1.csv'
        compare_csv(paths, 'S12_final_' + name, path, 'reference_S12_' + name, keys, module, checks)


def run_pelagic_support(paths):
    require_primary_acceptance()
    before = prepare(paths, 'pelagic_support')
    checks = []
    tree = {name: paths.output('source_data/' + name) for name in ('surfaces', 'lcp', 'summaries', 'validation', 'S12')}
    for folder in tree.values():
        folder.mkdir()
    with plotting_imports(paths):
        module = importlib.import_module('run_phase8s_observation_informed_corridor_sensitivity')
        base = module.phase8r
        with bound_attributes(base, {
            'ROOT': paths.data_root, 'GRID': paths.input('grid'), 'ENDPOINTS': paths.input('endpoints'),
            'prey_path': lambda year: paths.input(f'prey_{year}'),
            'ice_path': lambda year: paths.input(f'ice_{year}'),
            'ship_path': lambda year: paths.input(f'ship_{year}'),
            'established_grid_path': lambda scenario, year: paths.input(f'baseline_grid_{scenario}_{year}'),
            'established_route_path': lambda scenario: paths.input('baseline_route_' + scenario),
        }), bound_attributes(module, {
            'ROOT': paths.data_root, 'GRID': paths.input('grid'), 'ENDPOINTS': paths.input('endpoints'),
            'support_path': lambda year: paths.input(f'pelagic_{year}'),
            'bootstrap_path': lambda year: paths.input(f'draws_{year}'),
            'MODEL_GATES': paths.input('model_integration_gates'),
            'ASAMM_SIGHTINGS': paths.input('asamm_sightings'), 'ASAMM_EFFORT': paths.input('asamm_effort'),
            'CORRIDOR_OUT': paths.output('source_data'), 'SURFACES': tree['surfaces'], 'LCP': tree['lcp'],
            'SUMMARIES': tree['summaries'], 'VALIDATION': tree['validation'], 'STATUS': paths.output('qa/stage.json'),
        }):
            check_inputs(paths, module)
            topology = base.load_topology()
            assert not topology.warnings, topology.warnings
            assert len(topology.source_indices) == len(topology.destination_indices) == 40
            sightings, effort = module.load_asamm()
            assert sightings.record_id.nunique() == len(sightings) == 323
            assert sightings.groupby('year').size().to_dict() == {2017: 212, 2018: 111}
            save(paths, 'qa/scope.json', {'years': YEARS, 'scenarios': SCENARIOS, 'design': DESIGN,
                'sightings_by_year': {2017: 212, 2018: 111}, 'prediction_seed': 8675309,
                'stored_draws_per_year': 200, 'used_draws_per_year': 100,
                'draw_selection': 'First 100 stored draws in original order, aligned by grid_id',
                'prediction_refit': False, 'new_random_draws': False})
            print('Pelagic inputs verified; reproducing 72 central ensembles.', flush=True)
            route, _ = module.run_central(topology, sightings, effort)
            assert len(route) == 72 and route.successful_path_count.eq(1600).all() and route.failed_path_count.eq(0).all()
            compare_central(paths, module, checks)
            summarize_central(paths, module, route, tree, checks)
            print('Central results match; propagating the original 100 draws per year.', flush=True)
            uncertain, _ = module.run_uncertainty(topology, sightings, effort)
            assert len(uncertain) == 800 and uncertain.successful_path_count.eq(1600).all() and uncertain.failed_path_count.eq(0).all()
            for folder, name, keys in (
                (tree['summaries'], 'bootstrap_route_metrics_beta025', ['bootstrap_draw'] + CENTRAL_KEYS),
                (tree['validation'], 'bootstrap_asamm_metrics_beta025', ['bootstrap_draw'] + CENTRAL_KEYS),
                (tree['summaries'], 'bootstrap_route_metric_intervals_beta025', ['target_year', 'scenario', 'metric']),
                (tree['validation'], 'bootstrap_asamm_metric_intervals_beta025', ['target_year', 'scenario', 'metric']),
            ):
                compare_output(paths, module, checks, folder / f'phase8s_{name}.csv', keys)
            s12_source_data(paths, module, tree, checks)
    assert paths.snapshot() == before
    maximum = max(value for result in checks for value in result.get('max_numeric_difference', {}).values())
    save(paths, 'qa/acceptance.json', {
        'status': 'passed_archived_input_reproduction', 'task': 'pelagic_support',
        'starting_layer': 'Archived central pelagic support and first 100 stored prediction draws; archived benthic/ice/vessel surfaces and aligned ASAMM',
        'central_ensembles': 72, 'uncertainty_ensembles': 800, 'paths_recomputed': 1395200,
        'central_grid_cells_per_ensemble': 2839, 'sighting_records': 323,
        'comparison_checks_passed': len(checks), 'maximum_numeric_difference': maximum,
        'numeric_atol': 1e-12, 'numeric_rtol': 1e-12,
        'S12_numerical_sources_reproduced': True, 'TableS4_upstream_sources_reproduced': True,
        'central_cell_frequencies_and_path_summary_fields_checked': True,
        'uncertainty_route_and_ASAMM_metrics_checked_for_every_draw': True,
        'uncertainty_grids_have_archived_reference': False,
        'raw_to_prediction_reproduced': False, 'prediction_models_refitted': False,
        'new_random_draws': False, 'raw_benthic_processing_recomputed': False,
        'full_raw_to_paper_reproduction': False, 'figures_redrawn': False,
        'scientific_parameters_changed': False, 'input_hashes_unchanged': True,
    })
    print('Pelagic acceptance passed:', len(checks), 'checks; 72 + 800 ensembles.', flush=True)
