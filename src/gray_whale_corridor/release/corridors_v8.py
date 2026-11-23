"""Archived-input study corridor reproduction; preserve legacy scientific functions."""

from contextlib import contextmanager
import importlib
import json

from .archived_figures import bound_attributes
from .control import CONFIG, SOFTWARE, digest
from .figure_adapters import plotting_imports

VARIABLE = 'total_kJm2_frozen'
PERIODS = {'prey_only': list(range(2011, 2020)), 'prey_ice': list(range(2011, 2020)),
           'prey_vessel': list(range(2015, 2020)), 'prey_ice_vessel': list(range(2015, 2020))}
MODULES = {
    'prey_only': 'run_prey_only_lcp_brian_amphipod_C3_v2',
    'prey_ice': 'run_prey_ice_lcp_brian_amphipod_C3_v2',
    'prey_vessel': 'run_phase8k_prey_vessel_factorial_v1',
    'prey_ice_vessel': 'run_prey_ice_ship_lcp_brian_amphipod_C3_v2',
}


def save(paths, name, obj):
    paths.output(name).write_text(json.dumps(obj, indent=2) + '\n')


def prepare(paths, task):
    paths.require_task(task)
    scope = json.loads((CONFIG / 'scope.json').read_text())
    assert scope['primary']['prey_variable'] == VARIABLE
    assert scope['primary']['paths_per_scenario_year'] == 1600
    assert scope['graph']['numerical_epsilon'] == 0.001
    before = paths.snapshot()
    paths.reserve_output()
    for folder in ('source_data', 'qa'):
        paths.output(folder).mkdir(exist_ok=False)
    return before


def compare_frames(paths, label, actual, expected, keys, pd, np, checks):
    """Check identity and every field; report a mismatch rather than repairing it."""
    a, b = actual.copy(), expected.copy()
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError('Duplicate verification keys: ' + label)
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    result = {'comparison': label, 'rows': len(a), 'reference_rows': len(b),
              'keys': keys, 'numeric_rtol': 1e-12, 'numeric_atol': 1e-12}
    try:
        if set(a.columns) != set(b.columns):
            raise AssertionError(f'Column mismatch: {set(a.columns) ^ set(b.columns)}')
        b = b[a.columns]
        for col in a.columns:
            if a[col].dtype == object or b[col].dtype == object:
                a[col], b[col] = a[col].fillna(''), b[col].fillna('')
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        # Path/cell membership may not be accepted merely on floating tolerance.
        for col in a.columns:
            if col.endswith('_count') or col == 'lcp_count' or col.startswith('corridor_area_cells'):
                np.testing.assert_array_equal(a[col], b[col])
        result['passed'] = True
        result['max_numeric_difference'] = {col: float((a[col] - b[col]).abs().max())
            for col in a.columns if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col])}
    except (AssertionError, ValueError) as error:
        result.update(passed=False, error=str(error))
        save(paths, 'qa/comparisons.json', checks + [result])
        a.to_csv(paths.output('qa/mismatch_actual.csv'), index=False)
        b.to_csv(paths.output('qa/mismatch_reference.csv'), index=False)
        raise RuntimeError('Frozen V8 mismatch; stopped without altering science: ' + label) from error
    checks.append(result)
    save(paths, 'qa/comparisons.json', checks)


def run_sparse_year(module, scenario, surface, topology, year, pd, np):
    """Same per-year operations as each historical main; no plots or old-proxy work."""
    if scenario == 'prey_ice_vessel':
        grid, warning = module.prepare_grid(surface, topology.grid, VARIABLE, year)
        r_col = 'R_log_prey_ice_ship_alpha0_5'
        runner, formatter = module.run_paths, module.format_grid
    else:
        grid, warning = module.prepare_lcp_grid(surface, topology.grid, VARIABLE, year)
        r_col = 'R_log' if scenario == 'prey_only' else 'R_log_prey_ice_beta1'
        runner, formatter = module.run_year_paths, module.format_grid_output
    warnings = list(topology.warnings)
    if warning:
        warnings.append(warning)
    adjacency = module.build_weighted_csr(topology.edge_pairs, grid[r_col].to_numpy())
    counts = np.zeros(len(grid), dtype=int)
    rows, path_warnings = runner(VARIABLE, year, grid, adjacency, topology, counts)
    warnings.extend(path_warnings)
    grid['lcp_count'] = counts
    grid['lcp_frequency_normalized'] = counts / len(rows) if rows else 0.0
    out = formatter(grid, VARIABLE, year)
    attempted = len(topology.source_indices) * len(topology.destination_indices)
    route = module.summarize_route(VARIABLE, year, rows, out, attempted, topology, warnings)
    return out, pd.DataFrame(rows), route


def run_primary(paths):
    before = prepare(paths, 'primary_corridors')
    checks, routes = [], []
    with plotting_imports(paths):
        import numpy as np
        import pandas as pd
        topology_module = importlib.import_module('run_prey_only_lcp_main_south_to_north_v0')
        topology = topology_module.build_graph_topology(pd.read_csv(paths.input('grid')), pd.read_csv(paths.input('endpoints')))
        assert len(topology.source_indices) == len(topology.destination_indices) == 40
        for scenario, years in PERIODS.items():
            module = importlib.import_module(MODULES[scenario])
            expected_routes = pd.read_csv(paths.input('route_' + scenario))
            expected_routes = expected_routes.loc[expected_routes.prey_variable.eq(VARIABLE)]
            expected_paths = pd.read_csv(paths.input('paths_' + scenario))
            expected_paths = expected_paths.loc[expected_paths.prey_variable.eq(VARIABLE)]
            for year in years:
                if scenario == 'prey_vessel':
                    with bound_attributes(module, {'SURFACE_OUT': paths.input(f'surface_{scenario}_{year}').parent}):
                        grid, path_frame, route = module.run_year(year, topology)
                else:
                    surface = pd.read_csv(paths.input(f'surface_{scenario}_{year}'))
                    grid, path_frame, route = run_sparse_year(module, scenario, surface, topology, year, pd, np)
                stem = f'{scenario}_{year}'
                grid.to_csv(paths.output('source_data/' + stem + '_grid.csv'), index=False)
                path_frame.to_csv(paths.output('source_data/' + stem + '_paths.csv'), index=False)
                if route['successful_path_count'] != 1600 or route['failed_path_count'] != 0:
                    raise RuntimeError('Path completion differs: ' + stem)
                compare_frames(paths, stem + '_grid', grid, pd.read_csv(paths.input(f'grid_{scenario}_{year}')),
                               ['grid_id'], pd, np, checks)
                compare_frames(paths, stem + '_paths', path_frame, expected_paths.loc[expected_paths.target_year.eq(year)],
                               ['source_grid_id', 'destination_grid_id'], pd, np, checks)
                compare_frames(paths, stem + '_route', pd.DataFrame([route]), expected_routes.loc[expected_routes.target_year.eq(year)],
                               ['target_year'], pd, np, checks)
                routes.append({'scenario': scenario, **route})
                pd.DataFrame(routes).to_csv(paths.output('source_data/primary_route_summaries.csv'), index=False)
                print('Verified primary:', scenario, year, '1600 paths', flush=True)
        assert len(routes) == 28 and len(checks) == 84
    finish(paths, before, checks, {'scenario_years_recomputed': 28, 'paths_recomputed': 44800,
        'layer': 'Archived primary utility surfaces to corridor solutions; no raw-data preprocessing',
        'alternate_prey_currencies_recomputed': False})


def require_primary_acceptance():
    path = SOFTWARE / 'environment/runtime_acceptance_corridors_v1.json'
    if not path.is_file():
        raise PermissionError('Primary corridor acceptance is required before sensitivity execution')
    report = json.loads(path.read_text())
    if report.get('tasks', {}).get('primary_corridors', {}).get('status') != 'passed_archived_input_reproduction':
        raise PermissionError('Primary corridor baseline has not passed')
    for relative, expected in report['primary_scientific_code_sha256'].items():
        if digest(SOFTWARE / relative) != expected:
            raise PermissionError('Primary scientific implementation changed after acceptance: ' + relative)


@contextmanager
def sensitivity_bindings(paths, base):
    def stage(name, state, **details):
        if name != 'p3':
            raise ValueError('Only the adopted vessel branch may write status')
        save(paths, 'qa/stage.json', {'stage': name, 'state': state, **details})
    values = {
        'ROOT': paths.data_root, 'GRID': paths.input('grid'), 'ENDPOINTS': paths.input('endpoints'),
        'prey_path': lambda year: paths.input(f'prey_{year}'),
        'ice_path': lambda year: paths.input(f'ice_{year}'),
        'ship_path': lambda year: paths.input(f'ship_{year}'),
        'primary_grid_path': lambda scenario, year: paths.input(f'grid_{scenario}_{year}'),
        'primary_route_path': lambda scenario: paths.input('route_' + scenario),
        'SUMMARY_OUT': paths.output('source_data'), 'LCP_OUT': paths.output('source_data'),
        'SURFACE_OUT': paths.output('source_data'), 'QA_OUT': paths.output('qa'), 'set_stage': stage,
    }
    with bound_attributes(base, values):
        yield


def compare_outputs(paths, base, items, checks):
    for name, alias, keys in items:
        compare_frames(paths, name, base.pd.read_csv(paths.output('source_data/' + name)),
                       base.pd.read_csv(paths.input(alias)), keys, base.pd, base.np, checks)


def run_ice(paths):
    require_primary_acceptance()
    before = prepare(paths, 'sensitivity_ice')
    checks = []
    with plotting_imports(paths):
        base = importlib.import_module('run_phase8t_primary_robustness_smoke_test')
        joint = importlib.import_module('run_phase8t_joint_ice_anchor_candidate_v2')
        assert list(joint.SETTINGS.values()) == [(0.10, 0.45), (0.15, 0.55), (0.20, 0.65)]
        with sensitivity_bindings(paths, base), bound_attributes(joint, {
            'OUT': paths.output('source_data'),
            'protected_files': lambda: [paths.input_path(p) for p in paths.contract['inputs']],
        }):
            joint.main()
        keys = ['setting', 'scenario', 'target_year']
        compare_outputs(paths, base, [
            ('joint_ice_anchor_route_metrics.csv', 'reference_route_metrics', keys),
            ('joint_ice_anchor_frequency_surfaces.csv.gz', 'reference_frequency_surfaces', keys + ['grid_id']),
            ('joint_ice_anchor_prey_comparison_metrics.csv', 'reference_prey_comparison_metrics', ['setting', 'target_year']),
            ('joint_ice_anchor_accessibility_summary.csv', 'reference_accessibility_summary', ['setting', 'target_year']),
        ], checks)
    finish(paths, before, checks, {'scenario_year_rows': 42, 'alternative_scenario_years_recomputed': 28,
        'primary_scenario_years_reused_as_in_original': 14,
        'layer': 'Archived prey and monthly SIC/modifiers; original joint-anchor workflow'})


def run_vessel(paths):
    require_primary_acceptance()
    before = prepare(paths, 'sensitivity_vessel')
    checks = []
    with plotting_imports(paths):
        base = importlib.import_module('run_phase8t_primary_robustness_smoke_test')
        assert [v['alpha'] for v in base.VESSEL_SETTINGS.values()] == [0.25, 0.5, 0.75]
        with sensitivity_bindings(paths, base):
            base.p3_vessel(base.load_topology())
        keys = ['setting', 'scenario', 'target_year']
        compare_outputs(paths, base, [
            ('p3_vessel_route_metrics.csv', 'reference_route_metrics', keys),
            ('p3_vessel_frequency_surfaces.csv.gz', 'reference_frequency_surfaces', keys + ['grid_id']),
            ('p3_vessel_comparison_metrics.csv', 'reference_comparison_metrics', ['setting', 'target_year', 'scenario_a', 'scenario_b']),
            ('p3_vessel_modifier_summary.csv', 'reference_modifier_summary', ['setting', 'target_year']),
        ], checks)
    finish(paths, before, checks, {'scenario_year_rows': 30, 'alternative_scenario_years_recomputed': 20,
        'primary_scenario_years_reused_as_in_original': 10,
        'layer': 'Archived primary prey, ice and scaled vessel exposure; original alpha workflow'})


def finish(paths, before, checks, details):
    if paths.snapshot() != before:
        raise RuntimeError('Frozen task inputs changed')
    save(paths, 'qa/acceptance.json', {'task': paths.task, 'status': 'passed_archived_input_reproduction',
        'input_hashes': before, 'checks_passed': len(checks), 'checks': checks, **details,
        'raw_benthic_read': False, 'scientific_parameters_changed': False,
        'full_raw_to_paper_reproduction': False, 'new_experiment': False})
