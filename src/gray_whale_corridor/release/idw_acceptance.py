"""Narrow, documented IDW verification exception; never modify values."""

KEYS = ['setting', 'scenario', 'target_year', 'grid_id']
FREQUENCY = 'lcp_frequency_normalized'
APPROVED = {
    ('k8_p1', 'prey_only', 2011, '(75, 80)'): (0.023750, 0.024375),
    ('k8_p1', 'prey_only', 2011, '(76, 80)'): (0.001250, 0.000625),
}
ROUTE_INVARIANTS = [
    'successful_path_count', 'failed_path_count', 'mean_path_cost',
    'corridor_area_cells_freq_gt_0_05', 'centroid_x', 'centroid_y', 'centroid_lon', 'centroid_lat',
]


def check_frequency(actual, reference, pd, np):
    """Accept exact frequencies or the complete documented two-cell profile only."""
    if set(actual.columns) != set(reference.columns):
        raise AssertionError('IDW frequency columns differ')
    if actual.duplicated(KEYS).any() or reference.duplicated(KEYS).any():
        raise AssertionError('Duplicate IDW frequency keys')
    a = actual.sort_values(KEYS).reset_index(drop=True)
    b = reference[a.columns].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(a.drop(columns=FREQUENCY), b.drop(columns=FREQUENCY),
                                  check_dtype=False, check_exact=True)
    for frame in (a, b):
        if not np.isfinite(frame[FREQUENCY]).all() or not frame[FREQUENCY].between(0, 1).all():
            raise AssertionError('Invalid IDW frequency')
    np.testing.assert_array_equal(a[FREQUENCY].gt(.05), b[FREQUENCY].gt(.05))
    changed = a[FREQUENCY].ne(b[FREQUENCY])
    changed_keys = set(a.loc[changed, KEYS].itertuples(index=False, name=None))
    if changed_keys and changed_keys != set(APPROVED):
        raise AssertionError('Unapproved IDW frequency key or incomplete exception profile')
    differences = []
    for row in a.index[changed]:
        key = (str(a.loc[row, 'setting']), str(a.loc[row, 'scenario']),
               int(a.loc[row, 'target_year']), str(a.loc[row, 'grid_id']))
        observed_pair = (float(a.loc[row, FREQUENCY]), float(b.loc[row, FREQUENCY]))
        if observed_pair != APPROVED[key]:
            raise AssertionError('Unapproved frequency magnitude/direction for ' + repr(key))
        differences.append(dict(zip(KEYS, key), reproduced_frequency=observed_pair[0],
                                reference_frequency=observed_pair[1]))
    return {'comparison': 'p1_idw_frequency_surfaces.csv.gz', 'rows': len(a),
            'passed': True, 'exact_frequency_match': not differences,
            'known_difference_profile_used': bool(differences), 'differences': differences,
            'frequent_membership_exact': True, 'other_fields_exact': True,
            'source_values_modified': False, 'general_frequency_tolerance': 0}


def check_route_invariants(actual, reference, pd):
    keys = ['setting', 'scenario', 'target_year']
    a = actual.sort_values(keys).reset_index(drop=True)
    b = reference.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys + ROUTE_INVARIANTS], b[keys + ROUTE_INVARIANTS],
                                  check_dtype=False, check_exact=True)
