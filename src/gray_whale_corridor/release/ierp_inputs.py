"""Verified file identities for locally reconstructed IERP tables."""
import json
import re

from .control import CONFIG, confined_path, digest


def registry():
    policy = json.loads((CONFIG / 'ierp_input_preparation.json').read_text())
    path = CONFIG / 'ierp_output_acceptance.json'
    if digest(path) != policy['output_acceptance_sha256']:
        raise ValueError('IERP output acceptance metadata changed')
    return json.loads(path.read_text())


def check_outputs(outputs):
    expected = registry()['outputs']
    if set(outputs) != set(expected):
        raise ValueError('Expected the three IERP export stages')
    checks, profiles = {}, set()
    for key, path in outputs.items():
        spec = expected[key]
        if path.name != spec['filename'] or not path.is_file():
            raise ValueError('Missing or renamed prepared IERP output: ' + key)
        value = digest(path)
        matches = [name for name, known in spec['accepted_sha256'].items() if value == known]
        if len(matches) != 1:
            raise ValueError('Unapproved IERP output identity; no value repair or fallback: ' + key)
        profiles.add(matches[0])
        checks[key] = {'filename': path.name, 'sha256': value, 'accepted_profile': matches[0]}
    if len(profiles) != 1:
        raise ValueError('Mixed IERP export profiles are not accepted')
    return checks


def prepared_files(runs_root, run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id):
        raise ValueError('Invalid prepared IERP run ID')
    root = confined_path(runs_root, run_id + '/prepare_ierp')
    output_specs = registry()['outputs']
    outputs = {key: confined_path(root, 'source_data/' + spec['filename']) for key, spec in output_specs.items()}
    checks = check_outputs(outputs)
    receipt_path = confined_path(root, 'qa/acceptance.json')
    receipt = json.loads(receipt_path.read_text())
    policy = json.loads((CONFIG / 'ierp_input_preparation.json').read_text())
    expected_inputs = {'data:' + p: spec['sha256'] for p,spec in policy['files'].items()}
    if receipt.get('status') not in ('passed_exact_original_output', 'passed_with_documented_auxiliary_distance_exception'):
        raise ValueError('IERP preparation has no accepted receipt')
    if receipt.get('outputs') != checks or receipt.get('inputs') != expected_inputs:
        raise ValueError('IERP receipt and current input/output identities disagree')
    if receipt.get('observation_cells_by_year') != {'2017': 38, '2018': 28}:
        raise ValueError('Unexpected prepared IERP sample inventory')
    return {**outputs, 'receipt': receipt_path}
