# Package checks

With the software and data archives extracted into sibling `software/` and
`dataset/` directories, run from `software/` in the documented environment:

```sh
python -B -m unittest discover -s tests -p 'test_release_*.py' -v
python -B v8.py prepare-ierp --data-root ../dataset --run-id input_check --plan
```

The tests inspect input routing, fixed parameter contracts, source-data metadata,
version identity, file hashes and output protection. Small synthetic fixtures
exercise comparison guards and rounding; they do not fit models, interpolate
surfaces, calculate study paths, redraw figures or rerun the paper's analyses.
Tests of scientific statements use packaged syntax fingerprints rather than
unpublished development backups. Fingerprints detect changes but are not evidence
that a model is scientifically valid.

The external IERP CSV and separately downloaded GLORYS companion are needed to
execute `prepare-ierp`, but not to run these contract tests. The companion is in
the paired data record; follow [input preparation](IERP_INPUT_PREPARATION.md).
Its `--plan` report identifies missing inputs without
creating outputs. Similarly, the two station-processing tables needed for
`prey_processing` remain external. Tests check that unavailable inputs are
reported and refused, not silently substituted or treated as successfully run.
S11 can consume the accepted preparation receipt through `--prepared-ierp-run`;
its legacy observation table is not included in the data archive.

[Package-check evidence](../environment/package_checks.json) records the tested
release, metadata-only provenance changes and check results. Earlier
[scientific validation reports](ACCEPTANCE.md) describe separate completed runs
and retain their original input identities. Removing personal directory prefixes
from source-provenance fields changes file checksums, not scientific values;
the current task policy and manifests identify the distributed files.
