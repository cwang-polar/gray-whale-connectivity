# Guide to inspecting and reproducing results

Start with the [installation guide](../environment/CLEAN_INSTALL.md) and [minimal demonstration](QUICKSTART.md). The paired [figure/table index](https://doi.org/10.5281/zenodo.24479974) (`dataset/source_data/v8_figure_table_index.csv`) identifies the paper item, panel, stored source and rendering implementation. Historical filenames in that index do not change the paper's figure numbering.

## Choose the relevant layer

| Question | Entry or evidence | What it demonstrates |
|---|---|---|
| Can the package load and find exact example inputs? | `v8.py plan annual_correlation --data-root ../dataset --hashes` | Input identities only; not scientific execution |
| Can annual correlations be recalculated? | `annual_correlation` | Pearson and omitted-year ranges from archived annual values; not spatial matching |
| Can spatial comparisons be recalculated? | `spatial_matching`, `sensitivity_survey_year` | Existing sighting membership and adopted spatial metrics from aligned observations and corridors |
| Can the core corridor model be rerun? | `primary_corridors` | The four scenarios at their adopted years from archived utilities, fixed grid and endpoints |
| Can the reported sensitivities be rerun? | `sensitivity_ice`, `sensitivity_vessel`, `sensitivity_idw`, `late_season`, `pelagic_support` | The fixed adopted calculations from their documented starting layers, not additional parameter experiments |
| Can the figures and tables be regenerated? | `Fig1` through `Fig6`, `S1` through `S13`, `TableS1` through `TableS5` | Rendering/export from registered sources; S11 requires the preparation route below |
| Can the prey or pelagic inputs be prepared? | `prey_processing`; `prepare-ierp` | External station-checkpoint-to-prey surfaces or source-to-observation preparation, respectively; not an unsupported raw-to-paper claim |

## Numerical model example

From `software/`, after installation:

```bash
../environments/v8-review/bin/python -B v8.py plan primary_corridors --data-root ../dataset --hashes
../environments/v8-review/bin/python -B v8.py run primary_corridors --data-root ../dataset --run-id primary_review_001
```

This explicitly runs the model and is more substantial than the quick demonstration. The paired data archive supplies its registered inputs. It writes per-year grid and path CSVs plus `source_data/primary_route_summaries.csv` under `../runs/primary_review_001/primary_corridors/`, with checks under `qa/`. The primary configuration has 1,600 endpoint-pair paths per scenario-year; the run checks those results against the original references. See [corridor input definitions](https://doi.org/10.5281/zenodo.24479974) (`dataset/source_metadata/CORRIDOR_REPRODUCTION_INPUTS.md`) and [acceptance](ACCEPTANCE.md) for the precise boundary.

Each other entry follows the same explicit `plan` then `run` pattern and fresh output-directory rule. Do not run all historical scripts or change the configuration to obtain agreement. The reference outputs are for comparison, not replacements for missing computed outputs.

## External-input exceptions

- **S11:** follow [IERP_INPUT_PREPARATION.md](IERP_INPUT_PREPARATION.md), including the separate GLORYS companion download from the paired data record and the external IERP CSV. `plan S11` alone describes the legacy observation-table route; use `paths S11 ... --prepared-ierp-run ...` for the supported preparation route.
- **Prey preparation:** the two observation checkpoints are not bundled. Their [schema and provenance boundary](https://doi.org/10.5281/zenodo.24479974) (`dataset/source_metadata/BENTHIC_SURFACE_INPUTS.md`) are explicit. Their absence does not stop downstream numerical work using archived surfaces.
- The [upstream coverage table](UPSTREAM_COVERAGE.md) separates source code that is retained from processing that has actually been validated in this release.

## Interpreting success and reporting concerns

Successful file export alone does not establish numerical equivalence or ecological validity. Read the task's `qa/` outputs and [documented exceptions](ACCEPTANCE.md). Fonts may differ across systems; source values, categories and labels may not be silently changed. Linux execution has not been verified.

Report reproducible failures to cwang@umces.edu with software version, task, platform and error text, excluding private source data.
