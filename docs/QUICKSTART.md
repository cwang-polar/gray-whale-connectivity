# Minimal reproduction example

## Purpose and prerequisites

The numerical example recalculates the annual Pearson correlations and leave-one-year-out ranges reported in Supplementary Fig. S8 from archived annual whale rates and scenario costs. It retains every adopted year, including 2019. It does not rerun corridor models, reconstruct survey distances or calculate the three spatial correspondence metrics.

Install the [tested environment](../environment/CLEAN_INSTALL.md). Place the paired archives in sibling `software/` and `dataset/` directories. All inputs for this example are included in the data archive; no unpublished benthic table, IERP download, network API, GPU or manuscript file is needed during execution.

## Run the numerical example

From `software/`:

```bash
../environments/v8-review/bin/python -B v8.py plan annual_correlation --data-root ../dataset --hashes
../environments/v8-review/bin/python -B v8.py run annual_correlation --data-root ../dataset --run-id demo_001
```

The plan command checks registered input identities without scientific execution. The run command uses unchanged settings and creates `../runs/demo_001/annual_correlation/`. Choose a new run ID for a subsequent execution; existing task directories are refused. Historical identifiers in entry/configuration filenames are retained for compatibility, not as a required manuscript version.

## Expected numerical output

Under the new run directory:

| Output | Expected content |
|---|---|
| `source_data/available_period_correlations.csv` | Four rows, one for each scenario and its available time period |
| `source_data/leave_one_year_out_correlations.csv` | 28 omitted-year correlations across those four periods |
| `source_data/frozen_all_scenario_costs.csv` | 28 scenario-year cost rows |
| `source_data/fig5_panel_C_annual_data.csv` | The archived annual comparison used for Fig. 5C |
| `qa/run_manifest.json` | Recorded input hashes and `correlations_match_frozen: true` |
| `qa/execution.json` | Execution and access-check record |

Expected values below are rounded for reading only; the CSV retains full precision:

| Scenario | Years | Pearson r | Omitted-year minimum | Omitted-year maximum |
|---|---|---:|---:|---:|
| Prey | 2011-2019 | -0.767680 | -0.807508 | -0.698040 |
| Prey+ice | 2011-2019 | -0.795189 | -0.860542 | -0.746034 |
| Prey+vessel | 2015-2019 | 0.161842 | -0.112700 | 0.808023 |
| Prey+ice+vessel | 2015-2019 | -0.295889 | -0.977351 | 0.029169 |

The ranges are omitted-year extrema, not confidence intervals. The full-precision archived reference is [available_period_correlations.csv](https://doi.org/10.5281/zenodo.24479974) (`dataset/outputs/manuscript_figures_v3/supplementary_figures_working/available_period_S8_20261117_v2/source_data/available_period_correlations.csv`).

## Optional figure example

```bash
../environments/v8-review/bin/python -B v8.py plan Fig5 --data-root ../dataset --hashes
../environments/v8-review/bin/python -B v8.py run Fig5 --data-root ../dataset --run-id figure_demo_001
```

Outputs are under `../runs/figure_demo_001/Fig5/figures/Fig5/`:

- `candidates/gray_whale_figure5_spatial_annual_candidate_v1.png`, `.pdf` and `.svg`;
- `source_data/`: the four annual, correlation, cost-band and spatial-metric CSV exports;
- `caption.md` and `qa/gray_whale_figure5_spatial_annual_candidate_v1_checks.json`.

The figure uses stored values: panels A-B show spatial correspondence, whereas C shows annual cost/whale-rate comparisons. The accepted PNG was 4322 by 4251 pixels and the SVG contained 98 editable text elements. Fonts can cause small rendering differences; scientific values, labels and comparisons must remain unchanged. Figure export alone is not numerical or raw-data reproduction.

## Recorded runtime and limits

On 18 November 2026, the numerical example completed in **1.62 seconds** and the Fig. 5 export in **3.18 seconds**, measured around each full task subprocess in the tested macOS 14.6.1 ARM64/CPython 3.10.0 environment. These are individual recorded runs, not averages or timing guarantees. Hardware model, RAM and peak memory were not recorded. Font caches, disk and platform can affect elapsed time. A later locked installation took 25.53 seconds, and its first dependency import check took 42.05 seconds. Neither timing is a fresh measurement of the demonstrations. See [installation details](../environment/LOCK_INSTALL_VALIDATION.md).

Validation results and scientific limits are summarized in [ACCEPTANCE.md](ACCEPTANCE.md).

## Troubleshooting

- **Hash or missing-input error:** use the exact paired archive and documented external-input route; do not disable checks or substitute similarly named files.
- **Output already exists:** select a new run ID instead of deleting a previous result.
- **Unavailable wheel:** the lock is platform-specific; other platforms remain unverified.
- **S11 missing observation input:** follow [IERP_INPUT_PREPARATION.md](IERP_INPUT_PREPARATION.md); S11 is not the self-contained demonstration.
- **Prey preparation missing station tables:** see the [external benthic input contract](https://doi.org/10.5281/zenodo.24479974) (`dataset/source_metadata/BENTHIC_SURFACE_INPUTS.md`); archived-surface downstream analyses remain usable.
- **Unexpected numerical change:** retain the output and execution record and contact cwang@umces.edu with the version, environment and task. Do not replace reference values to obtain a pass.
