# Gray whale connectivity: analysis software

Python workflows for the Pacific Arctic gray whale connectivity study: least-cost corridors, seasonal and parameter sensitivities, observation comparisons, and manuscript figures and tables. Reproduction starts at the documented archived or external inputs, rather than a complete raw-observation-to-paper pipeline.

Version **1.0.0**. Cite the [versioned software archive](https://doi.org/10.5281/zenodo.24479214) and [supporting dataset](https://doi.org/10.5281/zenodo.24479974) separately. GitHub provides browsable source; the archives identify the fixed files used for reproduction. Source datasets should also receive their own citations.

[Repository](https://github.com/cwang-polar/gray-whale-connectivity) | [Installation](environment/CLEAN_INSTALL.md) | [Quick start](docs/QUICKSTART.md) | [Validation](docs/ACCEPTANCE.md) | [Reviewer guide](docs/REVIEWER_GUIDE.md) | [Data access](https://doi.org/10.5281/zenodo.24479974) (`dataset/DATA_ACCESS.md`)

## Installation and first use

Unpack the two archives into sibling `software/` and `dataset/` directories. Follow the tested [installation instructions](environment/CLEAN_INSTALL.md), then run the [small numerical demonstration](docs/QUICKSTART.md). It requires only archived inputs included in the paired dataset. An optional Fig. 5 example demonstrates figure export. A second isolated installation from the unchanged wheel-hash lock passed; [timing and entry-check evidence](environment/LOCK_INSTALL_VALIDATION.md) are available. For scientific model execution, use the separate [reviewer guide](docs/REVIEWER_GUIDE.md).

From `software/`:

```bash
../environments/v8-review/bin/python -B v8.py list
../environments/v8-review/bin/python -B v8.py plan annual_correlation --data-root ../dataset --hashes
../environments/v8-review/bin/python -B v8.py run annual_correlation --data-root ../dataset --run-id demo_001
```

Commands and filenames retain historical identifiers for compatibility; these are not manuscript version requirements. Inspect `list` for individual task names. `plan` and `paths` inspect inputs without scientific calculations. `run` executes one specified task using fixed parameters. Outputs go to a new `../runs/<run-id>/<task>/` directory; existing task directories are refused. Input hashes are checked and missing files do not fall back to the author's project. Do not run historical script `main()` functions directly.

## What can be reproduced

| Analysis or display | Starting input | Validation scope |
|---|---|---|
| Main figures and Supplementary Figs. S1-S13 | Archived figure sources and geographic references; S11 requires the preparation step below | Source values, captions and graphical content checked; small rendering differences documented |
| Tables S1-S5 | Stored numerical summaries and conversion coefficients | All displayed cells matched the paper reference; Table S5 formats coefficients rather than converting energy |
| Annual whale rates and correlations | Exact ASAMM event/flight-summary CSV exports, or archived annual rates for the correlation-only entry | Survey distance, reported whale counts, annual rates, Pearson correlations and omitted-year results matched |
| Spatial correspondence and survey-year sensitivity | Aligned sightings/effort and archived annual corridors | Record membership, pooled metrics and omitted-year results matched |
| Primary corridors and ice/vessel sensitivities | Archived primary prey utility and environmental inputs | Path, grid and summary outputs matched their recorded references |
| August-October connectivity | Archived prey/seasonal modifiers and ASAMM event CSVs | Utilities, paths, observation membership and spatial summaries matched |
| Pelagic-support sensitivity | Archived support predictions and ordered prediction draws | Integration into corridors and observation metrics matched; model fitting and draw generation are separate upstream stages |
| Prey interpolation sensitivity | Archived five-year observed-cell means | Reported summaries and frequent-cell sets matched; a documented two-cell frequency exception remains |
| Primary prey-surface preparation | External station-year energy totals and saved coordinate matches | Annual surfaces/scaling matched; initial provider-record preparation is not reconstructed |

See [validation and exceptions](docs/ACCEPTANCE.md), [upstream implementation coverage](docs/UPSTREAM_COVERAGE.md), the [analysis specification](docs/v8_analysis_contract.md), and the paired [figure/table source index](https://doi.org/10.5281/zenodo.24479974) (`dataset/source_data/v8_figure_table_index.csv`). These starting layers must not be described as complete raw-data reproduction.

## External inputs

- **Benthic source:** the [DBO portal](https://arcticdata.io/catalog/portals/DBO/Data) identifies the source collection. Original station observations, station-year energy totals and coordinate-matching tables are external inputs, and their exact public file location is not established by that collection link. Their absence does not prevent downstream reproduction from the archived surfaces. See the [input specification](https://doi.org/10.5281/zenodo.24479974) (`dataset/source_metadata/BENTHIC_SURFACE_INPUTS.md`).
- **Supplementary Fig. S11:** use the [IERP preparation instructions](docs/IERP_INPUT_PREPARATION.md) to generate its observation input from the specified external abundance CSV and unchanged GLORYS companion, supplied separately in the paired data record. The three observation-bearing tables are not redistributed. Other analyses use the archived predictions and results specified in their input guides.
- The [data-access guide](https://doi.org/10.5281/zenodo.24479974) (`dataset/DATA_ACCESS.md`) gives source versions, fields, checksums, licences and limits. Do not substitute a different source version merely because its columns match.

## Interpretation and reuse

Spatial sighting correspondence and annual whale-rate correlations are distinct analyses. The [analysis specification](docs/v8_analysis_contract.md) defines their samples, reference regions and parameters. Parameter ranges and omitted-year ranges are not confidence intervals. Numerical and rendering exceptions are disclosed in the validation summary.

Original software/documentation use [MIT](LICENSE). Third-party packages and source data retain their own terms. See [third-party notices](THIRD_PARTY_NOTICES.md), [citation metadata](CITATION.cff), and [AI assistance and authorship](AUTHORS.md). Questions or reproducible error reports: Chao Wang, cwang@umces.edu; include the software version, task, environment and error message, without private source data.
