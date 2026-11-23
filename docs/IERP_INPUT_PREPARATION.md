# IERP observation-input preparation

The source-preparation and S11 entries are validated with the documented auxiliary-distance exception. Preparation reconstructs observation inputs; it does not fit the pelagic models or run corridor analyses.

## Invocation and inputs

From `software/`, with the tested environment installed:

```bash
../environments/v8-review/bin/python -B v8.py prepare-ierp --data-root ../dataset --run-id ierp_review_001 --plan
../environments/v8-review/bin/python -B v8.py prepare-ierp --data-root ../dataset --run-id ierp_review_001
../environments/v8-review/bin/python -B v8.py paths S11 --data-root ../dataset --run-id s11_review_001 --prepared-ierp-run ierp_review_001
../environments/v8-review/bin/python -B v8.py run S11 --data-root ../dataset --run-id s11_review_001 --prepared-ierp-run ierp_review_001
```

The first command checks paths and hashes without creating outputs. The second writes only to a new `../runs/<run-id>/prepare_ierp/` directory. The last two inspect and render S11 using those generated observations; drawing outputs go to the separate new S11 run directory. A missing or changed input stops execution; no original-project fallback is allowed. `config/ncc_v8/ierp_input_preparation.json` registers the required source-preparation inputs:

- The original `Gonzalez_INPUT_env_phyto_zoo_taxa.csv` from the Arctic IERP synthesis package, not its modeled OUTPUT files. Exact source version and acquisition identity are in the [delivery plan](../../dataset/source_metadata/IERP_DELIVERY_PLAN.md).
- The original adopted GLORYS NetCDF subset, `data/raw/ice/cmems_mod_glo_phy_my_0.083deg_P1M-m_1779048578112.nc`, SHA-256 `0684a671d429280163db374299f4185644996136be3d00e8e5c23a5181ee5f8e`. Its June hydrographic predictors are distinct from SIC accessibility; see [source access](../../dataset/DATA_ACCESS.md).
- The fixed ocean grid and two archived pelagic surfaces. Only the saved lower/upper support-scaling anchors are read from those surfaces; predictions are not regenerated.

The three legacy observation tables are neither required nor read by this route. The acceptance configuration stores only their verified file identities, not observation values. S11 also requires its archived prediction map, model-comparison summary, temporal coverage and geographic references from the paired dataset. Thus this is source-to-observation reconstruction plus archived-result rendering, not raw-data-to-model reproduction. `plan S11` without the prepared-run option still describes the backward-compatible archived-table route; use the `paths S11` command above for the external-input route.

Obtain the two external raw inputs from their sources and place them at the registered relative paths; their checksums are verified before processing. The source is Rand et al. (2026), DOI [10.24431/rw1kai8](https://doi.org/10.24431/rw1kai8).

## Processing

The adapter calls the existing `parse_asgard_abundance`, `match_to_grid`, `load_model_data` and `sample_glorys` functions. It retains June 2017/2018; adds the two adopted abundance columns; assigns observations within 15 km of the fixed grid; averages abundance within year-grid groups; adds the original log/rank responses; and samples June GLORYS temperature, salinity and currents. The original CSV handoffs are preserved. The existing observed-support scaling block was extracted into `load_observed_support` and reused without changing its statements.

The three final products are the model-record table, its plotting export and the S11 observed-support export. No candidate fitting, cross-validation, bootstrap, prediction-domain calculation, corridor analysis, whale metric or figure rendering is invoked. This command is not an energy conversion and does not revisit the taxonomic-combination rule.

## Verification outcome

The isolated run retained 53 source rows in June 2017 and 32 in June 2018. Of these, 39 and 28 were assigned within 15 km, producing the same 38 and 28 observation cells as the paper. All three tables have the same schema, order, identities, missing values, abundance, environmental predictors, log/rank values and observed relative support as their references.

Strict acceptance stopped on the auxiliary `grid_match_distance_km` field. Its maximum difference was `2.658762099372325e-12` km; two rows exceeded the existing `atol=rtol=1e-12` check. The closest source observation to the 15-km inclusion boundary was 1.8208 km away, and no assigned cell or inclusion changed. All other parsed fields are exactly equal. The first difference is already present at source-to-grid projection, before hydrographic sampling or plotting export. The change is consistent with floating-point projection-library differences, but that attribution has not been proved by another environment run.

The [comparison report](../environment/ierp_input_verification_v1.json) records the exact auxiliary-distance discrepancy. `config/ncc_v8/ierp_output_acceptance.json` accepts either the complete original three-file profile or the exact validated three-file profile. Mixed, missing or changed files are rejected; no general numeric tolerance was widened. A different platform producing an unregistered profile requires review, even if its differences appear small.

Preparation/S11 verification used a separate 15-file input directory in which all three old observation tables were physically absent. Preparation reproduced the validated three files exactly. S11 consumed the accepted preparation receipt and generated table, preserving its five source exports (the observation export matches the new table), frozen caption and scientific drawing functions. The PNG and SVG-text comparisons are reported in the [new verification summary](../environment/ierp_input_verification_v2.json). Only input preparation and S11 rendering were executed; model fitting, validation, uncertainty generation and corridor calculations were not rerun.

The three observation tables are not redistributed. The two original external inputs remain separately acquired dependencies. The preparation route does not grant permission to rehost raw inputs or regenerated observations. Source credit and applicable use conditions still apply.
