# Analysis specification

This document specifies the fixed scientific settings for the paper's reproduction entries. Scope constraints are in `config/ncc_v8/scope.json` (relative to the software root). Original full-precision inputs and retained implementations govern numerical reproduction. See [validation](ACCEPTANCE.md) for verified starting layers and exceptions.

## Preserve the implemented model

Use the recurrent sampling-support study domain, 10-km grid, and the same fixed 40-by-40 endpoints. Do not substitute the old default 5-km/proxy configuration. Benthic prey uses the original complete trailing five-year windows and fixed pooled scaling, with energy as the primary representation. Do not recompute scaling from manuscript-rounded values.

Keep monthly ice and vessel transformations before their seasonal averages. Retain adopted v1 ice missing-value handling and the harmonization-grid pooled AIS reference. August-October uses the same prey surfaces, graph/endpoints and primary May-July AIS scaling reference; a diagnostic late-season q99 is not the primary reference.

Retain the original branch-specific least-cost-path implementations. Their graph construction, numerical utility floor, resistance transformation, node order, edge weights and tied-path behavior are part of the reproduction contract. A cleanup must not silently replace heap-based paths with a different sparse solver or use saved figure colors to infer numerical values.

## Preserve the distinction between observation analyses

| Property | Spatial comparison: Fig. 5A-B | Annual comparison: Fig. 5C and S8 |
|---|---|---|
| Observation unit | Retained sighting record | Matched annual reported whale total and transect effort |
| Main dates | June-July, 2011-2019 or common 2015-2019 | June-July, each scenario's available annual range |
| Inputs | Record-cell mapping, annual strict-effort backgrounds, annual corridor frequencies | Flight/event identifiers, GPS segments, group counts, flight effort, annual model costs |
| Statistics | Mean effort-footprint percentile; top-5% overlap; median nearest-frequent-corridor distance | Annual whales/100 transect km; Pearson correlation; omitted-year range |
| Ranges | Omit one survey year and recompute the same pooled spatial metrics | Ice-anchor cost sensitivity in 5C; omit one annual pair and recompute Pearson in S8 |
| Spatial domain | Percentile uses effort background; other targets remain regional | Original matched transect regional footprint; not cell-count effort |

Do not substitute one sample set for the other. Preserve 2019. Do not infer a confidence interval from min-max parameter sensitivity or omitted-year values. Annual rates are ratios of summed counts to summed effort, not averages of flight-level rates.

## Preserve the used sensitivity subset

- IDW: (k,p)=(8,2),(8,1),(8,3),(4,2).
- Ice: paired lower/upper anchors 0.10/0.45, 0.15/0.55 and 0.20/0.65. Do not substitute the older upper-anchor-only experiment.
- Vessel: alpha 0.25, 0.50 and 0.75.
- Spatial definitions and recurrence/exposure selection: use the exact saved Supplementary Figs. S7 and S10 specifications, including original tie handling. Phase8U is excluded.
- Prey currency: use adopted scenario/year/currency outputs for S13; do not promote them to a new primary analysis.
- Pelagic: observed 2017 and 2018 only; selected models/gates and supported domains unchanged. Keep the stored prediction-draw order and the first 100 corridor draws. No energy weighting, new data integration, or extrapolation.

## Reproduction levels and their current status

1. Archived source values to figures/tables: all 19 figures and five tables completed the documented checks. Live manuscript files are not inputs. Drawing acceptance does not resolve every originating-source or manuscript-provenance question.
2. Frozen inputs to adopted comparisons: four numerical entries completed archived-value checks. Spatial comparisons start with aligned observations and archived corridors; annual rates start with eight exact CSV exports. Input identities and output boundaries remain enforced.
3. Archived model inputs to corridor solutions: primary energy-density, joint ice-anchor, vessel-alpha, late-season and pelagic integration workflows passed their stated checks. Late-season utilities are reconstructed from stored prey and seasonal modifiers; its spatial validation starts with four ASAMM event CSV exports. Pelagic integration starts at central support predictions and saved draws, uses the original three operators/beta settings, and reproduces 323-record spatial metrics and S12/Table S4 sources. Raw environmental extraction is outside this boundary.
4. IDW from archived observed-cell checkpoints: accepted under the documented two-cell frequency exception. Fig. 2C costs/counts/ranges, frequent-cell sets and centroids match exactly. Both original values and the initial strict failure are retained. Verification accepts exact equality or the complete recorded k8_p1/Prey/2011 two-cell profile, not an expanded tolerance. Other differences still fail.
5. Raw-data end-to-end reproduction: not validated. Initial benthic processing, pelagic prediction fitting/CV/draw generation and source-access closure remain unresolved. Database-to-CSV equivalence is unverified; ASAMM entries start at the specified CSV exports.

Input and output tables retain full precision. The documented selectors, row identifiers and hashes identify the values used in each analysis.

## Validation boundaries

Existing checks cover input identity, adopted comparison samples and outputs, table display cells, figure content and the archived-input model workflows described in the acceptance report. Model checks include endpoints, path completion, cell-level frequencies, costs and branch-specific solver outputs; late-season path sequence hashes are also checked. Pelagic checks cover central grids/path summaries and every draw's stored route/ASAMM fields; unarchived uncertainty grids and path geometries cannot be compared. These checks do not validate all upstream preprocessing or alternate-currency solver reruns. All execution outputs must use a fresh directory, never original project outputs.
