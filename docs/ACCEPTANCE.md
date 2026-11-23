# Reproduction validation

This summary describes completed validation of the paper's analyses in the recorded macOS ARM/Python 3.10 environment. Validation is stated by starting input and result.

## Completed checks

| Scientific scope | Existing evidence and result |
|---|---|
| Primary prey surfaces | Existing station-year totals and coordinate matches produced nine primary surfaces and scaling summaries; values agreed within the recorded 1e-12 tolerances and flags/counts matched. [Report](../environment/runtime_acceptance_benthic_surfaces_v1.json) |
| Primary corridors | Archived utility landscapes to 28 scenario-years and 44,800 paths; grid, path and route comparisons passed. [Report](../environment/runtime_acceptance_corridors_v1.json) |
| Ice and vessel sensitivities | Adopted paired ice anchors and vessel alpha values reproduced their archived comparisons; primary baselines were verified separately. [Report](../environment/runtime_acceptance_corridors_v1.json) |
| August-October analysis | Stored environmental inputs and ASAMM CSV exports to utilities, paths and spatial comparisons; samples and reported metrics matched. [Report](../environment/runtime_acceptance_late_season_v1.json) |
| Pelagic-support integration | Saved predictions/draws to central and uncertainty corridor/observation summaries; comparison tables passed. This did not refit the spatial models. [Report](../environment/runtime_acceptance_pelagic_v1.json) |
| Interpolation sensitivity | Archived observed-cell checkpoints to the adopted IDW comparisons; accepted with the specific two-cell exception below. [Report](../environment/runtime_acceptance_idw_v2.json) |
| Annual observation rates/correlations | Exact ASAMM CSV exports or archived annual inputs to reported rates, Pearson correlations and annual omissions. [Report](../environment/runtime_acceptance_clean_v1.json) |
| Spatial observation metrics | Aligned records/effort and archived corridor grids to pooled metrics and survey-year omissions. [Report](../environment/runtime_acceptance_clean_v1.json) |
| Figures and tables | Archived-source values, table display cells, figure text/captions and scientific content checked. [Initial figure/table report](../environment/runtime_acceptance_clean_v1.json), [Fig. 2/3](../environment/runtime_acceptance_archived_figures_v1.json), [Fig. 1/S1/Table S5](../environment/runtime_acceptance_archived_context_v1.json) |
| Fig. 1 | Panel labels and archived sources checked. [Report](../environment/runtime_acceptance_fig1_labels_v3.json) |
| External IERP preparation and S11 | Observation input regenerated and used without the three legacy observation tables; non-distance fields matched. [Report](../environment/ierp_input_verification_v2.json) |

## Numerical and rendering exceptions

- **IDW:** in 2011, prey, k=8 and power=1, two grid frequencies differ by one path out of 1,600. Both cells remain below the frequent-corridor threshold. Fig. 2C values/ranges, frequent-cell sets and centroids match. The exact differing keys and magnitudes are retained in the [accepted report](../environment/runtime_acceptance_idw_v2.json); the [initial stopped comparison](../environment/runtime_acceptance_idw_v1.json) remains available. No values were replaced and general tolerances were not widened.
- **Coordinate assignment:** the late-season observation-to-grid distance differs by at most 5.526246127374179e-12 km; its separate comparison tolerance is 1e-10 km. IERP's maximum auxiliary-distance difference is 2.658762099372325e-12 km, accepted only as the recorded file profile. Record identities, cell assignment, inclusion and reported metrics were unchanged. These auxiliary distances are not sighting-to-corridor distances.
- **Table S4:** stored 0.6625 displays as 0.663 using decimal half-up rounding; the source value is unchanged.
- **Graphics:** font/native-library differences produce small pixel or size changes. Inspected outputs retain scientific content and checked source values/text. This is not pixel-identical validation. PDFs passed container checks, not independent page-by-page rendering.

## Limits

Primary prey preparation begins at station totals/matches, not original provider records. Primary models begin at archived surfaces; pelagic integration begins at saved predictions and draws. The [upstream coverage table](UPSTREAM_COVERAGE.md) identifies raw environmental extraction, early ASAMM alignment, pelagic fitting/CV/draw generation and alternate-prey-currency solver layers that were not independently rerun through dedicated release entries. The relevant source implementations are retained where identified.

Annual and late-season observation readers start at CSV exports, not Access databases. Source access is distinct from numerical acceptance. The [data-access guide](../../dataset/DATA_ACCESS.md) explains exact external dependencies; an input specification does not make an unavailable table public.

The generated wheel-hash lock passed a second isolated macOS ARM64 installation and read-only entry check on 23 November 2026; analyses were not rerun in that environment. Linux, Windows and Intel Mac execution remain unverified. Recorded control tests passed, but no unfamiliar-user installation trial is documented. See [installation](../environment/CLEAN_INSTALL.md) and [demonstration](QUICKSTART.md).

Machine-readable reports record comparison results and documented numerical exceptions. They do not establish complete upstream reproduction.
