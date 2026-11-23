# Locked-environment installation check

Date: 23 November 2026. A new isolated virtual environment was installed from the existing wheel-hash lock. The scientific code, lock, samples, parameters and data were not changed. No analysis, model, figure or scientific test suite was executed in this check.

## System and procedure

- macOS 14.6.1 ARM64; existing CPython 3.10.0; no inherited system site packages.
- A new environment was created with `venv --copies`; pip 25.3 was installed, followed by the unchanged lock with `--require-hashes --only-binary=:all:`.
- The index was `https://pypi.org/simple`; pip's cache was disabled. This is not a guarantee that every intermediate network cache was cold.
- Wheel-lock SHA-256: `4c9ed5fb9b6cd629434922ca3de0c30ef6b518903108a932c054ed711c08fe2c`.
- Installation host: MacBookPro18,3 with Apple M1 Pro, 8 logical CPUs and 16 GiB RAM, confirmed by a separate read-only system query. These are host specifications, not minimum requirements or measured peak use. No GPU was requested or used by these checks.

## Observed timing

| Operation | Seconds |
|---|---:|
| Create isolated environment | 2.13 |
| Install pip 25.3 | 1.55 |
| Install locked dependencies | 21.84 |
| Installation total | 25.53 |
| Check package dependency consistency | 0.26 |
| First imports of the scientific dependency stack | 42.05 |

These are one installation's elapsed times, not performance guarantees. Installation time excludes installing Python or the operating system, the subsequent import check and scientific execution. Network, machine and font/library initialization can change timings. The first import check was slower than installation and must not be mistaken for model runtime.

## Checks passed

- `pip check` reported no broken requirements.
- NumPy, pandas, SciPy, Matplotlib, GeoPandas, pyproj, Shapely, rasterio, pyogrio, Pillow, Fiona, xarray, scikit-learn and netCDF4 imported successfully.
- Entry help and the analysis catalogue loaded.
- The numerical example and Fig. 5 input plans passed exact file-identity checks using the paired archive.
- The output-path preview created no result directory.

The numerical and figure outputs described in [QUICKSTART.md](../docs/QUICKSTART.md) remain evidence from the previous accepted runs. They were not regenerated in this environment. These results establish installation and input inspection on the stated host, not Linux portability or complete raw-data reproduction.
