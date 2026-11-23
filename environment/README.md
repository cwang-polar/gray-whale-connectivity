# Tested environment

Use [CLEAN_INSTALL.md](CLEAN_INSTALL.md). The environment was installed and tested on macOS 14.6.1 ARM64 with CPython 3.10.0. Its scientific packages were installed in a new virtual environment; the interpreter, operating system and system fonts were reused. This was not a clean-machine or cross-platform test.

`requirements-clean-py310-candidate.txt` records the successful installation specification. `requirements-lock-macos-arm64-py310.txt` records the resolved package versions and wheel hashes from that installation. A second new environment was installed directly from the unchanged hash lock on 23 November 2026; dependency checks, imports and read-only entry checks passed, without analysis execution. Neither is a Linux, Windows or Intel Mac compatibility guarantee.

Core versions include NumPy 1.26.4, pandas 2.2.2, SciPy 1.15.2, Matplotlib 3.8.4, GeoPandas 0.13.2, Shapely 2.0.5, rasterio 1.4.4, pyogrio 0.10.0, Fiona 1.9.6 and pyproj 3.7.1 (PROJ 9.5.1). The [dependency inventory](dependency_license_inventory.csv) records installed-package licence metadata.

The earlier runtime used pyproj 3.1.0/PROJ 7.2.0. The compatible release installation was validated against archived results rather than assumed equivalent. Matplotlib used FreeType 2.6.1 and system Arial/Helvetica; native rendering differences are disclosed in [validation](../docs/ACCEPTANCE.md). Fonts and installed libraries are not redistributed.

Pelagic prediction archives retain seed 8675309, float32 precision, 200 saved draws per year and the original draw order; the first 100 were propagated in the validated corridor workflow. Prediction fitting and draw generation were not repeated. Retained upstream modules can require additional dependencies such as h5py/pyreadr that are not in the tested lock; they are not supported by invoking their historical main routines.

See the [quick start](../docs/QUICKSTART.md) for recorded demonstration runtimes and expected output. The new locked installation took 25.53 seconds, excluding Python/OS installation and subsequent checks; the first dependency import check took 42.05 seconds. [Installation evidence](LOCK_INSTALL_VALIDATION.md) records the scope and current host: Apple M1 Pro, 8 logical CPUs, 16 GiB RAM. This is not a peak-memory or minimum-hardware measurement.
