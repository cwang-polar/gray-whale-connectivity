# Software dependencies and notices

This project imports third-party Python/GIS libraries; it does not obtain ownership of them. The project [MIT licence](LICENSE) covers only eligible original project code and documentation, not these dependencies, third-party data or system fonts.

The clean installation produced [a package/version/license inventory](environment/dependency_license_inventory.csv), covering 34 resolved runtime packages plus pip and setuptools. Each row comes from installed wheel metadata. The exact runtime versions and wheel hashes are in `environment/requirements-lock-macos-arm64-py310.txt`; official packages retain their own notice files. This inventory is not a legal certification of every bundled native library.

- NumPy, pandas, SciPy, GeoPandas, Shapely, Fiona, rasterio and several dependencies carry BSD-family notices; inspect each package's own exact notice rather than replacing it with this summary.
- pyproj, pyogrio, netCDF4 and multiple supporting packages have MIT-family declarations; Matplotlib includes its own and third-party notices; Pillow declares MIT-CMU; xarray declares Apache-2.0; certifi declares MPL-2.0. Mixed or missing metadata labels are preserved, not guessed.
- Wheel binaries can include PROJ, GEOS, GDAL, numerical and image libraries with additional notices. **The software archive contains source and requirement files, not the local virtual environment or wheel cache.** Users install official wheels with their notices intact. Any future container, offline wheel bundle or binary distribution needs a separate bundled-notice review.
- Arial and Helvetica are resolved from macOS, not copied or licensed by this project. They must not be included in GitHub or Zenodo. A missing font may change rendering; do not claim pixel-identical cross-platform plots from the current test.
- A text scan of the selected project Python files found no explicit copied-code/license headers or external snippet acknowledgments. That is not proof of authorship or rights. The corresponding author has authorized licence preparation for the original project content; any separately identified reused code must retain its own notice.

See [LICENSE_STATUS.md](LICENSE_STATUS.md) and [AUTHORS.md](AUTHORS.md). The source-only release does not redistribute installed dependency binaries or fonts. Installing dependencies does not require giving the authors' data the software's licence.
