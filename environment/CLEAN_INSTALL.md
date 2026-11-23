# Installation

## Tested system

- macOS 14.6.1, ARM64; CPython 3.10.0.
- A new virtual environment without inherited system site packages.
- Existing system Arial/Helvetica fonts. No GPU is required by the documented demonstration.
- Package versions and wheel hashes: `requirements-lock-macos-arm64-py310.txt`.
- The new locked-installation host was an Apple M1 Pro MacBook Pro with 8 logical CPUs and 16 GiB RAM. This is not a minimum-hardware requirement; peak memory was not measured.

Linux, Windows and Intel Mac have not been validated. The wheel-hash lock selects macOS ARM/CPython 3.10 wheels, some requiring macOS 14 or later; do not treat it as portable to another platform.

## Install into a new environment

Place the software and dataset archives in sibling directories. From `software/`:

```bash
python3.10 -m venv --copies ../environments/v8-review
../environments/v8-review/bin/python -m pip --isolated install --disable-pip-version-check --no-cache-dir --only-binary=:all: --index-url https://pypi.org/simple pip==25.3
../environments/v8-review/bin/python -m pip --isolated install --disable-pip-version-check --no-cache-dir --only-binary=:all: --require-hashes --index-url https://pypi.org/simple -r environment/requirements-lock-macos-arm64-py310.txt
../environments/v8-review/bin/python -m pip check
```

The original successful installation used `requirements-clean-py310-candidate.txt`, pip 25.3 and binary wheels. Its installation report supplied the hash lock. On 23 November 2026, a second new environment was successfully installed directly from that unchanged lock; package consistency, dependency imports and read-only example-entry checks passed. See [installation evidence](LOCK_INSTALL_VALIDATION.md). Do not modify an existing research environment or remove hash checks to bypass a platform/version mismatch.

**Observed installation time:** approximately 26 seconds for environment creation, pip and locked dependency installation with pip's cache disabled. Python and macOS were already installed. A subsequent first-import check took about 42 seconds; this is not analysis runtime. These are one measured installation's times, not guarantees across machines or networks. Exact timings and exclusions are in [LOCK_INSTALL_VALIDATION.md](LOCK_INSTALL_VALIDATION.md).

## Verify and demonstrate

Follow [QUICKSTART.md](../docs/QUICKSTART.md) for a numerical example, expected outputs and existing measured runtime. The optional figure example writes PNG, PDF and editable-text SVG to a fresh output directory.

The prior installation passed package-consistency and recorded entry/effort tests. Its later use reproduced the analyses described in [ACCEPTANCE.md](../docs/ACCEPTANCE.md). The new locked installation was checked without rerunning those analyses. Upstream processing outside the documented entry points is not covered by these checks.

Compared with the earlier research environment, pyproj changed from 3.1.0 to 3.7.1 because a compatible binary wheel for the older version was unavailable. GeoPandas uses pyogrio 0.10.0; Fiona 1.9.6 imports successfully. Numerical equivalence was checked against references. Native font rendering differs and is not required to be pixel-identical.

Execution checks inputs before running, writes only to a fresh task output directory and refuses existing output directories. An informational blocked `lscpu` hardware query occurred in successful recorded runs; it did not change scientific output. Other errors or input mismatches must not be ignored.
