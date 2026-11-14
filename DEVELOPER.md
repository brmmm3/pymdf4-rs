# Developer documentation

This guide is for contributors building and releasing `pymdf4-rs`. For
installation and API usage, see the [user documentation](README.md).

## Build and test

Build wheels with maturin 1.9.4 or newer. For a manual extension build, set
`PYO3_BUILD_EXTENSION_MODULE=1` when running `cargo build`. On Windows, copy
the resulting `target/debug/pymdf4.dll` to `pymdf4.pyd` and put it on Python's
import path. Run Rust tests with `cargo test` without that environment variable.

The integration tests require `numpy` and `asammdf` and can be run with
`python -m unittest discover -s tests -v` after building the extension. Some
tests and examples use recordings in `testdata/`, which is not tracked.

The project depends on `mdf4-rs` via the Git dependency in `Cargo.toml`.

## Build and release wheels

The wheel workflow builds and smoke-tests CPython 3.10 through 3.15 wheels
for Linux x86-64 (manylinux 2.28), Windows x86-64, and macOS x86-64 and
ARM64. Pushing a release tag of the form `x.y.z`, such as `0.2.0`, builds
the wheels and attaches them to a GitHub release after all builds succeed.
The workflow can also be run manually; it then publishes a release only if a
release version tag points at the selected commit, and otherwise just builds
artifacts. Use a tag matching the package version in `Cargo.toml`.

## Publish to package indexes

To publish those release wheels, manually run `publish-test-pypi.yml` or
`publish-pypi.yml` with the release tag. Each index must have a trusted
publisher configured for this repository, the corresponding workflow file,
and its GitHub environment (`testpypi` or `pypi`). See the
[PyPI trusted publisher documentation](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
These workflows publish wheels only and skip files already present on the index.
