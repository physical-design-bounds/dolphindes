# Changelog

Notable changes to dolphindes. Versions follow [semantic versioning](https://semver.org);
while we're pre-1.0, the public API may still change between minor releases.

## [0.2.0] — 2026-07-22

A year of work on top of the first public release. The highlights:

- **Generalized constraints.** The QCQP core was rebuilt around a shared-projection
  formulation with a proper off-diagonal projector framework, so you're no longer
  limited to the few constraint types the original code handled.
- **Differentiable solver.** New optional JAX bridge (`pip install dolphindes[jax]`)
  lets you differentiate through the FDFD solve.
- **Polar FDFD solver.** Solve on polar grids, with rotational/mirror symmetry,
  non-zero inner boundaries, and inner PML.
- **Nicer setup.** Geometry and optimizer settings are now dataclasses
  (`CartesianFDFDGeometry`, `PolarFDFDGeometry`, `OptimizationHyperparameters`, …)
  instead of loose arguments and dicts.
- **Input validation** with clearer errors, and adjoint gradients for `Photonics_TM_FDFD`.
- **Docs, typing, CI.** Sphinx docs on Read the Docs, a typed package (`py.typed`),
  and ruff/mypy/pytest running on every PR.

## [0.1.0] — 2025-07-22

First public release: performance bounds for 2D photonics problems with Ez (TM)
polarization.

[0.2.0]: https://github.com/physical-design-bounds/dolphindes/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/physical-design-bounds/dolphindes/releases/tag/v0.1.0
