# Changelog

Notable changes to dolphindes. Versions follow [semantic versioning](https://semver.org);
while we're pre-1.0, the public API may still change between minor releases.

## Unreleased

- **Tighter GCD bounds:** the smallest-eigenvector constraint GCD generates each
  iteration is now normalized as a whole rather than entry-by-entry. Dividing each
  entry by its own modulus flattened every magnitude to 1, keeping only the phase and
  discarding how strongly each entry moves the smallest eigenvalue; the phase is the
  part the theory fixes, while the magnitude profile is a free heuristic. Measured on
  LDOS bounds at ndof = 400/900/1600, under both orthonormalization metrics and 5
  starting points: 0.14–1.35% tighter bound at equal iteration count in all 16 paired
  comparisons, reaching the old 15-iteration bound in 12–15 iterations.
- **New:** `GCDHyperparameters(method=...)` selects the optimizer GCD uses for its
  inner dual solve, `"newton"` (the default, unchanged behaviour) or `"bfgs"`. This
  was previously hardcoded.
- **Fixed:** `solve_current_dual_problem` returned the dual *plus* the PSD-boundary
  penalties at the point where the optimizer stopped, and `run_gcd` reported and
  tested convergence on that value. With GCD's single penalty level it sat up to 0.5%
  above the smallest dual already evaluated, more than `gcd_tol`, so the reported
  bound could rise between iterations and GCD stopped whenever it did.

## [0.2.2] — 2026-07-29

Performance. Dual solves are 2.4–7.5x faster with identical results, and the gains
grow with the number of constraints: an LDOS bound on a 2500-pixel design region takes
1.9 s instead of 14.0 s with 128 constraints, 0.77 s instead of 3.7 s with 32, and GCD
runs land around 2–3x. The main change is that `A(λ)` is now assembled on a sparsity
pattern computed once per constraint set, so each dual evaluation writes into a fixed
data array instead of summing the constraint matrices one at a time and merging their
index structures: 5–58x on assembly alone, which had grown to be the largest cost in
the solve. Cholesky factorizations are also reused across the line search rather than
recomputed, the PSD-boundary eigensolve reuses the factorization it already has instead
of building its own, and several smaller per-call costs are gone. Nothing about the
bounds changed: at fixed multipliers the dual value, gradient and Hessian agree with
0.2.1 to 1e-11 or better.

Two other things worth knowing:

- **Bug fix:** with more than one PSD-boundary penalty vector live, the penalty gradient
  and Hessian were computed from only the last one, so Newton and GCD runs were stepping
  on wrong second derivatives (affecting convergence but not invalidating existing limits).
- **New:** GCD can orthogonalize its constraints in the Hilbert–Schmidt metric
  (`ortho_metric="hilbert_schmidt"`, now the default), which converges better per
  iteration than the previous Euclidean normalization.

## [0.2.1] — 2026-07-22

Packaging fix. The 0.2.0 wheel accidentally shipped only the top-level package,
so `import dolphindes.photonics` (and the other subpackages) failed on a fresh
install. 0.2.0 has been yanked from PyPI — use 0.2.1.

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

[0.2.2]: https://github.com/physical-design-bounds/dolphindes/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/physical-design-bounds/dolphindes/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/physical-design-bounds/dolphindes/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/physical-design-bounds/dolphindes/releases/tag/v0.1.0
