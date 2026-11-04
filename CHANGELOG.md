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
- **Fixed:** `solve_current_dual_problem` returned the dual *plus* the PSD-boundary
  penalties at the point where the optimizer stopped, and `run_gcd` reported and
  tested convergence on that value. With GCD's single penalty level it sat up to 0.5%
  above the smallest dual already evaluated, more than `gcd_tol`, so the reported
  bound could rise between iterations and GCD stopped whenever it did. It now returns
  the smallest penalty-free dual it evaluated with the multipliers, primal vector,
  gradient and (Newton) Hessian where it found it, without penalties; the unused
  `current_hess` attribute is removed.
- **Faster, more reliable Newton (GCD's default):** `Alt_Newton_GD` now takes each
  Newton step at its own length, trying the full step first and accepting the first
  sufficient (Armijo) decrease, instead of normalizing the direction and relearning
  its length by doubling. Along a nearly flat direction of the dual that relearning
  does not reach the minimizer before the inner stopping test fires, so GCD stalled.
  On 9 2D fixtures (LDOS, absorption, ring; 32–64 pixels a side), over 30 GCD
  iterations, Newton now uses 0.44x the factorizations and ends on average 0.5% above
  the best bound found (worst 1.4%), against 1.1% (worst 4.0%) before. Full steps
  need about twice as many Hessians, each about one solve per multiplier, so above
  `newton_alternate_above` multipliers (new in `OptimizationHyperparameters`, default
  100), as with pixel-local constraints, Newton still alternates normalized Newton
  and gradient steps.
- **Fewer failed factorizations at the PSD boundary (Newton):** when a trial step
  leaves the region where `A` is positive definite, the line search now jumps to 0.9
  of the distance to that boundary along the step, instead of shrinking the step by
  0.7 and refactorizing until it is back inside. `A` is affine in the multipliers, so
  the distance is `-1/mu` for the most negative eigenvalue `mu` of `A^-1 B`, `B` the
  change of `A` along the step: ARPACK (`eigs`) finds it through the factor already
  in hand, in about 10 solves.
- **Fixed:** BFGS could loop forever. The guard on `y·s` in its inverse-Hessian
  update was commented out, so a step that did not move `x` made the inverse Hessian
  NaN and no stopping test could pass. The update is now skipped unless
  `y·s > 1e-10 |y||s|`: relative, so small but genuine curvature still updates it,
  and NaN skips too.
- **Faster feasibility check:** the sparse Cholesky analysis is now supernodal, so a
  matrix that is not positive definite raises during its factorization and
  `is_dual_feasible` no longer extracts `L` to find out. CHOLMOD's default ("auto")
  factors small or very sparse matrices as a simplicial LDL^H, which succeeds on
  indefinite matrices; that is what the extraction was guarding against. It cost
  19% of a successful check at n = 14400, and 3% of a GCD run.
- **New:** `GCDHyperparameters(method=...)` selects the optimizer GCD uses for its
  inner dual solve, `"newton"` (the default, unchanged behaviour) or `"bfgs"`. This
  was previously hardcoded.
- **New:** `BFGS` and `Alt_Newton_GD` accept `nonneg=mask` to keep selected variables
  `>= 0`. It is an active-set method: a variable at 0 that the step would push below 0
  is held there, and the Newton or BFGS step is taken on the other variables.
- **New:** general (`B_j`) constraints can be inequalities. `gen_ineq=mask` marks the
  ones that read `Re(-x† A2† B_j A2 x + 2 x† A2† s_2j + c_2j) >= 0`, and the dual
  solve keeps their multipliers `>= 0`; write a `<=` constraint by negating `B_j`,
  `s_2j` and `c_2j`. Projector constraints remain equalities, and GCD and projector
  refinement, which only change projector constraints, work unchanged.
- **Fixed:** with `sparseQCQP=False`, `structure_objective` applied `A0` and `s0` to
  the polarization `p`, but `set_objective` stores them in the area-weighted basis
  `x = sqrt(W) p` the dense QCQP uses. The design objective and its gradient were
  therefore wrong whenever pixel areas differ from 1. It now evaluates them
  on `x`. The sparse path was correct and is unchanged.

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
