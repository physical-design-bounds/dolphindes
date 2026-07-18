"""JAX-differentiable wrappers for the TM FDFD field solves.

This module makes the scipy sparse solve behind ``get_TM_field`` differentiable to
JAX, so that a JAX pipeline can call a dolphindes solver as a black box and use
``jax.grad`` / ``jax.jvp`` / ``jax.hessian`` for simple inverse design. The forward
and adjoint solves stay on the CPU via scipy (wrapped in ``jax.pure_callback``);
only the cheap, exactly-differentiable algebra -- the ``chi`` dependence of the
operator and the ``1j * omega`` scaling of the source -- is expressed in native
JAX. That split is what lets ``jax.lax.custom_linear_solve`` produce correct
forward-, reverse-, and higher-order derivatives.

This is a one-directional bridge: ``dolphindes.maxwell`` has no JAX dependency,
and JAX enters only through this module (installed via the ``jax`` extra).

Build the differentiable function once, outside any optimization loop, then
differentiate it as usual::

    import jax
    jax.config.update("jax_enable_x64", True)

    field = build_jax_field_solver(solver)          # setup cost paid once
    grad_fn = jax.grad(lambda chi: loss(field(source_flat, chi)))

``build_jax_field_solver`` copies the operator into a native-JAX sparse array, so
calling it per iteration is wasteful. The returned function takes and
returns flat arrays; reshape at the call site if the solver is 2D.

64-bit precision is required: call ``jax.config.update("jax_enable_x64", True)``
for exact correspondence with the scipy solver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax.experimental import sparse as jsparse

from dolphindes.types import ComplexArray

if TYPE_CHECKING:
    from jax.typing import ArrayLike
    from jaxtyping import Array, Complex

__all__ = ["build_jax_field_solver"]


class _FieldSolver(Protocol):
    """Structural interface required from a TM FDFD solver.

    Both ``TM_FDFD`` and ``TM_Polar_FDFD`` satisfy this.
    """

    omega: complex
    M0: Any

    def _get_diagM_from_chigrid(self, chigrid: ComplexArray) -> sp.dia_array: ...
    def _assemble_M(self, chigrid: ComplexArray | None = ...) -> sp.csc_array: ...


def _require_x64() -> None:
    """Raise if JAX is not configured for 64-bit precision."""
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "dolphindes JAX field solves require 64-bit precision. "
            "Run `jax.config.update('jax_enable_x64', True)` at startup."
        )


def build_jax_field_solver(
    solver: _FieldSolver,
) -> Callable[[ArrayLike, ArrayLike], Complex[Array, " n"]]:
    """Build a JAX-differentiable TM field solve bound to ``solver``.

    Parameters
    ----------
    solver : TM_FDFD or TM_Polar_FDFD
        Solver instance. Its ``omega`` and ``M0`` are captured at build time, so
        the returned function will not see later mutations of either; rebuild if
        you change them.

    Returns
    -------
    field : callable
        ``field(source_flat, chi_flat) -> Ez_flat``, a flat complex field that is
        differentiable with respect to both ``source_flat`` and ``chi_flat``.
        Inputs are flattened and cast to complex128 internally. Pass a zero
        ``chi_flat`` for a vacuum solve.

    Notes
    -----
    Building is not free (it copies the operator into a native-JAX sparse array),
    so call this once and reuse the returned function rather than rebuilding it
    inside an optimization loop.
    """
    _require_x64()

    omega = complex(solver.omega)
    M0 = sp.csc_array(solver.M0)
    n = M0.shape[0]
    out_shape = jax.ShapeDtypeStruct((n,), np.complex128)

    # Native-JAX copy of M0 so that custom_linear_solve can transpose the matvec
    # (pure_callbacks are not transposable). Only matvec uses this; the actual
    # solves go through scipy below.
    M0_bcoo = jsparse.BCOO.from_scipy_sparse(M0.tocoo())

    # Per-pixel diagonal coefficient d such that diagM(chi) @ x == d * chi * x.
    # Derived from the solver's own operator so this tracks any change to
    # _get_diagM_from_chigrid.
    diag_coeff = jnp.asarray(
        np.asarray(
            solver._get_diagM_from_chigrid(np.ones(n, dtype=complex)).diagonal(),
            dtype=np.complex128,
        )
    )

    def matvec(
        chi: Complex[Array, " n"], x: Complex[Array, " n"]
    ) -> Complex[Array, " n"]:
        """Apply M(chi) to x, fully in native JAX."""
        return M0_bcoo @ x + diag_coeff * chi * x

    def _assemble_csc(chi_np: ArrayLike, transpose: bool) -> sp.csc_array:
        """Assemble the scipy operator M(chi) (or its transpose) for a solve."""
        m = solver._assemble_M(np.asarray(chi_np))
        return sp.csc_array(m.T) if transpose else m

    def make_solve(
        chi: Complex[Array, " n"], transpose: bool
    ) -> Callable[[Any, Complex[Array, " n"]], Complex[Array, " n"]]:
        """Return a custom_linear_solve-compatible solver closing over ``chi``.

        ``transpose`` selects the plain transpose M^T, which is the adjoint that
        JAX's reverse mode expects for this complex-linear operator (verified
        against jnp.linalg.solve to machine precision).
        """

        def _solve(
            _unused_matvec: Any, b: Complex[Array, " n"]
        ) -> Complex[Array, " n"]:
            def _callback(
                chi_in: Complex[Array, " n"], rhs: Complex[Array, " n"]
            ) -> ComplexArray:
                m = _assemble_csc(chi_in, transpose)
                x = spla.spsolve(m, np.asarray(rhs))
                return np.asarray(x, dtype=np.complex128)

            return jax.pure_callback(
                _callback, out_shape, chi, b, vmap_method="sequential"
            )

        return _solve

    def field(
        source: ArrayLike, chi: ArrayLike
    ) -> Complex[Array, " n"]:
        """Solve M(chi) @ Ez = 1j*omega*source, differentiable in source & chi."""
        source = jnp.asarray(source, dtype=jnp.complex128).ravel()
        chi = jnp.asarray(chi, dtype=jnp.complex128).ravel()
        rhs = 1j * omega * source
        return jax.lax.custom_linear_solve(
            lambda x: matvec(chi, x),
            rhs,
            make_solve(chi, transpose=False),
            transpose_solve=make_solve(chi, transpose=True),
        )

    return field
