"""Tests for the JAX-differentiable FDFD field solve (build_jax_field_solver).

``build_jax_field_solver`` is a single implementation that works for any solver
satisfying its structural interface, so every test here runs against both
``TM_FDFD`` and ``TM_Polar_FDFD`` via the parametrized ``case`` fixture; the
solver-specific parts (geometry, dipole normalization) live in the builders.

Forward results are checked against the solver's own numpy ``get_TM_field``; all
derivatives (grad / jvp / hessian) are checked against a native dense
``jnp.linalg.solve`` reference, which JAX differentiates correctly by its own
convention -- so matching it confirms the custom_linear_solve rule composes with
any JAX pipeline.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as spla

from dolphindes.geometry import CartesianFDFDGeometry, PolarFDFDGeometry
from dolphindes.maxwell import TM_FDFD, TM_Polar_FDFD
from dolphindes.maxwell.jax_fdfd import build_jax_field_solver


@dataclass
class Case:
    """A solver plus its JAX field fn and matching flat test data."""

    sim: Any
    field: Any
    n: int
    omega: float
    source: np.ndarray  # flat
    chi: np.ndarray  # flat
    target: np.ndarray  # flat
    dipole_source: np.ndarray  # flat, solver's own unit-dipole normalization
    dipole_reference: np.ndarray  # flat, from the numpy get_TM_dipole_field


def _cartesian_case(nx: int, ny: int, npml: int, seed: int) -> Case:
    """Build a Cartesian TM_FDFD case."""
    rng = np.random.default_rng(seed)
    omega = 2 * np.pi
    dl = 1.0 / 12
    sim = TM_FDFD(omega, CartesianFDFDGeometry(nx, ny, npml, npml, dl, dl))
    n = nx * ny
    chi = 0.5 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    cx, cy = nx // 2, ny // 2
    dipole_source = np.zeros(n, dtype=complex)
    dipole_source[cx * ny + cy] = 1.0 / (sim.dx * sim.dy)

    return Case(
        sim=sim,
        field=build_jax_field_solver(sim),
        n=n,
        omega=omega,
        source=rng.standard_normal(n) + 1j * rng.standard_normal(n),
        chi=chi,
        target=rng.standard_normal(n) + 1j * rng.standard_normal(n),
        dipole_source=dipole_source,
        dipole_reference=np.asarray(
            sim.get_TM_dipole_field(cx, cy, chi.reshape(nx, ny))
        ).ravel(),
    )


def _polar_case(nphi: int, nr: int, npml: int, seed: int) -> Case:
    """Build a polar TM_Polar_FDFD case."""
    rng = np.random.default_rng(seed)
    omega = 2 * np.pi
    geo = PolarFDFDGeometry(
        Nphi=nphi,
        Nr=nr,
        Npml=npml,
        dr=0.05,
        n_sectors=1,
        r_inner=0.0,
        Npml_inner=0,
        mirror=False,
        bloch_phase=0.0,
        m=3,
        lnR=-16.0,
    )
    sim = TM_Polar_FDFD(omega, geo)
    n = nphi * nr
    chi = 0.5 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    ir, iphi = nr // 2, nphi // 2
    idx = iphi * nr + ir
    dipole_source = np.zeros(n, dtype=complex)
    dipole_source[idx] = 1.0 / geo.get_pixel_areas()[idx]

    return Case(
        sim=sim,
        field=build_jax_field_solver(sim),
        n=n,
        omega=omega,
        source=rng.standard_normal(n) + 1j * rng.standard_normal(n),
        chi=chi,
        target=rng.standard_normal(n) + 1j * rng.standard_normal(n),
        dipole_source=dipole_source,
        dipole_reference=np.asarray(sim.get_TM_dipole_field(ir, iphi, chi)).ravel(),
    )


@pytest.fixture(params=["cartesian", "polar"])
def case(request):
    """A solver case big enough to exercise autodiff, small enough to be quick."""
    if request.param == "cartesian":
        return _cartesian_case(nx=16, ny=14, npml=3, seed=0)
    return _polar_case(nphi=16, nr=12, npml=3, seed=0)


@pytest.fixture(params=["cartesian", "polar"])
def small_case(request):
    """A smaller case, so the O(n) finite-difference sweep stays cheap."""
    if request.param == "cartesian":
        return _cartesian_case(nx=10, ny=8, npml=2, seed=3)
    return _polar_case(nphi=10, nr=8, npml=2, seed=3)


def _dense_field_fn(sim, omega):
    """Native-JAX dense reference solve, differentiated by JAX itself."""
    M0_dense = jnp.asarray(sim.M0.toarray())

    def field_dense(source, chi):
        M = M0_dense - omega**2 * jnp.diag(jnp.asarray(chi).ravel())
        return jnp.linalg.solve(M, 1j * omega * jnp.asarray(source).ravel())

    return field_dense


def test_forward_matches_numpy(case):
    """The JAX field fn reproduces the numpy get_TM_field (material + vacuum)."""
    ez_np = np.asarray(case.sim.get_TM_field(case.source, case.chi)).ravel()
    ez_jx = np.asarray(case.field(case.source, case.chi))
    assert ez_jx.shape == (case.n,)
    assert np.allclose(ez_np, ez_jx, atol=1e-10)

    # vacuum is a zero chi, since the flat interface has no None default
    ez_np_vac = np.asarray(case.sim.get_TM_field(case.source)).ravel()
    ez_jx_vac = np.asarray(case.field(case.source, np.zeros(case.n, dtype=complex)))
    assert np.allclose(ez_np_vac, ez_jx_vac, atol=1e-10)


def test_dipole_source_matches_numpy(case):
    """A hand-built dipole source reproduces the numpy get_TM_dipole_field.

    Confirms each solver's unit-dipole normalization is what the flat interface
    expects, since build_jax_field_solver has no dipole convenience of its own.
    """
    ez_jx = np.asarray(case.field(case.dipole_source, case.chi))
    assert np.allclose(case.dipole_reference, ez_jx, atol=1e-10)


def test_grad_chi_and_source(case):
    """Reverse-mode grad wrt chi and source matches native dense JAX."""
    field_dense = _dense_field_fn(case.sim, case.omega)
    tgt = jnp.asarray(case.target)
    s_j, c_j = jnp.asarray(case.source), jnp.asarray(case.chi)

    def loss(f, s, c):
        return jnp.real(jnp.vdot(tgt, f(s, c).ravel()))

    g_chi = np.asarray(jax.grad(lambda c: loss(case.field, s_j, c))(c_j))
    g_chi_ref = np.asarray(jax.grad(lambda c: loss(field_dense, s_j, c))(c_j))
    assert np.allclose(g_chi, g_chi_ref, atol=1e-8)

    g_src = np.asarray(jax.grad(lambda s: loss(case.field, s, c_j))(s_j))
    g_src_ref = np.asarray(jax.grad(lambda s: loss(field_dense, s, c_j))(s_j))
    assert np.allclose(g_src, g_src_ref, atol=1e-8)


def test_real_param_grad_jvp_hessian(case):
    """Realistic inverse design: real DOF -> complex chi -> real loss.

    Checks reverse grad, forward jvp, and the Hessian against native dense JAX.
    """
    field_dense = _dense_field_fn(case.sim, case.omega)
    tgt, s_j = jnp.asarray(case.target), jnp.asarray(case.source)
    chi_mat = 4.0 + 0.2j
    rng = np.random.default_rng(7)
    rho0 = jnp.asarray(rng.standard_normal(case.n))

    def make_loss(f):
        def loss(rho):
            chi = chi_mat * rho.astype(jnp.complex128)
            return jnp.real(jnp.vdot(tgt, f(s_j, chi).ravel()))

        return loss

    loss_jx, loss_dn = make_loss(case.field), make_loss(field_dense)

    # gradient must be real for real DOFs
    g = np.asarray(jax.grad(loss_jx)(rho0))
    assert np.max(np.abs(np.imag(g))) == 0.0
    assert np.allclose(g, np.asarray(jax.grad(loss_dn)(rho0)), atol=1e-8)

    # forward mode
    v = jnp.asarray(rng.standard_normal(case.n))
    _, t_jx = jax.jvp(loss_jx, (rho0,), (v,))
    _, t_dn = jax.jvp(loss_dn, (rho0,), (v,))
    assert np.allclose(np.asarray(t_jx), np.asarray(t_dn), atol=1e-8)

    # hessian
    h_jx = np.asarray(jax.hessian(loss_jx)(rho0))
    h_dn = np.asarray(jax.hessian(loss_dn)(rho0))
    assert h_jx.shape == (case.n, case.n)
    assert np.allclose(h_jx, h_dn, atol=1e-7)


def test_grad_matches_finite_difference(small_case):
    """jax.grad matches a central finite-difference of the numpy forward solve.

    This is an independent ground truth: the FD gradient is built purely from the
    existing numpy ``get_TM_field`` (no JAX autodiff), so agreement confirms the
    custom rule's sign/conjugation convention on its own terms.
    """
    c = small_case
    rng = np.random.default_rng(3)
    chi_mat = 3.0 + 0.15j
    rho0 = rng.standard_normal(c.n)

    def loss_numpy(rho):
        ez = c.sim.get_TM_field(c.source, chi_mat * rho)
        return float(np.real(np.vdot(c.target, np.asarray(ez).ravel())))

    eps = 1e-6
    basis = np.eye(c.n)
    fd = np.array(
        [
            (loss_numpy(rho0 + eps * basis[i]) - loss_numpy(rho0 - eps * basis[i]))
            / (2 * eps)
            for i in range(c.n)
        ]
    )

    def loss_jax(rho):
        ez = c.field(jnp.asarray(c.source), chi_mat * rho.astype(jnp.complex128))
        return jnp.real(jnp.vdot(jnp.asarray(c.target), ez.ravel()))

    g = np.asarray(jax.grad(loss_jax)(jnp.asarray(rho0)))
    assert np.allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_accepts_unflattened_input():
    """A 2D (Nx, Ny) source/chi gives the same result as pre-flattened input.

    Only meaningful for the Cartesian solver, whose natural grid is 2D; the flat
    interface is documented to ravel its inputs.
    """
    c = _cartesian_case(nx=10, ny=8, npml=2, seed=5)
    flat = np.asarray(c.field(c.source, c.chi))
    grid = np.asarray(c.field(c.source.reshape(10, 8), c.chi.reshape(10, 8)))
    assert np.allclose(flat, grid, atol=1e-12)


def test_jit_matches_eager(case):
    """jit(field) and jit(grad(loss)) reproduce their eager counterparts.

    The solve runs inside a jax.pure_callback, so this guards that the bridge
    stays correct under jit (the intended usage in an optimization loop).
    """
    s_j, c_j = jnp.asarray(case.source), jnp.asarray(case.chi)

    f_eager = np.asarray(case.field(s_j, c_j))
    f_jit = np.asarray(jax.jit(case.field)(s_j, c_j))
    assert np.allclose(f_eager, f_jit, atol=1e-10)

    tgt = jnp.asarray(case.target)

    def loss(c):
        return jnp.real(jnp.vdot(tgt, case.field(s_j, c).ravel()))

    g_eager = np.asarray(jax.grad(loss)(c_j))
    g_jit = np.asarray(jax.jit(jax.grad(loss))(c_j))
    assert np.allclose(g_eager, g_jit, atol=1e-10)


def test_repeated_calls_distinct_chi(case):
    """Successive solves at distinct chi each match numpy.

    Guards the LU cache keying: a stale factorization would make a later chi
    silently reuse an earlier operator.
    """
    rng = np.random.default_rng(11)
    for _ in range(3):
        chi = rng.standard_normal(case.n) + 1j * rng.standard_normal(case.n)
        ez_np = np.asarray(case.sim.get_TM_field(case.source, chi)).ravel()
        ez_jx = np.asarray(case.field(case.source, chi))
        assert np.allclose(ez_np, ez_jx, atol=1e-10)


def test_factorization_reused_between_forward_and_adjoint(small_case, monkeypatch):
    """One value_and_grad factorizes M once; the adjoint reuses it via trans='T'.

    Directly checks the performance intent of the LU cache rather than just the
    numerical result.
    """
    c = small_case  # function-scoped fixture -> the field's LU cache is empty
    tgt, s_j = jnp.asarray(c.target), jnp.asarray(c.source)

    calls = {"n": 0}
    real_splu = spla.splu

    def counting_splu(*a, **k):
        calls["n"] += 1
        return real_splu(*a, **k)

    monkeypatch.setattr(spla, "splu", counting_splu)

    def loss(chi):
        return jnp.real(jnp.vdot(tgt, c.field(s_j, chi).ravel()))

    jax.value_and_grad(loss)(jnp.asarray(c.chi))
    assert calls["n"] == 1


def test_lu_cache_evicts_least_recently_used(small_case, monkeypatch):
    """Filling past the cache cap evicts the oldest chi, keeping the rest.

    Overflowing the cap (8) by one must drop exactly the least-recently-used
    entry: a still-cached chi is reused (no refactorization) while the evicted
    one is factorized again.
    """
    c = small_case  # function-scoped fixture -> the field's LU cache is empty
    cap = 8  # build_jax_field_solver's lu_cache_max
    rng = np.random.default_rng(23)
    chis = [
        rng.standard_normal(c.n) + 1j * rng.standard_normal(c.n) for _ in range(cap + 1)
    ]

    # One forward solve per distinct chi factorizes once; inserting the (cap+1)th
    # evicts chis[0], leaving chis[1..cap] cached (chis[cap] most recent).
    for chi in chis:
        np.asarray(c.field(jnp.asarray(c.source), jnp.asarray(chi)))

    real_splu = spla.splu
    calls = {"n": 0}

    def counting_splu(*a, **k):
        calls["n"] += 1
        return real_splu(*a, **k)

    monkeypatch.setattr(spla, "splu", counting_splu)

    # Still cached -> reused, no new factorization.
    np.asarray(c.field(jnp.asarray(c.source), jnp.asarray(chis[cap])))
    assert calls["n"] == 0

    # Evicted -> must factorize again.
    np.asarray(c.field(jnp.asarray(c.source), jnp.asarray(chis[0])))
    assert calls["n"] == 1
