"""Tests for the JAX-differentiable TM_FDFD field solve (get_TM_field_jax).

Forward results are checked against the existing numpy ``get_TM_field``; all
derivatives (grad / jvp / hessian) are checked against a native dense
``jnp.linalg.solve`` reference, which JAX differentiates correctly by its own
convention -- so matching it confirms the custom_linear_solve rule composes with
any JAX pipeline.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from dolphindes.geometry import CartesianFDFDGeometry  # noqa: E402
from dolphindes.maxwell import TM_FDFD  # noqa: E402


@pytest.fixture
def setup():
    """Small Cartesian solver plus random source/chi/target."""
    rng = np.random.default_rng(0)
    gpr = 12
    dl = 1.0 / gpr
    Nx, Ny = 16, 14
    Npmlx = Npmly = 3
    omega = 2 * np.pi
    geo = CartesianFDFDGeometry(Nx, Ny, Npmlx, Npmly, dl, dl)
    sim = TM_FDFD(omega, geo)
    n = Nx * Ny
    source = rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))
    chi = 0.5 * (rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny)))
    target = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return sim, source, chi, target, omega


def _dense_field_fn(sim, omega):
    """Native-JAX dense reference solve, differentiated by JAX itself."""
    M0_dense = jnp.asarray(sim.M0.toarray())

    def field_dense(source, chi):
        M = M0_dense - omega**2 * jnp.diag(jnp.asarray(chi).ravel())
        return jnp.linalg.solve(M, 1j * omega * jnp.asarray(source).ravel())

    return field_dense


def test_forward_matches_numpy(setup):
    """get_TM_field_jax reproduces numpy get_TM_field (material + vacuum)."""
    sim, source, chi, _, _ = setup
    ez_np = sim.get_TM_field(source, chi)
    ez_jx = np.asarray(sim.get_TM_field_jax(jnp.asarray(source), jnp.asarray(chi)))
    assert ez_jx.shape == (sim.Nx, sim.Ny)
    assert np.allclose(ez_np, ez_jx, atol=1e-10)

    ez_np_vac = sim.get_TM_field(source)
    ez_jx_vac = np.asarray(sim.get_TM_field_jax(jnp.asarray(source)))
    assert np.allclose(ez_np_vac, ez_jx_vac, atol=1e-10)


def test_dipole_matches_numpy(setup):
    """get_TM_dipole_field_jax reproduces numpy get_TM_dipole_field."""
    sim, _, chi, _, _ = setup
    cx, cy = sim.Nx // 2, sim.Ny // 2
    ez_np = sim.get_TM_dipole_field(cx, cy, chi)
    ez_jx = np.asarray(sim.get_TM_dipole_field_jax(cx, cy, jnp.asarray(chi)))
    assert np.allclose(ez_np, ez_jx, atol=1e-10)


def test_grad_chi_and_source(setup):
    """Reverse-mode grad wrt chi and source matches native dense JAX."""
    sim, source, chi, target, omega = setup
    field_dense = _dense_field_fn(sim, omega)
    tgt, s_j, c_j = jnp.asarray(target), jnp.asarray(source), jnp.asarray(chi)

    def loss(field, s, c):
        return jnp.real(jnp.vdot(tgt, field(s, c).ravel()))

    g_chi = np.asarray(jax.grad(lambda c: loss(sim.get_TM_field_jax, s_j, c))(c_j))
    g_chi_ref = np.asarray(jax.grad(lambda c: loss(field_dense, s_j, c))(c_j))
    assert np.allclose(g_chi, g_chi_ref, atol=1e-8)

    g_src = np.asarray(jax.grad(lambda s: loss(sim.get_TM_field_jax, s, c_j))(s_j))
    g_src_ref = np.asarray(jax.grad(lambda s: loss(field_dense, s, c_j))(s_j))
    assert np.allclose(g_src, g_src_ref, atol=1e-8)


def test_real_param_grad_jvp_hessian(setup):
    """Realistic inverse design: real DOF -> complex chi -> real loss.

    Checks reverse grad, forward jvp, and the Hessian against native dense JAX.
    """
    sim, source, _, target, omega = setup
    field_dense = _dense_field_fn(sim, omega)
    tgt, s_j = jnp.asarray(target), jnp.asarray(source)
    n = sim.Nx * sim.Ny
    chi_mat = 4.0 + 0.2j
    rng = np.random.default_rng(7)
    rho0 = jnp.asarray(rng.standard_normal(n))

    def make_loss(field):
        def loss(rho):
            chi = (chi_mat * rho.astype(jnp.complex128)).reshape(sim.Nx, sim.Ny)
            return jnp.real(jnp.vdot(tgt, field(s_j, chi).ravel()))

        return loss

    loss_jx, loss_dn = make_loss(sim.get_TM_field_jax), make_loss(field_dense)

    # gradient must be real for real DOFs
    g = np.asarray(jax.grad(loss_jx)(rho0))
    assert np.max(np.abs(np.imag(g))) == 0.0
    assert np.allclose(g, np.asarray(jax.grad(loss_dn)(rho0)), atol=1e-8)

    # forward mode
    v = jnp.asarray(rng.standard_normal(n))
    _, t_jx = jax.jvp(loss_jx, (rho0,), (v,))
    _, t_dn = jax.jvp(loss_dn, (rho0,), (v,))
    assert np.allclose(np.asarray(t_jx), np.asarray(t_dn), atol=1e-8)

    # hessian
    h_jx = np.asarray(jax.hessian(loss_jx)(rho0))
    h_dn = np.asarray(jax.hessian(loss_dn)(rho0))
    assert h_jx.shape == (n, n)
    assert np.allclose(h_jx, h_dn, atol=1e-7)


def test_grad_matches_finite_difference():
    """jax.grad matches a central finite-difference of the numpy forward solve.

    This is an independent ground truth: the FD gradient is built purely from the
    existing numpy ``get_TM_field`` (no JAX autodiff), so agreement confirms the
    custom rule's sign/conjugation convention on its own terms.
    """
    rng = np.random.default_rng(3)
    dl = 1.0 / 12
    nx, ny = 10, 8
    sim = TM_FDFD(2 * np.pi, CartesianFDFDGeometry(nx, ny, 2, 2, dl, dl))
    n = nx * ny
    source = rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))
    target = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    chi_mat = 3.0 + 0.15j
    rho0 = rng.standard_normal(n)

    def loss_numpy(rho):
        chi = (chi_mat * rho).reshape(nx, ny)
        ez = sim.get_TM_field(source, chi)
        return float(np.real(np.vdot(target, ez.ravel())))

    eps = 1e-6
    fd = np.array(
        [
            (
                loss_numpy(rho0 + eps * np.eye(n)[i])
                - loss_numpy(rho0 - eps * np.eye(n)[i])
            )
            / (2 * eps)
            for i in range(n)
        ]
    )

    def loss_jax(rho):
        chi = (chi_mat * rho.astype(jnp.complex128)).reshape(nx, ny)
        ez = sim.get_TM_field_jax(jnp.asarray(source), chi)
        return jnp.real(jnp.vdot(jnp.asarray(target), ez.ravel()))

    g = np.asarray(jax.grad(loss_jax)(jnp.asarray(rho0)))
    assert np.allclose(g, fd, rtol=1e-4, atol=1e-6)
