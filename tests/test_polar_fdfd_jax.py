"""Tests for the JAX-differentiable TM_Polar_FDFD field solve (get_TM_field_jax).

Mirrors tests/test_fdfd_jax.py: forward checked against numpy ``get_TM_field``,
derivatives checked against a native dense ``jnp.linalg.solve`` reference.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from dolphindes.geometry import PolarFDFDGeometry  # noqa: E402
from dolphindes.maxwell import TM_Polar_FDFD  # noqa: E402


@pytest.fixture
def setup():
    """Small polar solver plus random source/chi/target."""
    rng = np.random.default_rng(0)
    omega = 2 * np.pi
    Nr, Nphi, Npml = 12, 16, 3
    geo = PolarFDFDGeometry(
        Nphi=Nphi,
        Nr=Nr,
        Npml=Npml,
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
    n = Nphi * Nr
    source = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    chi = 0.5 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    target = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return sim, source, chi, target, omega, n


def _dense_field_fn(sim, omega):
    """Native-JAX dense reference solve, differentiated by JAX itself."""
    M0_dense = jnp.asarray(sim.M0.toarray())

    def field_dense(source, chi):
        M = M0_dense - omega**2 * jnp.diag(jnp.asarray(chi).ravel())
        return jnp.linalg.solve(M, 1j * omega * jnp.asarray(source).ravel())

    return field_dense


def test_forward_matches_numpy(setup):
    """get_TM_field_jax reproduces numpy get_TM_field (material + vacuum)."""
    sim, source, chi, _, _, n = setup
    ez_np = sim.get_TM_field(source, chi)
    ez_jx = np.asarray(sim.get_TM_field_jax(jnp.asarray(source), jnp.asarray(chi)))
    assert ez_jx.shape == (n,)
    assert np.allclose(ez_np, ez_jx, atol=1e-10)

    ez_np_vac = sim.get_TM_field(source)
    ez_jx_vac = np.asarray(sim.get_TM_field_jax(jnp.asarray(source)))
    assert np.allclose(ez_np_vac, ez_jx_vac, atol=1e-10)


def test_dipole_matches_numpy(setup):
    """get_TM_dipole_field_jax reproduces numpy get_TM_dipole_field."""
    sim, _, chi, _, _, _ = setup
    ir, iphi = sim.geometry.Nr // 2, sim.geometry.Nphi // 2
    ez_np = sim.get_TM_dipole_field(ir, iphi, chi)
    ez_jx = np.asarray(sim.get_TM_dipole_field_jax(ir, iphi, jnp.asarray(chi)))
    assert np.allclose(ez_np, ez_jx, atol=1e-10)


def test_grad_chi_and_source(setup):
    """Reverse-mode grad wrt chi and source matches native dense JAX."""
    sim, source, chi, target, omega, _ = setup
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
    """Realistic inverse design: real DOF -> complex chi -> real loss."""
    sim, source, _, target, omega, n = setup
    field_dense = _dense_field_fn(sim, omega)
    tgt, s_j = jnp.asarray(target), jnp.asarray(source)
    chi_mat = 4.0 + 0.2j
    rng = np.random.default_rng(7)
    rho0 = jnp.asarray(rng.standard_normal(n))

    def make_loss(field):
        def loss(rho):
            chi = chi_mat * rho.astype(jnp.complex128)
            return jnp.real(jnp.vdot(tgt, field(s_j, chi).ravel()))

        return loss

    loss_jx, loss_dn = make_loss(sim.get_TM_field_jax), make_loss(field_dense)

    g = np.asarray(jax.grad(loss_jx)(rho0))
    assert np.max(np.abs(np.imag(g))) == 0.0
    assert np.allclose(g, np.asarray(jax.grad(loss_dn)(rho0)), atol=1e-8)

    v = jnp.asarray(rng.standard_normal(n))
    _, t_jx = jax.jvp(loss_jx, (rho0,), (v,))
    _, t_dn = jax.jvp(loss_dn, (rho0,), (v,))
    assert np.allclose(np.asarray(t_jx), np.asarray(t_dn), atol=1e-8)

    h_jx = np.asarray(jax.hessian(loss_jx)(rho0))
    h_dn = np.asarray(jax.hessian(loss_dn)(rho0))
    assert h_jx.shape == (n, n)
    assert np.allclose(h_jx, h_dn, atol=1e-7)


def test_grad_matches_finite_difference():
    """jax.grad matches a central finite-difference of the numpy forward solve.

    Independent of JAX autodiff: the FD gradient comes purely from the existing
    numpy ``get_TM_field``.
    """
    rng = np.random.default_rng(3)
    geo = PolarFDFDGeometry(
        Nphi=10,
        Nr=8,
        Npml=2,
        dr=0.05,
        n_sectors=1,
        r_inner=0.0,
        Npml_inner=0,
        mirror=False,
        bloch_phase=0.0,
        m=3,
        lnR=-16.0,
    )
    sim = TM_Polar_FDFD(2 * np.pi, geo)
    n = 10 * 8
    source = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    target = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    chi_mat = 3.0 + 0.15j
    rho0 = rng.standard_normal(n)

    def loss_numpy(rho):
        chi = chi_mat * rho
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
        chi = chi_mat * rho.astype(jnp.complex128)
        ez = sim.get_TM_field_jax(jnp.asarray(source), chi)
        return jnp.real(jnp.vdot(jnp.asarray(target), ez.ravel()))

    g = np.asarray(jax.grad(loss_jax)(jnp.asarray(rho0)))
    assert np.allclose(g, fd, rtol=1e-4, atol=1e-6)
