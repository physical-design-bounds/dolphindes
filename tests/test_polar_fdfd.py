"""
Tests for polar FDFD Maxwell solver and Green's function utilities.

Tests verify:
1. TM_Polar_FDFD class instantiation and basic functionality
2. Green's function computation matches direct solves
3. Multi-source superposition (linearity)
"""

import numpy as np
import pytest
import scipy.sparse.linalg as spla
from scipy.special import hankel1

from dolphindes.maxwell import (
    TM_Polar_FDFD,
)


class TestTMPolarFDFD:
    """Test TM_Polar_FDFD class functionality."""

    @pytest.fixture(params=[1, 6])
    def basic_solver(self, request):
        """Create a basic polar FDFD solver for testing."""
        omega = 2 * np.pi
        Nr = 40
        Nphi = 50
        Npml = 10
        dr = 0.05
        n_sectors = request.param
        return TM_Polar_FDFD(omega, Nphi, Nr, Npml, dr, n_sectors)

    def test_pixel_areas(self, basic_solver):
        """Test pixel area computation."""
        areas = basic_solver.get_pixel_areas()
        assert len(areas) == basic_solver.Nr * basic_solver.Nphi

        # Check total area matches geometry (pi * R^2 / n_sectors)
        R_max = basic_solver.Nr * basic_solver.dr
        expected_total_area = (np.pi * R_max**2) / basic_solver.n_sectors
        assert np.isclose(np.sum(areas), expected_total_area)

        # Check specific pixel area manually
        ir = 10
        r_center = (ir + 0.5) * basic_solver.dr
        dphi = 2 * np.pi / basic_solver.n_sectors / basic_solver.Nphi
        expected_pixel_area = r_center * basic_solver.dr * dphi

        # areas is flattened [r0..rNr-1, r0..rNr-1, ...], so index ir corresponds to that radius
        assert np.isclose(areas[ir], expected_pixel_area)

    def test_dipole_field(self, basic_solver):
        """Test dipole field computation."""
        ir = basic_solver.Nr // 2
        iphi = basic_solver.Nphi // 2
        Ez = basic_solver.get_TM_dipole_field(ir, iphi)
        assert Ez.shape == (basic_solver.Nphi * basic_solver.Nr,)
        # Field should be nonzero at source location
        idx = iphi * basic_solver.Nr + ir
        assert np.abs(Ez[idx]) > 0


class TestPolarGreensFunction:
    """Test Green's function computation and properties."""

    @pytest.fixture(params=[1, 6])
    def greens_setup(self, request):
        """Set up solver and masks for Green's function tests."""
        omega = 2 * np.pi
        Nr = 40
        Nphi = 50
        Npml = 10
        dr = 0.05
        n_sectors = request.param

        solver = TM_Polar_FDFD(omega, Nphi, Nr, Npml, dr, n_sectors)

        r_inner_des = 0.3
        r_outer_des = 1.0
        design_mask = np.zeros((Nr, Nphi), dtype=bool)
        for ir, r in enumerate(solver.r_grid):
            if r_inner_des <= r <= r_outer_des:
                design_mask[ir, :] = True

        observe_mask = design_mask.copy()

        return solver, design_mask, observe_mask

    def test_greens_function_vs_direct_solve(self, greens_setup):
        """Test that Green's function matches direct solve results."""
        solver, design_mask, observe_mask = greens_setup
        G = solver.get_TM_G_od(design_mask, observe_mask)

        area_vec = solver.get_pixel_areas()
        design_lin = np.nonzero(design_mask.flatten(order="F"))[0]
        observe_lin = np.nonzero(observe_mask.flatten(order="F"))[0]

        solve = spla.factorized(solver.M0.tocsc())

        # Test a few source locations
        N_des = len(design_lin)
        test_indices = [0, N_des // 4, N_des // 2]

        for des_idx in test_indices:
            pixel_global = design_lin[des_idx]

            # Direct solve
            J_full = np.zeros(solver.Nphi * solver.Nr, dtype=complex)
            J_full[pixel_global] = 1.0 / area_vec[pixel_global]
            E_direct = solve(1j * solver.omega * J_full)

            # Green's function approach
            J_design = np.zeros(N_des, dtype=complex)
            J_design[des_idx] = 1.0 / area_vec[pixel_global]
            E_green_obs = (1j / solver.omega) * (G @ J_design)

            # Compare at observation points
            E_direct_obs = E_direct[observe_lin]
            rel_error = np.linalg.norm(E_direct_obs - E_green_obs) / np.linalg.norm(
                E_direct_obs
            )

            assert rel_error < 1e-10, f"Relative error {rel_error} too large"

    @pytest.mark.parametrize(
        "amp1,amp2,description",
        [
            (1.0, 1.0, "equal_amplitudes"),
            (1.0, 1j, "90_degree_phase"),
            (0.5 + 0.5j, 0.3 - 0.2j, "complex_amplitudes"),
        ],
    )
    def test_greens_multi_source_superposition(
        self, greens_setup, amp1, amp2, description
    ):
        """Test Green's function with various multi-source configurations."""
        solver, design_mask, observe_mask = greens_setup
        G = solver.get_TM_G_od(design_mask, observe_mask)

        area_vec = solver.get_pixel_areas()
        design_lin = np.nonzero(design_mask.flatten(order="F"))[0]
        observe_lin = np.nonzero(observe_mask.flatten(order="F"))[0]

        solve = spla.factorized(solver.M0.tocsc())

        N_des = len(design_lin)
        # Choose two well-separated source locations
        idx1, idx2 = N_des // 5, 4 * N_des // 5
        pixel1, pixel2 = design_lin[idx1], design_lin[idx2]

        # Direct solve with both sources
        J_full = np.zeros(solver.Nphi * solver.Nr, dtype=complex)
        J_full[pixel1] = amp1 / area_vec[pixel1]
        J_full[pixel2] = amp2 / area_vec[pixel2]
        E_direct = solve(1j * solver.omega * J_full)

        # Green's function approach
        J_design = np.zeros(N_des, dtype=complex)
        J_design[idx1] = amp1 / area_vec[pixel1]
        J_design[idx2] = amp2 / area_vec[pixel2]
        E_green_obs = (1j / solver.omega) * (G @ J_design)

        E_direct_obs = E_direct[observe_lin]
        rel_error = np.linalg.norm(E_direct_obs - E_green_obs) / np.linalg.norm(
            E_direct_obs
        )

        assert rel_error < 1e-10, (
            f"Multi-source test '{description}' failed with error {rel_error}"
        )

    @pytest.mark.parametrize("n_sectors", [1, 6])
    def test_greens_different_observe_region(self, n_sectors):
        """Test Green's function when observe region differs from design region."""
        omega = 2 * np.pi
        Nr = 40
        Nphi = 50
        Npml = 10
        dr = 0.05

        solver = TM_Polar_FDFD(omega, Nphi, Nr, Npml, dr, n_sectors)

        # Design region: inner annulus
        design_mask = np.zeros((Nr, Nphi), dtype=bool)
        for ir, r in enumerate(solver.r_grid):
            if 0.3 <= r <= 0.6:
                design_mask[ir, :] = True

        # Observe region: outer annulus (non-overlapping)
        observe_mask = np.zeros((Nr, Nphi), dtype=bool)
        for ir, r in enumerate(solver.r_grid):
            if 0.8 <= r <= 1.2:
                observe_mask[ir, :] = True

        G = solver.get_TM_G_od(design_mask, observe_mask)

        area_vec = solver.get_pixel_areas()
        design_lin = np.nonzero(design_mask.flatten(order="F"))[0]
        observe_lin = np.nonzero(observe_mask.flatten(order="F"))[0]

        solve = spla.factorized(solver.M0.tocsc())

        N_des = len(design_lin)
        des_idx = N_des // 2
        pixel_global = design_lin[des_idx]

        # Direct solve
        J_full = np.zeros(solver.Nphi * solver.Nr, dtype=complex)
        J_full[pixel_global] = 1.0 / area_vec[pixel_global]
        E_direct = solve(1j * solver.omega * J_full)

        # Green's function approach
        J_design = np.zeros(N_des, dtype=complex)
        J_design[des_idx] = 1.0 / area_vec[pixel_global]
        E_green_obs = (1j / solver.omega) * (G @ J_design)

        E_direct_obs = E_direct[observe_lin]
        rel_error = np.linalg.norm(E_direct_obs - E_green_obs) / np.linalg.norm(
            E_direct_obs
        )

        assert rel_error < 1e-10, f"Non-overlapping regions error {rel_error}"

    def test_reciprocity(self, greens_setup):
        """
        Test Lorentz reciprocity for dipoles: E(r2 from p1) = E(r1 from p2).

        For unit dipoles p1=p2=1, the fields should be identical
        """
        solver, _, _ = greens_setup
        area_vec = solver.get_pixel_areas()

        p1 = solver.Nr // 3
        p2 = 2 * solver.Nr // 3
        A1 = area_vec[p1]
        A2 = area_vec[p2]

        # Field at 2 due to unit dipole moment source at 1
        J1 = np.zeros(solver.Nphi * solver.Nr, dtype=complex)
        J1[p1] = 1.0 / A1
        E21 = solver.get_TM_field(J1)[p2]

        # Field at 1 due to unit dipole moment source at 2
        J2 = np.zeros(solver.Nphi * solver.Nr, dtype=complex)
        J2[p2] = 1.0 / A2
        E12 = solver.get_TM_field(J2)[p1]

        # Check reciprocity relation
        assert np.isclose(E21, E12, rtol=1e-10)


class TestGaaInv:
    """Test inverse Green's function computation."""

    @pytest.fixture(params=[1, 6])
    def gaainv_setup(self, request):
        """Set up for GaaInv tests."""
        omega = 2 * np.pi
        Nr = 30
        Nphi = 40
        Npml = 8
        dr = 0.05
        n_sectors = request.param

        solver = TM_Polar_FDFD(omega, Nphi, Nr, Npml, dr, n_sectors)

        design_mask = np.zeros((Nr, Nphi), dtype=bool)
        design_mask[5:12, 10:18] = True

        return solver, design_mask

    def test_gaainv_identity(self, gaainv_setup):
        """Test that GaaInv @ Gaa is identity."""
        solver, design_mask = gaainv_setup
        GaaInv, _ = solver.get_GaaInv(design_mask)
        Gaa = solver.get_TM_G_od(design_mask, design_mask)

        product = GaaInv @ Gaa
        identity = np.eye(product.shape[0])
        rel_error = np.linalg.norm(product - identity) / np.linalg.norm(identity)
        assert rel_error < 1e-10

    @pytest.mark.parametrize("n_sectors", [1, 6])
    def test_gaainv_with_chigrid(self, n_sectors):
        """Test GaaInv computation with a background susceptibility."""
        omega = 2 * np.pi
        Nr = 30
        Nphi = 40
        Npml = 8
        dr = 0.05

        solver = TM_Polar_FDFD(omega, Nphi, Nr, Npml, dr, n_sectors)

        design_mask = np.zeros((Nr, Nphi), dtype=bool)
        design_mask[5:12, 10:18] = True

        # Create a simple background susceptibility
        chigrid = np.zeros((Nr, Nphi), dtype=complex)
        chigrid[3:8, 5:15] = 0.5 + 0.01j  # Some material outside design region

        GaaInv_vac, M_vac = solver.get_GaaInv(design_mask)
        GaaInv_mat, M_mat = solver.get_GaaInv(design_mask, chigrid)

        # Operators should be different when background is present
        assert np.linalg.norm((M_vac - M_mat).toarray()) > 0
        assert np.linalg.norm((GaaInv_vac - GaaInv_mat).toarray()) > 0


class TestPolarPML:
    """Test Perfectly Matched Layer (PML) performance."""

    @pytest.mark.parametrize("m", [3])
    def test_pml_behavior(self, m):
        """
        Verify PML works for graded (m=3) profiles.

        Checks:
        1. Field decay inside PML.
        2. Agreement with analytical solution in non-PML region (low reflection).
        """
        omega = 2 * np.pi
        Nr = 100
        Nphi = 10
        Npml = 40
        dr = 0.02
        n_sectors = 1

        solver = TM_Polar_FDFD(omega, Nphi, Nr, Npml, dr, n_sectors=n_sectors, m=m)

        # use a symmetric ring source at the first radial bin
        # to excite only the m=0 mode (cylindrical wave), matching H0(kr).
        J = np.zeros((Nr, Nphi), dtype=complex)
        area_2d = solver.get_pixel_areas().reshape((Nr, Nphi), order="F")
        J[0, :] = 1.0 / area_2d[0, :] / Nphi
        Ez = solver.get_TM_field(J.flatten(order="F"))

        Ez_2d = Ez.reshape((Nr, Nphi), order="F")
        Ez_radial = np.abs(Ez_2d[:, 0])  # Take slice at phi=0

        # 1. Check Decay in PML
        idx_interface = Nr - Npml - 1
        idx_back = Nr - 1

        val_interface = Ez_radial[idx_interface]
        val_back = Ez_radial[idx_back]

        decay_factor = val_back / val_interface
        assert decay_factor < 0.01, (
            f"PML (m={m}) did not decay enough: factor {decay_factor}"
        )

        # 2. Analytical: E = (omega * mu / 4) * H0(1)(k*r)
        skip = 5
        r_phys = solver.r_grid[skip:idx_interface]
        E_phys = Ez_radial[skip:idx_interface]
        E_anal = np.abs((omega / 4) * hankel1(0, omega * r_phys))

        rel_error = np.linalg.norm(E_phys - E_anal) / np.linalg.norm(E_anal)

        assert rel_error < 0.05, (
            f"PML (m={m}) reflection check failed: error {rel_error}"
        )
