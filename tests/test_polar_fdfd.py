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

from dolphindes.geometry import PolarFDFDGeometry
from dolphindes.maxwell import (
    TM_Polar_FDFD, expand_symmetric_field
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
        geometry = PolarFDFDGeometry(
            Nphi=Nphi,
            Nr=Nr,
            Npml=Npml,
            dr=dr,
            n_sectors=n_sectors,
            r_inner=0.0,
            Npml_inner=0,
            mirror=False,
            bloch_phase=0.0,
            m=3,
            lnR=-16.0,
        )
        return TM_Polar_FDFD(omega, geometry)

    def test_pixel_areas(self, basic_solver):
        """Test pixel area computation."""
        areas = basic_solver.geometry.get_pixel_areas()
        assert len(areas) == basic_solver.geometry.Nr * basic_solver.geometry.Nphi

        # Check total area matches geometry (pi * R^2 / n_sectors)
        R_max = basic_solver.geometry.Nr * basic_solver.geometry.dr
        expected_total_area = (np.pi * R_max**2) / basic_solver.geometry.n_sectors
        assert np.isclose(np.sum(areas), expected_total_area)

        # Check specific pixel area manually
        ir = 10
        r_center = (ir + 0.5) * basic_solver.geometry.dr
        dphi = 2 * np.pi / basic_solver.geometry.n_sectors / basic_solver.geometry.Nphi
        expected_pixel_area = r_center * basic_solver.geometry.dr * dphi

        # areas is flattened [r0..rNr-1, r0..rNr-1, ...], so index ir corresponds to that radius
        assert np.isclose(areas[ir], expected_pixel_area)

    def test_dipole_field(self, basic_solver):
        """Test dipole field computation."""
        ir = basic_solver.geometry.Nr // 2
        iphi = basic_solver.geometry.Nphi // 2
        Ez = basic_solver.get_TM_dipole_field(ir, iphi)
        assert Ez.shape == (basic_solver.geometry.Nphi * basic_solver.geometry.Nr,)
        # Field should be nonzero at source location
        idx = iphi * basic_solver.geometry.Nr + ir
        assert np.abs(Ez[idx]) > 0


class TestMirrorSymmetry:
    """Test mirror symmetry with Neumann boundary conditions"""
    @pytest.fixture()
    def solvers(self):
        wvlgth = 1.0
        omega = 2*np.pi / wvlgth
        r_nonpml = 2.0 # center circle radius
        w_pml = 0.5 # surrounding pml thickness
        r_tot = r_nonpml + w_pml # total computational domain radius
        
        gpr = 30
        #gpr = 60 # high res option
        dr = 1.0/gpr # radial grid size
        Nr = int(np.round(r_tot / dr))
        Npml = int(np.round(w_pml / dr))
        
        # setting azimuthal grid size. Note the azimuthal pixel width gets larger with radius
        Nphi_full = 180 # this gives ~ 0.043 pixel width at the edge of computational domain for R_tot=2.5
        #Nphi_full = 360 # high res option
        
        # Example: 6-fold rotational symmetry and mirror symmetry (60° sectors, 30° half sectors)
        n_sectors = 6

        Nphi_sector = Nphi_full // n_sectors
        Nphi_halfsector = Nphi_sector // 2  # azimuthal points in one 30° irreducible domain

        geo_halfsector = PolarFDFDGeometry(Nphi_halfsector, Nr, Npml, dr, 
                                       n_sectors=n_sectors, mirror=True)
    
        FDFD_halfsector = TM_Polar_FDFD(omega, geo_halfsector)
        
        geo_full = PolarFDFDGeometry(Nphi_full, Nr, Npml, dr)
        FDFD_full = TM_Polar_FDFD(omega, geo_full)
        
        return (FDFD_halfsector, FDFD_full)
    
    def test_center_dipole(self, solvers):
        """test Neumann boundaries for dipole source at the origin"""
        FDFD_halfsector = solvers[0]
        FDFD_full = solvers[1]
        omega = FDFD_halfsector.omega
        r_grid = FDFD_halfsector.geometry.r_grid
        Nr = FDFD_halfsector.geometry.Nr
        Nphi_halfsector = FDFD_halfsector.geometry.Nphi
        Nphi_full = FDFD_full.geometry.Nphi
        n_sectors = FDFD_halfsector.geometry.n_sectors
        Npml = FDFD_halfsector.geometry.Npml
        dr = FDFD_halfsector.geometry.dr
        
        J_r = np.zeros(Nr)
        J_r[0] = 1.0 / (np.pi* dr**2)

        J_center_dipole = np.kron(np.ones(Nphi_halfsector), J_r)
        
        ## first test vacuum field against analytical result
        E_center_dipole = FDFD_halfsector.get_TM_field(J_center_dipole)
        
        analytic_E_grid = (-omega/4) * hankel1(0, omega*r_grid)
        
        rel_err = np.linalg.norm(analytic_E_grid[5:Nr-2*Npml] 
                                 - E_center_dipole[5:Nr-2*Npml])
        
        rel_err /= np.linalg.norm(analytic_E_grid[5:Nr-2*Npml])
        assert rel_err < 1e-2, f"Relative error {rel_err} too large."
    
        # introduce a structure
        r_ring = 1.0  # center radius of ring
        r_t = 0.3     # thickness of ring
        r_inner = r_ring
        r_outer = r_ring + r_t
        
        # Create radial mask for the ring
        chi_r_grid = np.zeros(Nr)
        chi_r_grid[int(r_inner/dr):int(r_outer/dr)] = 1.0
        
        ## setup structure grid
        chi_phi_grid = np.zeros(Nphi_halfsector, dtype=complex)
        phi_start = 0  # start at phi = 0
        phi_end = int(np.round(Nphi_halfsector / 2))
        chi_phi_grid[phi_start:phi_end] = 1.0
        
        chi_grid = np.kron(chi_phi_grid, chi_r_grid)

        chi = 3.0 + 0.1j
        chi_grid *= chi
        
        E_struct = FDFD_halfsector.get_TM_field(J_center_dipole, chi_grid)
        E_struct_full = expand_symmetric_field(E_struct, n_sectors, Nr, mirror=True)
        
        J_full = expand_symmetric_field(J_center_dipole, n_sectors, Nr, mirror=True)
        chi_grid_full = expand_symmetric_field(chi_grid, n_sectors, Nr, mirror=True)
        E_full_reference = FDFD_full.get_TM_field(J_full, chi_grid_full)
        
        rel_err = np.linalg.norm(E_struct_full - E_full_reference)
        rel_err /= np.linalg.norm(E_struct_full)
        assert rel_err < 1e-8, f"Relative error {rel_err} too large."
    
    
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

        geometry = PolarFDFDGeometry(
            Nphi=Nphi,
            Nr=Nr,
            Npml=Npml,
            dr=dr,
            n_sectors=n_sectors,
            r_inner=0.0,
            Npml_inner=0,
            mirror=False,
            bloch_phase=0.0,
            m=3,
            lnR=-16.0,
        )
        solver = TM_Polar_FDFD(omega, geometry)

        r_inner_des = 0.3
        r_outer_des = 1.0
        design_mask = np.zeros((Nr, Nphi), dtype=bool)
        for ir, r in enumerate(solver.geometry.r_grid):
            if r_inner_des <= r <= r_outer_des:
                design_mask[ir, :] = True

        observe_mask = design_mask.copy()

        return solver, design_mask, observe_mask

    def test_greens_function_vs_direct_solve(self, greens_setup):
        """Test that Green's function matches direct solve results."""
        solver, design_mask, observe_mask = greens_setup
        G = solver.get_TM_Gba(design_mask, observe_mask)

        area_vec = solver.geometry.get_pixel_areas()
        design_lin = np.nonzero(design_mask.flatten(order="F"))[0]
        observe_lin = np.nonzero(observe_mask.flatten(order="F"))[0]

        solve = spla.factorized(solver.M0.tocsc())

        # Test a few source locations
        N_des = len(design_lin)
        test_indices = [0, N_des // 4, N_des // 2]

        for des_idx in test_indices:
            pixel_global = design_lin[des_idx]

            # Direct solve
            J_full = np.zeros(solver.geometry.Nphi * solver.geometry.Nr, dtype=complex)
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
        G = solver.get_TM_Gba(design_mask, observe_mask)

        area_vec = solver.geometry.get_pixel_areas()
        design_lin = np.nonzero(design_mask.flatten(order="F"))[0]
        observe_lin = np.nonzero(observe_mask.flatten(order="F"))[0]

        solve = spla.factorized(solver.M0.tocsc())

        N_des = len(design_lin)
        # Choose two well-separated source locations
        idx1, idx2 = N_des // 5, 4 * N_des // 5
        pixel1, pixel2 = design_lin[idx1], design_lin[idx2]

        # Direct solve with both sources
        J_full = np.zeros(solver.geometry.Nphi * solver.geometry.Nr, dtype=complex)
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

        geometry = PolarFDFDGeometry(
            Nphi=Nphi,
            Nr=Nr,
            Npml=Npml,
            dr=dr,
            n_sectors=n_sectors,
            r_inner=0.0,
            Npml_inner=0,
            mirror=False,
            bloch_phase=0.0,
            m=3,
            lnR=-16.0,
        )
        solver = TM_Polar_FDFD(omega, geometry)

        # Design region: inner annulus
        design_mask = np.zeros((Nr, Nphi), dtype=bool)
        for ir, r in enumerate(solver.geometry.r_grid):
            if 0.3 <= r <= 0.6:
                design_mask[ir, :] = True

        # Observe region: outer annulus (non-overlapping)
        observe_mask = np.zeros((Nr, Nphi), dtype=bool)
        for ir, r in enumerate(solver.geometry.r_grid):
            if 0.8 <= r <= 1.2:
                observe_mask[ir, :] = True

        G = solver.get_TM_Gba(design_mask, observe_mask)

        area_vec = solver.geometry.get_pixel_areas()
        design_lin = np.nonzero(design_mask.flatten(order="F"))[0]
        observe_lin = np.nonzero(observe_mask.flatten(order="F"))[0]

        solve = spla.factorized(solver.M0.tocsc())

        N_des = len(design_lin)
        des_idx = N_des // 2
        pixel_global = design_lin[des_idx]

        # Direct solve
        J_full = np.zeros(solver.geometry.Nphi * solver.geometry.Nr, dtype=complex)
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
        area_vec = solver.geometry.get_pixel_areas()

        p1 = solver.geometry.Nr // 3
        p2 = 2 * solver.geometry.Nr // 3
        A1 = area_vec[p1]
        A2 = area_vec[p2]

        # Field at 2 due to unit dipole moment source at 1
        J1 = np.zeros(solver.geometry.Nphi * solver.geometry.Nr, dtype=complex)
        J1[p1] = 1.0 / A1
        E21 = solver.get_TM_field(J1)[p2]

        # Field at 1 due to unit dipole moment source at 2
        J2 = np.zeros(solver.geometry.Nphi * solver.geometry.Nr, dtype=complex)
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

        geometry = PolarFDFDGeometry(
            Nphi=Nphi,
            Nr=Nr,
            Npml=Npml,
            dr=dr,
            n_sectors=n_sectors,
            r_inner=0.0,
            Npml_inner=0,
            mirror=False,
            bloch_phase=0.0,
            m=3,
            lnR=-16.0,
        )
        solver = TM_Polar_FDFD(omega, geometry)

        design_mask = np.zeros((Nr, Nphi), dtype=bool)
        design_mask[5:12, 10:18] = True

        return solver, design_mask

    def test_gaainv_identity(self, gaainv_setup):
        """Test that GaaInv @ Gaa is identity."""
        solver, design_mask = gaainv_setup
        GaaInv, _ = solver.get_GaaInv(design_mask)
        Gaa = solver.get_TM_Gba(design_mask, design_mask)

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

        geometry = PolarFDFDGeometry(
            Nphi=Nphi,
            Nr=Nr,
            Npml=Npml,
            dr=dr,
            n_sectors=n_sectors,
            r_inner=0.0,
            Npml_inner=0,
            mirror=False,
            bloch_phase=0.0,
            m=3,
            lnR=-16.0,
        )
        solver = TM_Polar_FDFD(omega, geometry)

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

        geometry = PolarFDFDGeometry(
            Nphi=Nphi,
            Nr=Nr,
            Npml=Npml,
            dr=dr,
            n_sectors=n_sectors,
            r_inner=0.0,
            Npml_inner=0,
            mirror=False,
            bloch_phase=0.0,
            m=m,
            lnR=-16.0,
        )
        solver = TM_Polar_FDFD(omega, geometry)

        # use a symmetric ring source at the first radial bin
        # to excite only the m=0 mode (cylindrical wave), matching H0(kr).
        J = np.zeros((Nr, Nphi), dtype=complex)
        area_2d = solver.geometry.get_pixel_areas().reshape((Nr, Nphi), order="F")
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
        r_phys = solver.geometry.r_grid[skip:idx_interface]
        E_phys = Ez_radial[skip:idx_interface]
        E_anal = np.abs((omega / 4) * hankel1(0, omega * r_phys))

        rel_error = np.linalg.norm(E_phys - E_anal) / np.linalg.norm(E_anal)

        assert rel_error < 0.05, (
            f"PML (m={m}) reflection check failed: error {rel_error}"
        )
    
    def test_inner_pml(self):
        """test consistency of nonpml fields when r_inner>0 and inner pml present"""
        omega = 2*np.pi
        r_inner = 5.0
        r_center = 3.0
        w_pml_thin = 0.5
        w_pml_thick = 1.0
        
        r_delta = w_pml_thin + r_center + w_pml_thick
        r_outer = r_inner + r_delta
        
        gpr = 40
        dr = 1.0 / gpr
        Nr = int(r_delta / dr)
        
        n_sectors = 6
        Nphi_sector = int(2*np.pi*r_outer / n_sectors / dr)
        
        Npml_thin = int(w_pml_thin / dr)
        Npml_thick = int(w_pml_thick / dr)
        
        # flip thickness of inner and outer pml and compare consistency of fields
        geo_1 = PolarFDFDGeometry(Nphi_sector, Nr, Npml_thin, dr, 
                                  n_sectors=n_sectors,
                                  r_inner=r_inner,
                                  Npml_inner=Npml_thick)
        
        FDFD_1 = TM_Polar_FDFD(omega, geo_1)
        
        geo_2 = PolarFDFDGeometry(Nphi_sector, Nr, Npml_thick, dr, 
                                  n_sectors=n_sectors,
                                  r_inner=r_inner,
                                  Npml_inner=Npml_thin)
        
        FDFD_2 = TM_Polar_FDFD(omega, geo_2)
        
        phi_grid_sector, r_grid, phi_grid_full = FDFD_1.get_symmetric_grids()
        
        J_r_ind = Npml_thick + gpr // 4
        J_m = n_sectors * 4 # adjust wave order here
        J_rgrid = np.zeros(Nr, dtype=complex)
        J_rgrid[J_r_ind] = 1.0 / (2*np.pi*r_grid[J_r_ind]*dr)
        J_phigrid = np.cos(phi_grid_sector * J_m)
        J_grid = np.kron(J_phigrid, J_rgrid)
        

        E_IPML_1 = FDFD_1.get_TM_field(J_grid)
        E_IPML_2 = FDFD_2.get_TM_field(J_grid)
        
        
        test_ind_l = Npml_thick + 10
        test_ind_r = Nr - (Npml_thick + 10)

        rel_err = np.linalg.norm(E_IPML_1[test_ind_l:test_ind_r] - E_IPML_2[test_ind_l:test_ind_r])
        rel_err /= np.linalg.norm(E_IPML_1[test_ind_l:-test_ind_r])
        assert rel_err < 1e-3, f"inner pml relative err{rel_err} too large."
