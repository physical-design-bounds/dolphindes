"""Tests for Photonics_TM_Polar_FDFD class."""

import numpy as np
import pytest
import scipy.sparse as sp

from dolphindes.photonics import Photonics_TM_Polar_FDFD, PolarFDFDGeometry


class TestPhotonicsTMPolarFDFD:
    """Test Photonics_TM_Polar_FDFD class functionality."""

    @pytest.fixture
    def absorption_problem_setup(self):
        """Set up a radial absorption problem for testing."""
        # Parameters
        wavelength = 1.0
        omega = 2 * np.pi / wavelength
        chi = 3 + 1e-2j

        # Grid parameters
        dr = 0.05
        Nr = 40
        Nphi = 50
        Npml = 10
        n_sectors = 1

        # Design region: annular region away from center and PML
        r_grid = (np.arange(Nr) + 0.5) * dr
        r_inner_des = 0.3
        r_outer_des = 1.0

        des_mask = np.zeros((Nr, Nphi), dtype=bool)
        for ir, r in enumerate(r_grid):
            if r_inner_des <= r <= r_outer_des:
                des_mask[ir, :] = True

        Ndes = int(np.sum(des_mask))

        # Point source at center-ish
        ji = np.zeros(Nr * Nphi, dtype=complex)
        # Place source inside non-PML region but outside design region
        ir_src = 3  # close to center
        iphi_src = Nphi // 2
        area_r = r_grid * dr * (2 * np.pi / n_sectors / Nphi)
        area_vec = np.kron(np.ones(Nphi), area_r)
        idx_src = iphi_src * Nr + ir_src
        ji[idx_src] = 1.0 / area_vec[idx_src]

        # Absorption objective (in polarization basis)
        c0 = 0.0
        s0_p = np.zeros(Ndes, dtype=complex)
        A0_p = (omega / 2) * np.imag(1.0 / chi) * sp.eye_array(Ndes)

        # Geometry
        geometry = PolarFDFDGeometry(
            Nphi=Nphi,
            Nr=Nr,
            Npml=Npml,
            dr=dr,
            n_sectors=n_sectors,
            bloch_phase=0.0,
        )

        return {
            "omega": omega,
            "chi": chi,
            "geometry": geometry,
            "des_mask": des_mask,
            "ji": ji,
            "A0_p": A0_p,
            "s0_p": s0_p,
            "c0": c0,
            "Ndes": Ndes,
        }

    def test_instantiation(self, absorption_problem_setup):
        """Test that solver instantiates correctly."""
        setup = absorption_problem_setup

        problem = Photonics_TM_Polar_FDFD(
            omega=setup["omega"],
            geometry=setup["geometry"],
            chi=setup["chi"],
            des_mask=setup["des_mask"],
            ji=setup["ji"],
            sparseQCQP=True,
        )

        assert problem.EM_solver is not None
        assert problem.Ginv is not None
        assert problem.M is not None

    def test_get_ei(self, absorption_problem_setup):
        """Test incident field computation."""
        setup = absorption_problem_setup

        problem = Photonics_TM_Polar_FDFD(
            omega=setup["omega"],
            geometry=setup["geometry"],
            chi=setup["chi"],
            des_mask=setup["des_mask"],
            ji=setup["ji"],
            sparseQCQP=True,
        )

        ei = problem.get_ei(setup["ji"], update=True)
        assert ei is not None
        assert len(ei) == setup["geometry"].Nr * setup["geometry"].Nphi
        assert not np.allclose(ei, 0)

    def test_setup_qcqp_sparse(self, absorption_problem_setup):
        """Test QCQP setup with sparse formulation."""
        setup = absorption_problem_setup

        problem = Photonics_TM_Polar_FDFD(
            omega=setup["omega"],
            geometry=setup["geometry"],
            chi=setup["chi"],
            des_mask=setup["des_mask"],
            ji=setup["ji"],
            sparseQCQP=True,
        )
        problem.get_ei(setup["ji"], update=True)
        problem.set_objective(
            A0=setup["A0_p"], s0=setup["s0_p"], c0=setup["c0"], denseToSparse=True
        )
        problem.setup_QCQP()

        assert problem.QCQP is not None
        assert problem.Ndes == setup["Ndes"]

    def test_bound_qcqp(self, absorption_problem_setup):
        """Test that QCQP bound can be computed."""
        setup = absorption_problem_setup

        problem = Photonics_TM_Polar_FDFD(
            omega=setup["omega"],
            geometry=setup["geometry"],
            chi=setup["chi"],
            des_mask=setup["des_mask"],
            ji=setup["ji"],
            sparseQCQP=True,
        )
        problem.get_ei(setup["ji"], update=True)
        problem.set_objective(
            A0=setup["A0_p"], s0=setup["s0_p"], c0=setup["c0"], denseToSparse=True
        )
        problem.setup_QCQP()

        result = problem.bound_QCQP(method="bfgs")
        dual_value, lags, grad, hess, xstar = result

        assert isinstance(dual_value, float)
        assert not np.isnan(dual_value)
        assert xstar is not None

    def test_get_chi_inf(self, absorption_problem_setup):
        """Test inferred chi computation after solving."""
        setup = absorption_problem_setup

        problem = Photonics_TM_Polar_FDFD(
            omega=setup["omega"],
            geometry=setup["geometry"],
            chi=setup["chi"],
            des_mask=setup["des_mask"],
            ji=setup["ji"],
            sparseQCQP=True,
        )
        problem.get_ei(setup["ji"], update=True)
        problem.set_objective(
            A0=setup["A0_p"], s0=setup["s0_p"], c0=setup["c0"], denseToSparse=True
        )
        problem.setup_QCQP()
        problem.bound_QCQP(method="bfgs")

        chi_inf = problem.get_chi_inf()
        assert chi_inf.shape == (setup["Ndes"],)
        assert not np.any(np.isnan(chi_inf))

    def test_dense_setup_consistency(self, absorption_problem_setup):
        """Check that dense setup_EM_operators works (M and G consistency)."""
        setup = absorption_problem_setup

        # Create a problem with dense formulation and background chi
        chi_bg = np.zeros_like(setup["des_mask"], dtype=complex)
        # Add some background feature to ensure M is not just M0
        chi_bg[5, 5] = 1.0 + 0.1j

        problem = Photonics_TM_Polar_FDFD(
            omega=setup["omega"],
            geometry=setup["geometry"],
            chi=setup["chi"],
            des_mask=setup["des_mask"],
            ji=setup["ji"],
            chi_background=chi_bg,
            sparseQCQP=False,
        )

        # Setup operators (creates M and G)
        problem.setup_EM_operators()

        # Test source inside design region
        Nr = setup["geometry"].Nr
        Nphi = setup["geometry"].Nphi
        test_src = np.zeros((Nr, Nphi), dtype=complex)

        # Find a point inside design mask
        # des_mask is True for r_inner_des <= r <= r_outer_des
        ir = 10
        iphi = 10
        assert setup["des_mask"][ir, iphi]
        test_src[ir, iphi] = 1.0

        # Flatten source (F order for polar)
        src_flat = test_src.flatten(order="F")

        # Solve using M
        rhs = 1j * setup["omega"] * src_flat
        E_full = sp.linalg.spsolve(problem.M, rhs)
        M_E_des = E_full[setup["des_mask"].flatten(order="F")]

        # Solve using G
        src_des_flat = test_src.flatten(order="F")[setup["des_mask"].flatten(order="F")]
        G_E_des = (1j / setup["omega"]) * problem.G @ src_des_flat

        assert np.allclose(M_E_des, G_E_des, atol=1e-8), (
            "Dense M and G operators are inconsistent."
        )

    def test_sparse_vs_dense_limit(self, absorption_problem_setup):
        """Verify that sparse and dense formulations yield the same dual bound."""
        setup = absorption_problem_setup

        # 1. Sparse
        prob_sparse = Photonics_TM_Polar_FDFD(
            omega=setup["omega"],
            geometry=setup["geometry"],
            chi=setup["chi"],
            des_mask=setup["des_mask"],
            ji=setup["ji"],
            sparseQCQP=True,
        )
        prob_sparse.get_ei(setup["ji"], update=True)
        prob_sparse.set_objective(
            A0=setup["A0_p"], s0=setup["s0_p"], c0=setup["c0"], denseToSparse=True
        )
        prob_sparse.setup_QCQP()
        val_sparse, _, _, _, _ = prob_sparse.bound_QCQP(method="bfgs")

        # 2. Dense
        prob_dense = Photonics_TM_Polar_FDFD(
            omega=setup["omega"],
            geometry=setup["geometry"],
            chi=setup["chi"],
            des_mask=setup["des_mask"],
            ji=setup["ji"],
            sparseQCQP=False,
        )
        prob_dense.get_ei(setup["ji"], update=True)
        prob_dense.set_objective(
            A0=setup["A0_p"], s0=setup["s0_p"], c0=setup["c0"], denseToSparse=False
        )
        prob_dense.setup_QCQP()
        val_dense, _, _, _, _ = prob_dense.bound_QCQP(method="bfgs")

        assert np.isclose(val_sparse, val_dense, rtol=1e-3), (
            f"Sparse ({val_sparse}) and dense ({val_dense}) dual bounds do not match."
        )
