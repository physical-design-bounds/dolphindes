"""Tests for the Maxwell module in dolphindes."""

import numpy as np
import pytest
import scipy.sparse as sp

from dolphindes.geometry import CartesianFDFDGeometry
from dolphindes.maxwell import TM_FDFD


class TestTMFDFD:
    """Tests for TM_FDFD Cartesian solver."""

    @pytest.fixture
    def solver_setup(self):
        """Set up solver and common parameters."""
        wvlgth = 1.0
        omega = 2 * np.pi / wvlgth
        gpr = 40
        dl = 1 / gpr
        Nx, Ny = int(2.0 * gpr), int(2.0 * gpr)
        Npmlx, Npmly = int(0.5 * gpr), int(0.5 * gpr)

        geometry = CartesianFDFDGeometry(Nx, Ny, Npmlx, Npmly, dl, dl)
        simulation = TM_FDFD(omega, geometry)

        A_mask = np.zeros((Nx, Ny), dtype=bool)
        A_mask[Nx // 4 : -Nx // 4, Ny // 4 : -Ny // 4] = True

        return simulation, A_mask, dl

    def test_greens_function_agreement(self, solver_setup):
        """Test agreement between TM_FDFD dipole solve and Gaa."""
        simulation, A_mask, dl = solver_setup
        Nx, Ny = simulation.Nx, simulation.Ny

        # Direct simulation of a dipole
        Ez_simulation = simulation.get_TM_dipole_field(Nx // 2, Ny // 2)

        # Green's function calculation
        Gaa = simulation.get_TM_Gba(A_mask, A_mask)
        sourcegrid = np.zeros((Nx, Ny), dtype=complex)
        sourcegrid[Nx // 2, Ny // 2] = 1.0 / dl / dl

        # Calculate Ez from Gaa
        # Note: Inverse of scaling factor applied in get_TM_Gba
        scale_factor = (-1j * simulation.k / simulation.ETA_0) ** -1
        Ez_from_G = scale_factor * (Gaa @ sourcegrid[A_mask])

        # Compare fields only in the masked region
        Ez_sim_masked = Ez_simulation[A_mask]

        assert np.allclose(Ez_from_G, Ez_sim_masked, atol=1e-6), (
            "Ez from Gaa does not match Ez from simulation"
        )

    def test_gaainv_identity(self, solver_setup):
        """Test that GaaInv @ Gaa approximates the identity matrix."""
        simulation, A_mask, _ = solver_setup

        Gaa = simulation.get_TM_Gba(A_mask, A_mask)
        GaaInv, M = simulation.get_GaaInv(A_mask)

        assert isinstance(GaaInv, sp.csc_array)
        assert isinstance(M, sp.csc_array)

        n_points = np.sum(A_mask)
        assert GaaInv.shape == (n_points, n_points), "GaaInv shape mismatch"

        identity_approx = GaaInv @ Gaa
        assert np.allclose(identity_approx, np.eye(n_points), atol=1e-3), (
            "GaaInv @ Gaa does not equal identity matrix"
        )
