"""Tests for the Maxwell module in dolphindes."""

import numpy as np
import scipy.sparse as sp

from dolphindes.maxwell import TM_FDFD


def test_maxwell_TM() -> None:
    """Test the Maxwell TM_FDFD module."""
    wvlgth = 1.0
    omega = 2 * np.pi / wvlgth
    gpr = 50
    dl = 1 / gpr
    Nx, Ny = int(3.0 * gpr), int(3.0 * gpr)
    Npmlx, Npmly = int(0.5 * gpr), int(0.5 * gpr)

    print("Testing agreement between TM_FDFD and Gaa for a dipole source")
    simulation = TM_FDFD(omega, Nx, Ny, Npmlx, Npmly, dl)
    Ez_simulation = simulation.get_TM_dipole_field(Nx // 2, Ny // 2)

    A_mask = np.zeros((Nx, Ny), dtype=bool)
    A_mask[Nx // 4 : -Nx // 4, Ny // 4 : -Ny // 4] = True
    Gaa = simulation.get_TM_Gba(A_mask, A_mask)
    sourcegrid = np.zeros((Nx, Ny), dtype=complex)
    sourcegrid[Nx // 2, Ny // 2] = 1.0 / dl / dl
    Ez = ((-1j * simulation.k / simulation.ETA_0) ** -1) * Gaa @ sourcegrid[A_mask]
    Ez_large = np.zeros((Nx, Ny), dtype=complex)
    Ez_large[A_mask] = Ez

    Ez_sim_large = np.zeros((Nx, Ny), dtype=complex)
    Ez_sim_large[A_mask] = Ez_simulation[A_mask]

    assert np.allclose(Ez_large, Ez_sim_large, atol=1e-6), (
        "Ez from Gaa does not match Ez from simulation"
    )

    print("Testing GaaInv and Gaa agreement")
    GaaInv, M = simulation.get_GaaInv(A_mask)
    assert type(GaaInv) is sp.csc_array
    assert type(M) is sp.csc_array
    assert GaaInv.shape == (np.sum(A_mask), np.sum(A_mask)), "GaaInv shape mismatch"

    assert np.allclose(GaaInv @ Gaa, np.eye(np.sum(A_mask)), atol=1e-6), (
        "GaaInv @ Gaa does not equal identity matrix"
    )
