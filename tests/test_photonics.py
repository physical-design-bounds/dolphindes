import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dolphindes.geometry import CartesianFDFDGeometry
from dolphindes.photonics import Photonics_TM_FDFD


class TestPhotonicsAdjoint:
    """Tests for gradient accuracy and operator consistency in Photonics_TM_FDFD."""

    @pytest.fixture
    def setup_params(self):
        """
        Set up a planewave absorption problem parameters.
        Returns a dictionary of parameters.
        """
        ## wavelength, geometry and materials of the planewave absorption problem ##
        wavelength = 1.0
        omega = 2 * np.pi / wavelength

        chi = 3 + 1e-2j

        px_per_length = 20  # pixels per length unit
        dl = 1 / px_per_length

        des_x = 1.5
        des_y = 1.5  # size of the design region for the absorbing structure
        pmlsep = 0.5
        pmlthick = 0.5
        Mx = int(des_x / dl)
        My = int(des_y / dl)

        Npmlsepx = Npmlsepy = int(pmlsep / dl)
        Npmlx = Npmly = int(pmlthick / dl)
        Nx = Mx + 2 * (Npmlsepx + Npmlx)
        Ny = My + 2 * (Npmlsepy + Npmly)

        des_mask = np.zeros((Nx, Ny), dtype=bool)
        des_mask[
            Npmlx + Npmlsepx : -(Npmlx + Npmlsepx),
            Npmly + Npmlsepy : -(Npmly + Npmlsepy),
        ] = True
        Ndes = int(np.sum(des_mask))

        ## add a non-vacuum background
        chi_background = np.zeros((Nx, Ny), dtype=complex)
        chi_background[Npmlx : Npmlx + Npmlsepx, Npmly:-Npmly] = 2 + 1e-1j

        ## planewave source
        ji = np.zeros((Nx, Ny), dtype=complex)
        ji[Npmlx, :] = (
            2.0 / dl
        )  # linesource for unit amplitude planewave traveeling in x direction

        ## absorption obective
        c0 = 0.0
        s0_p = np.zeros(Ndes, dtype=complex)
        A0_p = (omega / 2) * np.imag(1.0 / chi) * sp.eye_array(Ndes) * dl**2

        ## setup geometry
        geometry = CartesianFDFDGeometry(
            Nx=Nx,
            Ny=Ny,
            Npmlx=Npmlx,
            Npmly=Npmly,
            dx=dl,
            dy=dl,
            bloch_x=0.0,
            bloch_y=0.0,
        )

        return {
            "omega": omega,
            "geometry": geometry,
            "chi": chi,
            "chi_background": chi_background,
            "des_mask": des_mask,
            "ji": ji,
            "A0": A0_p,
            "s0": s0_p,
            "c0": c0,
            "dl": dl,
            "Nx": Nx,
            "Ny": Ny,
        }

    def test_dense_operators_consistency(self, setup_params):
        """Check that dense setup_EM_operators G matches solver M."""
        p = setup_params
        abs_problem_dense = Photonics_TM_FDFD(
            omega=p["omega"],
            geometry=p["geometry"],
            chi=p["chi"],
            chi_background=p["chi_background"],
            des_mask=p["des_mask"],
            ji=p["ji"],
            sparseQCQP=False,
        )

        dl = p["dl"]
        Nx, Ny = p["Nx"], p["Ny"]
        omega = p["omega"]
        des_mask = p["des_mask"]

        test_src = np.zeros((Nx, Ny), dtype=complex)
        test_src[Nx // 2, Ny // 2] = 1.0 / dl**2

        # Solve using M
        M_E_des = spla.spsolve(abs_problem_dense.M, 1j * omega * test_src.flatten())[
            des_mask.flatten()
        ]

        # Solve using G
        G_E_des = (1j / omega) * abs_problem_dense.G @ test_src[des_mask]

        assert np.allclose(np.linalg.norm(M_E_des - G_E_des), 0, atol=1e-10), (
            "dense setup_EM_operators M and G do not line up."
        )

    @pytest.mark.parametrize("sparse_mode", [True, False])
    def test_adjoint_gradient_finite_difference(self, setup_params, sparse_mode):
        """Compare adjoint gradient with finite differences."""
        p = setup_params
        problem = Photonics_TM_FDFD(
            omega=p["omega"],
            geometry=p["geometry"],
            chi=p["chi"],
            chi_background=p["chi_background"],
            des_mask=p["des_mask"],
            ji=p["ji"],
            sparseQCQP=sparse_mode,
        )
        problem.get_ei(p["ji"], update=True)
        problem.set_objective(
            A0=p["A0"], s0=p["s0"], c0=p["c0"], denseToSparse=sparse_mode
        )

        ndof = int(np.sum(p["des_mask"]))
        dof = 0.5 * np.ones(ndof)  # half slab initialization
        grad = np.zeros(ndof)

        # Pick random index to test
        ind = np.random.randint(ndof)
        delta = 1e-3

        # Compute adjoint gradient
        obj0 = problem.structure_objective(dof, grad)

        # Compute finite difference
        dof[ind] += delta
        obj1 = problem.structure_objective(dof, [])

        fd_grad = (obj1 - obj0) / delta
        adj_grad = grad[ind]

        assert np.allclose(fd_grad, adj_grad, rtol=delta * 3), (
            f"{'Sparse' if sparse_mode else 'Dense'} objective gradient failed "
            f"finite difference test. FD: {fd_grad}, Adjoint: {adj_grad}"
        )
