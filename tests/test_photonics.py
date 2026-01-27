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

    def test_sparse_dense_objective_consistency(self, setup_params):
        """Check that sparse and dense objectives give the same value for the same dof."""
        p = setup_params

        # Initialize sparse problem
        prob_sparse = Photonics_TM_FDFD(
            omega=p["omega"],
            geometry=p["geometry"],
            chi=p["chi"],
            chi_background=p["chi_background"],
            des_mask=p["des_mask"],
            ji=p["ji"],
            sparseQCQP=True,
        )
        prob_sparse.get_ei(p["ji"], update=True)
        prob_sparse.set_objective(
            A0=p["A0"], s0=p["s0"], c0=p["c0"], denseToSparse=True
        )

        # Initialize dense problem
        prob_dense = Photonics_TM_FDFD(
            omega=p["omega"],
            geometry=p["geometry"],
            chi=p["chi"],
            chi_background=p["chi_background"],
            des_mask=p["des_mask"],
            ji=p["ji"],
            sparseQCQP=False,
        )
        prob_dense.get_ei(p["ji"], update=True)
        prob_dense.set_objective(
            A0=p["A0"], s0=p["s0"], c0=p["c0"], denseToSparse=False
        )

        ndof = int(np.sum(p["des_mask"]))
        dof = np.random.rand(ndof)  # Random structure

        obj_sparse = prob_sparse.structure_objective(dof, [])
        obj_dense = prob_dense.structure_objective(dof, [])

        assert np.isclose(obj_sparse, obj_dense, atol=1e-10), (
            f"Sparse and dense objectives mismatch! "
            f"Sparse: {obj_sparse}, Dense: {obj_dense}"
        )

    @pytest.fixture
    def setup_ldos_params(self):
        """
        Small 2D TM LDOS setup (delta-current source + rectangular design region).
        Tuned to be fast but nontrivial.
        """
        wavelength = 1.0
        omega = 2 * np.pi / wavelength
        chi = 4.0 + 1e-4j

        px_per_length = 18
        dl = 1 / px_per_length

        Npmlsep = int(0.25 / dl)
        Npmlx = Npmly = int(0.25 / dl)

        Mx = My = int(0.35 / dl)
        Dx = int(0.10 / dl)

        Nx = int(2 * Npmlx + 2 * Npmlsep + Dx + Mx)
        Ny = int(2 * Npmly + 2 * Npmlsep + My)

        cx, cy = Npmlx + Npmlsep, Ny // 2

        ji = np.zeros((Nx, Ny), dtype=complex)
        ji[cx, cy] = 1.0 / dl**2  # 2D delta approx so ∫J dA ≈ 1

        des_mask = np.zeros((Nx, Ny), dtype=bool)
        des_mask[
            Npmlx + Npmlsep + Dx : Npmlx + Npmlsep + Dx + Mx,
            Npmly + Npmlsep : Npmly + Npmlsep + My,
        ] = True
        ndof = int(np.sum(des_mask))

        chi_background = np.zeros((Nx, Ny), dtype=complex)

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

        return dict(
            omega=omega,
            chi=chi,
            dl=dl,
            geometry=geometry,
            ji=ji,
            des_mask=des_mask,
            ndof=ndof,
            chi_background=chi_background,
        )

    def test_ldos_gradient_matches_reciprocity_sparse_and_dense(
        self, setup_ldos_params
    ):
        """
        LDOS objective:
            LDOS = -1/2 Re( <J, E> )  (with area integration)
        Analytical reciprocity gradient (w.r.t. dof scaling chi in the design region):
            d(LDOS)/d(dof) = -1/2 * Im( (chi*omega) * E^2 ) * area
        Compare against structure_objective gradients for both sparse and dense QCQP modes
        after setting (A0,s0,c0) as in examples/limits/LDOS.ipynb.
        """
        p = setup_ldos_params
        omega, chi = p["omega"], p["chi"]
        ji, des_mask = p["ji"], p["des_mask"]
        ndof = p["ndof"]

        rng = np.random.default_rng(0)
        dof = rng.random(ndof)

        # --- Build sparse problem and set LDOS objective (notebook convention) ---
        prob_sparse = Photonics_TM_FDFD(
            omega=omega,
            geometry=p["geometry"],
            chi=chi,
            des_mask=des_mask,
            ji=ji,
            chi_background=p["chi_background"],
            sparseQCQP=True,
        )
        ei = prob_sparse.get_ei(ji, update=True)

        # Areas are returned flattened by geometry; stay flattened and match solver order
        areas_flat = prob_sparse.geometry.get_pixel_areas()
        flatten_order = getattr(prob_sparse, "_flatten_order", "C")  # "C" for Cartesian, "F" for Polar
        ji_flat = ji.flatten(order=flatten_order)
        ei_flat = ei.flatten(order=flatten_order)

        vac_ldos = -0.5 * np.sum(np.real(ji_flat.conj() * ei_flat) * areas_flat)

        ei_design = ei[des_mask]
        s0_p = -(1 / 4) * 1j * omega * ei_design.conj()
        A0_sparse = sp.csc_array((ndof, ndof), dtype=complex)
        prob_sparse.set_objective(
            s0=s0_p, A0=A0_sparse, c0=vac_ldos, denseToSparse=True
        )

        grad_sparse = np.zeros(ndof, dtype=float)
        obj_sparse = prob_sparse.structure_objective(dof, grad_sparse)

        # --- Build dense problem and set LDOS objective (notebook convention) ---
        prob_dense = Photonics_TM_FDFD(
            omega=omega,
            geometry=p["geometry"],
            chi=chi,
            des_mask=des_mask,
            ji=ji,
            chi_background=p["chi_background"],
            sparseQCQP=False,
        )
        prob_dense.get_ei(ji, update=True)  # keep consistent incident field definition
        A0_dense = np.zeros((ndof, ndof), dtype=complex)
        prob_dense.set_objective(s0=s0_p, A0=A0_dense, c0=vac_ldos, denseToSparse=False)

        grad_dense = np.zeros(ndof, dtype=float)
        obj_dense = prob_dense.structure_objective(dof, grad_dense)

        # --- Direct total-field solve for analytical reciprocity gradient ---
        chigrid_dof = np.zeros_like(ji, dtype=complex)
        chigrid_dof[des_mask] = dof * chi

        assert prob_sparse.M is not None
        assert prob_sparse.EM_solver is not None
        M_tot = prob_sparse.M + prob_sparse.EM_solver._get_diagM_from_chigrid(
            chigrid_dof
        )
        E_tot = spla.spsolve(M_tot, 1j * omega * ji.flatten()).reshape(ji.shape)
        E_tot_flat = E_tot.flatten(order=flatten_order)

        ldos_direct = -0.5 * np.sum(np.real(ji_flat.conj() * E_tot_flat) * areas_flat)

        des_mask_flat = des_mask.flatten(order=flatten_order)
        areas_des = areas_flat[des_mask_flat]
        grad_analytic = (
            -0.5 * np.imag((chi * omega) * (E_tot[des_mask] ** 2)) * areas_des
        ).astype(float)

        # Sanity: both objective paths should represent the same LDOS
        assert np.isclose(obj_sparse, ldos_direct, rtol=5e-3, atol=1e-8)
        assert np.isclose(obj_dense, ldos_direct, rtol=5e-3, atol=1e-8)

        # Compare gradients (tolerances account for sparse solves + PML conditioning)
        def rel_err(a, b):
            denom = max(np.linalg.norm(b), 1e-14)
            return np.linalg.norm(a - b) / denom

        assert rel_err(grad_sparse, grad_analytic) < 5e-3, (
            f"Sparse LDOS gradient mismatch: relerr={rel_err(grad_sparse, grad_analytic)}"
        )
        assert rel_err(grad_dense, grad_analytic) < 5e-3, (
            f"Dense LDOS gradient mismatch: relerr={rel_err(grad_dense, grad_analytic)}"
        )
