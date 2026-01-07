import numpy as np
import scipy.sparse as sp

from dolphindes.geometry import CartesianFDFDGeometry
from dolphindes.photonics import Photonics_TM_FDFD


def test_Photonics_TM_FDFD_adjoint():
    """
    Use a planewave absorption problem to test setup and adjoint gradient
    of Photonics_TM_FDFD
    """
    ## wavelength, geometry and materials of the planewave absorption problem ##
    wavelength = 1.0
    omega = 2 * np.pi / wavelength

    chi = 3 + 1e-2j

    px_per_length = 20  # pixels per length unit. If wavelength = 1.0, then this is pixels per wavelength.
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
        Npmlx + Npmlsepx : -(Npmlx + Npmlsepx), Npmly + Npmlsepy : -(Npmly + Npmlsepy)
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
        Nx=Nx, Ny=Ny, Npmlx=Npmlx, Npmly=Npmly, dx=dl, dy=dl, bloch_x=0.0, bloch_y=0.0
    )

    ## setup Photonics_TM_FDFD objects
    abs_problem_sparse = Photonics_TM_FDFD(
        omega=omega,
        geometry=geometry,
        chi=chi,
        chi_background=chi_background,
        des_mask=des_mask,
        ji=ji,
        sparseQCQP=True,
    )
    abs_problem_sparse.get_ei(ji, update=True)
    abs_problem_sparse.set_objective(A0=A0_p, s0=s0_p, c0=c0, denseToSparse=True)

    abs_problem_dense = Photonics_TM_FDFD(
        omega=omega,
        geometry=geometry,
        chi=chi,
        chi_background=chi_background,
        des_mask=des_mask,
        ji=ji,
        sparseQCQP=False,
    )
    abs_problem_dense.get_ei(ji, update=True)
    abs_problem_dense.set_objective(A0=A0_p, s0=s0_p, c0=c0, denseToSparse=False)

    ## check that dense setup_EM_operators works for non-vacuum chi_background
    test_src = np.zeros((Nx, Ny), dtype=complex)
    test_src[Nx // 2, Ny // 2] = 1.0 / dl**2
    M_E_des = sp.linalg.spsolve(abs_problem_dense.M, 1j * omega * test_src.flatten())[
        des_mask.flatten()
    ]
    G_E_des = (1j / omega) * abs_problem_dense.G @ test_src[des_mask]
    assert np.allclose(np.linalg.norm(M_E_des - G_E_des), 0, atol=1e-10), (
        "dense setup_EM_operators M and G do not line up."
    )

    ## compare adjoint with finite differences
    ndof = int(np.sum(des_mask))
    dof = 0.5 * np.ones(ndof)  # half slab initialization
    grad = np.zeros(ndof)

    ind = np.random.randint(ndof)
    delta = 1e-3

    obj0 = abs_problem_sparse.structure_objective(dof, grad)
    dof[ind] += delta
    obj1 = abs_problem_sparse.structure_objective(dof, [])
    assert np.allclose((obj1 - obj0) / delta, grad[ind], rtol=delta * 3), (
        "Sparse objective gradient failed finite difference test."
    )

    obj0 = abs_problem_dense.structure_objective(dof, grad)
    dof[ind] += delta
    obj1 = abs_problem_dense.structure_objective(dof, [])
    assert np.allclose((obj1 - obj0) / delta, grad[ind], rtol=delta * 3), (
        "Dense objective gradient failed finite difference test."
    )
