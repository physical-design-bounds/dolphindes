"""
Concrete implementations of photonics QCQP design bounds solvers.

Provides TM and TE polarization FDFD solvers that bridge the QCQP Dual Problem
Interface in cvxopt and the Maxwell Solvers in maxwell.
"""

__all__ = [
    "Photonics_TM_FDFD",
    "Photonics_TM_Polar_FDFD",
    "Photonics_TE_Yee_FDFD",
    "chi_to_feasible_rho",
]

import warnings
from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from dolphindes.cvxopt import DenseSharedProjQCQP, SparseSharedProjQCQP
from dolphindes.maxwell import TM_FDFD, TM_Polar_FDFD
from dolphindes.types import BoolGrid, ComplexArray, ComplexGrid
from dolphindes.util import check_attributes

from ._base_photonics import (
    CartesianFDFDGeometry,
    Photonics_FDFD,
    PolarFDFDGeometry,
)


class Photonics_TM_FDFD(Photonics_FDFD):
    """
    TM polarization FDFD photonics problem.

    Attributes
    ----------
    All attributes from Photonics_FDFD plus:
    Ginv : csc_array or None
        Inverse Green's function (sparse QCQP).
    G : ndarray of complex or None
        Green's function (dense QCQP).
    M : csc_array or None
        Maxwell operator.
    EM_solver : TM_FDFD or None
        Electromagnetic field solver.
    structure_objective : Callable
        Function for structure optimization objective.
    """

    def __init__(
        self,
        omega: complex,
        geometry: Optional[CartesianFDFDGeometry] = None,
        chi: Optional[complex] = None,
        des_mask: Optional[BoolGrid] = None,
        ji: Optional[ComplexGrid] = None,
        ei: Optional[ComplexGrid] = None,
        chi_background: Optional[ComplexGrid] = None,
        sparseQCQP: bool = True,
        A0: Optional[Union[ComplexArray, sp.csc_array]] = None,
        s0: Optional[ComplexArray] = None,
        c0: float = 0.0,
    ) -> None:
        """
        Initialize Photonics_TM_FDFD.

        Parameters
        ----------
        omega : complex
            Circular frequency.
        geometry : CartesianFDFDGeometry, optional
            Cartesian geometry specification.
        chi : complex, optional
            Bulk susceptibility of material.
        des_mask : ndarray of bool, optional
            Design region mask.
        ji : ndarray of complex, optional
            Incident current source.
        ei : ndarray of complex, optional
            Incident field.
        chi_background : ndarray of complex, optional
            Background structure susceptibility.
        sparseQCQP : bool, optional
            Flag for sparse QCQP formulation. Default: True
        A0 : ndarray or csc_array, optional
            Objective quadratic matrix.
        s0 : ndarray of complex, optional
            Objective linear vector.
        c0 : float, optional
            Objective constant. Default: 0.0
        """
        self.des_mask = des_mask
        self.Ginv: Optional[sp.csc_array] = None
        self.G: Optional[ComplexArray] = None
        self.M: Optional[sp.csc_array] = None
        self.EM_solver: Optional[TM_FDFD] = None
        self.Ndes: Optional[int] = None
        self.Plist: Optional[list] = None
        self.dense_s0: Optional[ComplexArray] = None

        super().__init__(
            omega,
            geometry,
            chi,
            des_mask,
            ji,
            ei,
            chi_background,
            sparseQCQP,
            A0,
            s0,
            c0,
        )

        try:
            check_attributes(self, "omega", "geometry", "chi", "des_mask", "sparseQCQP")
            check_attributes(
                self.geometry,
                "Nx",
                "Ny",
                "Npmlx",
                "Npmly",
                "dx",
                "dy",
                "bloch_x",
                "bloch_y",
            )
            self.setup_EM_solver()
            self.setup_EM_operators()
        except AttributeError:
            warnings.warn(
                "Photonics_TM_FDFD initialized with missing attributes "
                "(lazy initialization). "
                "We strongly recommend passing all arguments for expected behavior."
            )

        # structure adjoint
        self.structure_objective: Callable[[NDArray, NDArray], float]
        if sparseQCQP:
            self.structure_objective = self.structure_objective_sparse
        else:
            self.structure_objective = self.structure_objective_dense

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Photonics_TM_FDFD(omega={self.omega}, geometry={self.geometry}, chi={self.chi}, "
            f"des_mask={self.des_mask is not None}, ji={self.ji is not None}, "
            f"ei={self.ei is not None}, chi_background={self.chi_background is not None}, "
            f"sparseQCQP={self.sparseQCQP})"
        )

    def setup_EM_solver(self, geometry: Optional[CartesianFDFDGeometry] = None) -> None:
        """
        Set up the FDFD electromagnetic solver with given geometry.

        Parameters
        ----------
        geometry : CartesianFDFDGeometry, optional
            Geometry specification. If None, uses self.geometry.

        Notes
        -----
        Creates a TM_FDFD solver instance and stores it in self.EM_solver.
        """
        if geometry is not None:
            self.geometry = geometry

        assert self.geometry is not None
        check_attributes(
            self.geometry,
            "Nx",
            "Ny",
            "Npmlx",
            "Npmly",
            "dx",
            "dy",
            "bloch_x",
            "bloch_y",
        )
        self.EM_solver = TM_FDFD(
            self.omega,
            self.geometry.Nx,
            self.geometry.Ny,
            self.geometry.Npmlx,
            self.geometry.Npmly,
            self.geometry.dx,
            self.geometry.bloch_x,
            self.geometry.bloch_y,
        )

    def setup_EM_operators(self) -> None:
        """
        Set up electromagnetic operators for the design region and background.

        Notes
        -----
        This method creates the appropriate operators based on whether sparse or dense
        QCQP formulation is used:
        - For sparse QCQP: Creates Ginv (inverse Green's function) and M operators
        - For dense QCQP: Creates G (Green's function) operator

        Requires self.des_mask to be defined.

        Raises
        ------
        AttributeError
            If des_mask is not defined.
        """
        check_attributes(self, "des_mask")
        assert self.EM_solver is not None
        if self.sparseQCQP:
            self.Ginv, self.M = self.EM_solver.get_GaaInv(
                self.des_mask, self.chi_background
            )
        else:
            if self.chi_background is None:
                self.M = self.EM_solver.M0
                self.G = self.EM_solver.get_TM_Gba(self.des_mask, self.des_mask)
            else:
                self.M = self.EM_solver.M0 + self.EM_solver._get_diagM_from_chigrid(
                    self.chi_background
                )
                assert self.des_mask is not None
                Id = np.diag(self.des_mask.astype(complex).flatten())[
                    :, self.des_mask.flatten()
                ]
                self.G = (
                    self.omega**2
                    * np.linalg.solve(self.M.toarray(), Id)[self.des_mask.flatten(), :]
                )

    def _get_dof_chigrid_M_es(
        self, dof: NDArray[np.floating]
    ) -> Tuple[ComplexGrid, sp.csc_array, ComplexArray]:
        """
        Set up method for structure_objective_sparse and structure_objective_dense.

        Parameters
        ----------
        dof : ndarray of float
            Degrees of freedom.

        Returns
        -------
        tuple
            (chigrid_dof, M_dof, es) - susceptibility grid, Maxwell operator,
            scattered field.
        """
        assert self.geometry is not None
        assert self.des_mask is not None
        assert self.chi is not None
        assert self.EM_solver is not None
        assert self.M is not None
        assert self.ei is not None

        Nx, Ny = self.geometry.get_grid_size()
        chigrid_dof: ComplexGrid = np.zeros((Nx, Ny), dtype=complex)
        chigrid_dof[self.des_mask] = dof * self.chi
        M_dof = self.M + self.EM_solver._get_diagM_from_chigrid(chigrid_dof)
        es: ComplexArray = spla.spsolve(
            M_dof, self.omega**2 * (chigrid_dof * self.ei).flatten()
        )[self.des_mask.flatten()]

        return chigrid_dof, M_dof, es

    def structure_objective_sparse(
        self, dof: NDArray[np.floating], grad: NDArray[np.floating]
    ) -> float:
        """
        Structural optimization objective and gradient when sparseQCQP=True.

        Follows convention of the optimization package NLOPT: returns objective value
        and stores gradient with respect to objective in the input argument grad.

        Parameters
        ----------
        dof : ndarray of float
            Pixel-wise structure degrees of freedom over the design region as
            specified by self.des_mask.
            dof[j] is a linear interpolation between dof[j] = 0 (self.chi_background)
            and dof[j] = 1 (self.chi_background + self.chi)
        grad : ndarray of float
            Adjoint gradient of the design objective with respect to dof.
            Specify grad = [] if only the objective is needed.
            Otherwise, grad should be an array of the same size as dof; upon method
            exit grad will store the gradient.

        Returns
        -------
        obj : float
            The design objective for the structure specified by dof.
        """
        assert self.geometry is not None
        assert self.des_mask is not None
        assert self.A0 is not None
        assert self.s0 is not None
        assert self.ei is not None

        chigrid_dof, M_dof, es = self._get_dof_chigrid_M_es(dof)
        obj = np.real(-np.vdot(es, self.A0 @ es) + 2 * np.vdot(self.s0, es) + self.c0)

        if len(grad) > 0:
            Nx, Ny = self.geometry.get_grid_size()
            adj_src: ComplexGrid = np.zeros((Nx, Ny), dtype=complex)
            adj_src[self.des_mask] = np.conj(self.s0 - self.A0 @ es)
            adj_v: ComplexArray = spla.spsolve(M_dof, adj_src.flatten())[
                self.des_mask.flatten()
            ]
            grad[:] = 2 * np.real(
                self.omega**2 * self.chi * (adj_v * (self.ei[self.des_mask] + es))
            )

        return float(obj)

    def structure_objective_dense(
        self, dof: NDArray[np.floating], grad: NDArray[np.floating]
    ) -> float:
        """
        Structural optimization objective and gradient when sparseQCQP=False.

        Specifications exactly the same as structure_objective_sparse.

        Parameters
        ----------
        dof : ndarray of float
            Pixel-wise structure degrees of freedom.
        grad : ndarray of float
            Gradient storage array.

        Returns
        -------
        obj : float
            Design objective value.
        """
        assert self.geometry is not None
        assert self.des_mask is not None
        assert self.A0 is not None
        assert self.s0 is not None
        assert self.ei is not None
        assert self.chi is not None

        chigrid_dof, M_dof, es = self._get_dof_chigrid_M_es(dof)

        et = self.ei[self.des_mask] + es
        p = chigrid_dof[self.des_mask] * et

        obj = np.real(-np.vdot(p, self.A0 @ p) + 2 * np.vdot(self.s0, p) + self.c0)

        if len(grad) > 0:
            Nx, Ny = self.geometry.get_grid_size()
            adj_src: ComplexGrid = np.zeros((Nx, Ny), dtype=complex)
            adj_src[self.des_mask] = chigrid_dof[self.des_mask] * np.conj(
                self.s0 - self.A0 @ p
            )
            adj_v: ComplexArray = spla.spsolve(M_dof, adj_src.flatten())[
                self.des_mask.flatten()
            ]
            grad[:] = 2 * np.real(
                (
                    self.chi * np.conj(self.s0 - self.A0 @ p)
                    + self.omega**2 * self.chi * adj_v
                )
                * et
            )

        return float(obj)


class Photonics_TE_Yee_FDFD(Photonics_FDFD):
    """TE polarization FDFD photonics problem (placeholder)."""

    def __init__(self) -> None:
        """Initialize placeholder."""
        pass


class Photonics_TM_Polar_FDFD(Photonics_FDFD):
    """
    TM polarization FDFD photonics problem in polar coordinates.

    Attributes
    ----------
    All attributes from Photonics_FDFD plus:
    Ginv : csc_array or None
        Inverse Green's function (sparse QCQP).
    G : ndarray of complex or None
        Green's function (dense QCQP).
    M : csc_array or None
        Maxwell operator.
    EM_solver : TM_Polar_FDFD or None
        Electromagnetic field solver.
    structure_objective : Callable
        Function for structure optimization objective.
    """

    _flatten_order = "F"

    def __init__(
        self,
        omega: complex,
        geometry: Optional[PolarFDFDGeometry] = None,
        chi: Optional[complex] = None,
        des_mask: Optional[BoolGrid] = None,
        ji: Optional[ComplexGrid] = None,
        ei: Optional[ComplexGrid] = None,
        chi_background: Optional[ComplexGrid] = None,
        sparseQCQP: bool = True,
        A0: Optional[Union[ComplexArray, sp.csc_array]] = None,
        s0: Optional[ComplexArray] = None,
        c0: float = 0.0,
    ) -> None:
        """
        Initialize Photonics_TM_Polar_FDFD.

        Parameters
        ----------
        omega : complex
            Circular frequency.
        geometry : PolarFDFDGeometry, optional
            Polar geometry specification.
        chi : complex, optional
            Bulk susceptibility of material.
        des_mask : ndarray of bool, optional
            Design region mask (shape: Nr x Nphi).
        ji : ndarray of complex, optional
            Incident current source.
        ei : ndarray of complex, optional
            Incident field.
        chi_background : ndarray of complex, optional
            Background structure susceptibility.
        sparseQCQP : bool, optional
            Flag for sparse QCQP formulation. Default: True
        A0 : ndarray or csc_array, optional
            Objective quadratic matrix.
        s0 : ndarray of complex, optional
            Objective linear vector.
        c0 : float, optional
            Objective constant. Default: 0.0
        """
        self.des_mask = des_mask
        self.Ginv: Optional[sp.csc_array] = None
        self.G: Optional[ComplexArray] = None
        self.M: Optional[sp.csc_array] = None
        self.EM_solver: Optional[TM_Polar_FDFD] = None
        self.Ndes: Optional[int] = None
        self.Plist: Optional[list] = None
        self.dense_s0: Optional[ComplexArray] = None

        super().__init__(
            omega,
            geometry,
            chi,
            des_mask,
            ji,
            ei,
            chi_background,
            sparseQCQP,
            A0,
            s0,
            c0,
        )

        try:
            check_attributes(self, "omega", "geometry", "chi", "des_mask", "sparseQCQP")
            check_attributes(
                self.geometry,
                "Nr",
                "Nphi",
                "Npml",
                "dr",
                "n_sectors",
                "bloch_phase",
            )
            self.setup_EM_solver()
            self.setup_EM_operators()
        except AttributeError:
            warnings.warn(
                "Photonics_TM_Polar_FDFD initialized with missing attributes "
                "(lazy initialization). "
                "We strongly recommend passing all arguments for expected behavior."
            )

        # structure adjoint
        self.structure_objective: Callable[[NDArray, NDArray], float]
        if sparseQCQP:
            self.structure_objective = self.structure_objective_sparse
        else:
            self.structure_objective = self.structure_objective_dense

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Photonics_TM_Polar_FDFD(omega={self.omega}, geometry={self.geometry}, chi={self.chi}, "
            f"des_mask={self.des_mask is not None}, ji={self.ji is not None}, "
            f"ei={self.ei is not None}, chi_background={self.chi_background is not None}, "
            f"sparseQCQP={self.sparseQCQP})"
        )

    def setup_EM_solver(self, geometry: Optional[PolarFDFDGeometry] = None) -> None:
        """
        Set up the FDFD electromagnetic solver with given geometry.

        Parameters
        ----------
        geometry : PolarFDFDGeometry, optional
            Geometry specification. If None, uses self.geometry.

        Notes
        -----
        Creates a TM_Polar_FDFD solver instance and stores it in self.EM_solver.
        """
        if geometry is not None:
            self.geometry = geometry

        assert self.geometry is not None
        check_attributes(
            self.geometry,
            "Nr",
            "Nphi",
            "Npml",
            "dr",
            "n_sectors",
            "bloch_phase",
        )
        geo = self.geometry
        self.EM_solver = TM_Polar_FDFD(
            self.omega,
            geo.Nphi,
            geo.Nr,
            geo.Npml,
            geo.dr,
            geo.n_sectors,
            geo.bloch_phase,
            geo.m,
            geo.lnR,
        )

    def setup_EM_operators(self) -> None:
        """
        Set up electromagnetic operators for the design region and background.

        Notes
        -----
        This method creates the appropriate operators based on whether sparse or dense
        QCQP formulation is used:
        - For sparse QCQP: Creates Ginv (inverse Green's function) and M operators
        - For dense QCQP: Creates G (Green's function) operator

        Requires self.des_mask to be defined.

        Raises
        ------
        AttributeError
            If des_mask is not defined.
        """
        check_attributes(self, "des_mask")
        assert self.EM_solver is not None
        assert self.des_mask is not None

        if self.sparseQCQP:
            self.Ginv, self.M = self.EM_solver.get_GaaInv(
                self.des_mask, self.chi_background
            )
        else:
            if self.chi_background is None:
                self.M = self.EM_solver.M0
                self.G = self.EM_solver.get_TM_G_od(self.des_mask, self.des_mask)
            else:
                self.M = self.EM_solver.M0 + self.EM_solver._get_diagM_from_chigrid(
                    self.chi_background
                )
                # For dense QCQP with background, compute G by solving
                Id = np.diag(self.des_mask.astype(complex).flatten(order="F"))[
                    :, self.des_mask.flatten(order="F")
                ]
                self.G = (
                    self.omega**2
                    * np.linalg.solve(self.M.toarray(), Id)[
                        self.des_mask.flatten(order="F"), :
                    ]
                )

    def setup_QCQP(self, Pdiags: str = "global", verbose: float = 0) -> None:
        """Set up the QCQP problem with polar area scaling."""
        from dolphindes.util import check_attributes

        check_attributes(self, "des_mask", "A0", "s0", "c0")
        assert self.des_mask is not None
        assert self.EM_solver is not None

        # Standard setup steps
        self.Ndes = int(np.sum(self.des_mask))
        des_mask_flat = self._get_des_mask_flat()

        if (self.ji is None) and (self.ei is None):
            raise AttributeError("an initial current ji or field ei must be specified.")
        if self.ji is not None and self.ei is not None:
            warnings.warn("If both ji and ei are specified then ji is ignored.")

        self.get_ei(self.ji, update=True)

        if Pdiags == "global":
            I = sp.eye_array(self.Ndes, dtype=complex, format="csc")
            self.Plist = [I, (-1j) * I]
        else:
            raise ValueError("Not a valid Pdiags specification / needs implementation")

        assert self.chi is not None
        assert self.ei is not None

        ei_des = self.ei[des_mask_flat]

        areas = self.EM_solver.get_pixel_areas()[des_mask_flat]
        sqrtW = np.sqrt(areas)
        invSqrtW = 1.0 / sqrtW

        if self.sparseQCQP:
            raise NotImplementedError("Sparse QCQP not yet implemented for polar FDFD.")
            # Transform objective
            A0_qcqp = W @ self.A0
            s0_qcqp = W @ self.s0
            SparseSharedProjQCQP()
            # if (self.Ginv is None) or (self.M is None):
            #     self.setup_EM_operators()

            # assert self.Ginv is not None

            # A2_qcqp = self.Ginv @ S_inv
            # A1_qcqp = S_inv @ sp.csc_array(
            #     np.conj(1.0 / self.chi) * (self.Ginv.conj().T) - sp.eye(self.Ndes)
            # )
            # s1_qcqp = S_inv @ (ei_des / 2)

            # self.QCQP = SparseSharedProjQCQP(
            #     A0_qcqp,
            #     s0_qcqp,
            #     self.c0,
            #     A1_qcqp,
            #     A2_qcqp,
            #     s1_qcqp,
            #     self.Plist,
            #     verbose=int(verbose),
            # )
        else:
            if self.G is None:
                self.setup_EM_operators()
            assert self.G is not None

            # Transform objective
            if sp.issparse(self.A0):
                self.A0 = self.A0.toarray()
            A0_qcqp = sqrtW[:, None] * self.A0 * invSqrtW[None, :]
            s0_qcqp = sqrtW * self.s0

            G_weighted = sqrtW[:, None] * self.G * invSqrtW[None, :]
            A1_dense = (
                np.conj(1.0 / self.chi) * np.eye(self.G.shape[0]) - G_weighted.conj().T
            )
            s1_qcqp = sqrtW * (ei_des / 2)
            A2_dense = sp.eye(self.Ndes, dtype=complex)

            self.QCQP = DenseSharedProjQCQP(
                A0_qcqp,
                s0_qcqp,
                self.c0,
                A1_dense,
                s1_qcqp,
                self.Plist,
                A2=A2_dense,
                verbose=int(verbose),
            )

    def get_chi_inf(self) -> ComplexArray:
        """Get the inferred susceptibility from the QCQP dual solution."""
        assert hasattr(self, "QCQP"), (
            "QCQP not initialized. Initialize and solve QCQP first."
        )
        assert self.QCQP is not None
        assert self.QCQP.current_xstar is not None, (
            "Inferred chi not available before solving QCQP dual"
        )
        assert self.des_mask is not None
        raise NotImplementedError("get_chi_inf not yet implemented for polar FDFD.")

    def _get_dof_chigrid_M_es(
        self, dof: NDArray[np.floating]
    ) -> Tuple[ComplexGrid, sp.csc_array, ComplexArray]:
        pass

    def structure_objective_sparse(
        self, dof: NDArray[np.floating], grad: NDArray[np.floating]
    ) -> float:
        pass

    def structure_objective_dense(
        self, dof: NDArray[np.floating], grad: NDArray[np.floating]
    ) -> float:
        pass


# Utility functions for photonics problems


def chi_to_feasible_rho(
    chi_inf: ComplexArray, chi_design: complex
) -> NDArray[np.floating]:
    """
    Project the inferred chi to the feasible set defined by chi_design.

    Resulting chi is chi_design * rho, where rho is in [0, 1].

    Parameters
    ----------
    chi_inf : ndarray of complex
        Inferred chi from Verlan optimization.
    chi_design : complex
        The design susceptibility of the problem.

    Returns
    -------
    rho : ndarray of float
        Projected density values in [0, 1].
    """
    rho = np.real(chi_inf.conj() * chi_design) / np.abs(chi_design) ** 2
    rho = np.clip(rho, 0, 1)
    return rho
