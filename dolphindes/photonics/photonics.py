"""
Concrete implementations of photonics QCQP design bounds solvers.

Provides TM and TE polarization FDFD solvers that bridge the QCQP Dual Problem
Interface in cvxopt and the Maxwell Solvers in maxwell.
"""

__all__ = [
    "Photonics_TM_FDFD",
    "Photonics_TE_Yee_FDFD",
    "chi_to_feasible_rho",
]

import warnings
from typing import Callable, Optional, Tuple, Union, cast

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from dolphindes.geometry import (
    CartesianFDFDGeometry,
    GeometryHyperparameters,
    PolarFDFDGeometry,
)
from dolphindes.maxwell import TM_FDFD, TM_Polar_FDFD
from dolphindes.types import (
    BoolGrid,
    ComplexArray,
    ComplexGrid,
    FloatNDArray,
)
from dolphindes.util import check_attributes

from ._base_photonics import Photonics_FDFD


class Photonics_TM_FDFD(Photonics_FDFD):
    """
    TM polarization FDFD photonics problem (Cartesian or Polar).

    Attributes
    ----------
    EM_solver : :class:`dolphindes.maxwell.TM_FDFD` | None
        Electromagnetic field solver.
    QCQP : :class:`dolphindes.cvxopt.qcqp.SparseSharedProjQCQP` | :class:`dolphindes.cvxopt.qcqp.DenseSharedProjQCQP` | None
        QCQP instance for optimization.
    Ginv : csc_array or None
        Inverse Green's function (sparse QCQP).
    G : ndarray of complex or None
        Green's function (dense QCQP).
    M : csc_array or None
        Maxwell operator.
    EM_solver : TM_FDFD or TM_Polar_FDFD or None
        Electromagnetic field solver.
    structure_objective : Callable
        Function for structure optimization objective.
    """

    def __init__(
        self,
        omega: complex,
        geometry: CartesianFDFDGeometry | PolarFDFDGeometry,
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
        geometry : CartesianFDFDGeometry or PolarFDFDGeometry
            Geometry specification.
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
        self.EM_solver: Optional[Union[TM_FDFD, TM_Polar_FDFD]] = None
        self.Ndes: Optional[int] = None
        self.Plist: Optional[list] = None
        self.dense_s0: Optional[ComplexArray] = None

        if isinstance(geometry, CartesianFDFDGeometry):
            self._flatten_order = "C"
        elif isinstance(geometry, PolarFDFDGeometry):
            self._flatten_order = "F"

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
            self.setup_EM_solver(geometry)
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

    def setup_EM_solver(
        self, geometry: Optional[GeometryHyperparameters] = None
    ) -> None:
        """
        Set up the FDFD electromagnetic solver with given geometry.

        Parameters
        ----------
        geometry : CartesianFDFDGeometry or PolarFDFDGeometry
            Geometry specification. If None, uses self.geometry.

        Notes
        -----
        Creates a TM_FDFD or TM_Polar_FDFD solver instance and stores it in
        self.EM_solver.
        """
        if geometry is not None:
            self.geometry = geometry

        assert self.geometry is not None
        if isinstance(self.geometry, CartesianFDFDGeometry):
            self._flatten_order = "C"
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
            self.EM_solver = TM_FDFD(self.omega, self.geometry)
        elif isinstance(self.geometry, PolarFDFDGeometry):
            self._flatten_order = "F"
            check_attributes(
                self.geometry,
                "Nr",
                "Nphi",
                "Npml",
                "dr",
                "n_sectors",
                "bloch_phase",
            )
            self.EM_solver = TM_Polar_FDFD(self.omega, self.geometry)
        else:
            raise TypeError(f"Unsupported geometry type: {type(self.geometry)}")

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
                Id = np.diag(
                    self.des_mask.astype(complex).flatten(order=self._flatten_order)
                )[:, self.des_mask.flatten(order=self._flatten_order)]
                self.G = (
                    self.omega**2
                    * np.linalg.solve(self.M.toarray(), Id)[
                        self.des_mask.flatten(order=self._flatten_order), :
                    ]
                )

    def get_chi_inf(self) -> ComplexArray:
        """Get the inferred susceptibility from the QCQP dual solution."""
        return super().get_chi_inf()

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
        if isinstance(self.geometry, PolarFDFDGeometry):
            raise NotImplementedError("Not implemented for Polar geometry yet.")

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
        if isinstance(self.geometry, PolarFDFDGeometry):
            raise NotImplementedError("Not implemented for Polar geometry yet.")

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
        if isinstance(self.geometry, PolarFDFDGeometry):
            raise NotImplementedError("Not implemented for Polar geometry yet.")

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
        pass


# Utility functions for photonics problems


def chi_to_feasible_rho(chi_inf: ComplexArray, chi_design: complex) -> FloatNDArray:
    """
    Project the inferred chi to the feasible set defined by chi_design.

    Resulting chi is chi_design * rho, where rho is in [0, 1].

    Parameters
    ----------
    chi_inf : ComplexArray
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
    return cast(FloatNDArray, rho)
