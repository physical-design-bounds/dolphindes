"""
Base classes and hyperparameters for photonics problems.

This module contains abstract base classes and geometry specifications that
are inherited by concrete photonics solver implementations.
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import scipy.sparse as sp

from dolphindes.cvxopt import DenseSharedProjQCQP, SparseSharedProjQCQP
from dolphindes.types import BoolGrid, ComplexArray, ComplexGrid, FloatNDArray


@dataclass
class GeometryHyperparameters(ABC):
    """Base class for geometry specifications."""

    @abstractmethod
    def get_grid_size(self) -> Tuple[int, int]:
        """
        Return (Nx, Ny) or equivalent grid dimensions.

        Returns
        -------
        tuple of int
            Grid dimensions (Nx, Ny).
        """
        pass


@dataclass
class CartesianFDFDGeometry(GeometryHyperparameters):
    """
    Cartesian FDFD geometry specification.

    Attributes
    ----------
    Nx : int
        Number of pixels along the x direction.
    Ny : int
        Number of pixels along the y direction.
    Npmlx : int
        Size of the x direction PML in pixels.
    Npmly : int
        Size of the y direction PML in pixels.
    dx : float
        Finite difference grid pixel size in x direction, in units of 1.
    dy : float
        Finite difference grid pixel size in y direction, in units of 1.
    bloch_x : float
        x-direction phase shift associated with the periodic boundary conditions.
        Default: 0.0
    bloch_y : float
        y-direction phase shift associated with the periodic boundary conditions.
        Default: 0.0
    """

    Nx: int
    Ny: int
    Npmlx: int
    Npmly: int
    dx: float
    dy: float
    bloch_x: float = 0.0
    bloch_y: float = 0.0

    def get_grid_size(self) -> Tuple[int, int]:
        """
        Return grid dimensions.

        Returns
        -------
        tuple of int
            Grid dimensions (Nx, Ny).
        """
        return (self.Nx, self.Ny)


class Photonics_FDFD(ABC):
    """
    Mother class for frequency domain problems with finite difference discretization.

    To allow for lazy initialization, only demands omega upon init.

    Specification of the photonics design objective:
    if sparseQCQP is False, the objective is specified as a quadratic function of the
    polarization p: max_p -p^dagger A0 p + 2 Re (p^dagger s0) + c0

    if sparseQCQP is True, the objective is specified as a quadratic function of (Gp):
    max_{Gp} -(Gp)^dagger A0 (Gp) + 2 Re((Gp)^dagger s0) + c0

    Attributes
    ----------
    omega : complex
        Circular frequency, can be complex to allow for finite bandwidth effects.
    geometry : GeometryHyperparameters or None
        Geometry specification (Cartesian, Polar, etc.)
    chi : complex or None
        Bulk susceptibility of material used.
    des_mask : ndarray of bool or None
        Boolean mask over computation domain that is TRUE for pixels in design region.
    ji : ndarray of complex or None
        Incident current source that produces an incident field.
    ei : ndarray of complex or None
        Incident field.
    chi_background : ndarray of complex or None
        The background structure.
        The default is None, in which case it is set to vacuum.
    sparseQCQP : bool or None
        Boolean flag indicating whether the sparse QCQP convention is used.
    A0 : ndarray of complex or scipy.sparse.csc_array or None
        A0 array in the QCQP field design objective.
    s0 : ndarray of complex or None
        The vector s0 in the QCQP field design objective.
    c0 : float
        The constant c0 in the QCQP field design objective.
    QCQP : SparseSharedProjQCQP or DenseSharedProjQCQP or None
        The QCQP instance for optimization.
    """

    def __init__(
        self,
        omega: complex,
        geometry: Optional[GeometryHyperparameters] = None,
        chi: Optional[complex] = None,
        des_mask: Optional[BoolGrid] = None,
        ji: Optional[ComplexGrid] = None,
        ei: Optional[ComplexGrid] = None,
        chi_background: Optional[ComplexGrid] = None,
        sparseQCQP: Optional[bool] = None,
        A0: Optional[Union[ComplexArray, sp.csc_array]] = None,
        s0: Optional[ComplexArray] = None,
        c0: float = 0.0,
        Pdiags: Optional[str] = None,
    ) -> None:
        """
        Initialize Photonics_FDFD.

        Only omega is absolutely needed for initialization, other attributes can be
        added later.

        Parameters
        ----------
        omega : complex
            Circular frequency.
        geometry : GeometryHyperparameters, optional
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
            Flag for sparse QCQP formulation.
        A0 : ndarray or csc_array, optional
            Objective quadratic matrix.
        s0 : ndarray of complex, optional
            Objective linear vector.
        c0 : float, optional
            Objective constant. Default: 0.0
        Pdiags : str, optional
            Projector specification.
        """
        self.omega = omega
        self.geometry = geometry
        self.chi = chi

        self.des_mask = des_mask
        self.ji = ji
        self.ei = ei
        self.chi_background = chi_background

        self.sparseQCQP = sparseQCQP
        self.A0 = A0
        self.s0 = s0
        self.c0 = c0
        self.Pdiags = Pdiags
        self.QCQP: Optional[Union[SparseSharedProjQCQP, DenseSharedProjQCQP]] = None

    def __deepcopy__(self, memo: dict[Any, Any]) -> "Photonics_FDFD":
        """
        Deep copy this instance.

        Parameters
        ----------
        memo : dict
            Memoization dictionary for deepcopy.

        Returns
        -------
        Photonics_FDFD
            Deep copied instance.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            try:
                setattr(new, k, copy.deepcopy(v, memo))
            except Exception:
                setattr(new, k, v)  # fallback to reference
        return new

    @abstractmethod
    def set_objective(
        self,
        A0: Optional[Union[ComplexArray, sp.csc_array]] = None,
        s0: Optional[ComplexArray] = None,
        c0: Optional[float] = None,
        denseToSparse: bool = False,
    ) -> None:
        """
        Set QCQP objective parameters.

        Parameters
        ----------
        A0 : ndarray or csc_array, optional
            Objective quadratic matrix.
        s0 : ndarray of complex, optional
            Objective linear vector.
        c0 : float, optional
            Objective constant.
        denseToSparse : bool, optional
            Convert dense to sparse representation. Default: False
        """
        raise NotImplementedError

    @abstractmethod
    def setup_EM_solver(
        self, geometry: Optional[GeometryHyperparameters] = None
    ) -> None:
        """
        Initialize EM solver.

        Parameters
        ----------
        geometry : GeometryHyperparameters, optional
            Geometry specification.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_EM_operators(self) -> None:
        """Build EM operators (G/Ginv, M, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def get_ei(
        self, ji: Optional[ComplexGrid] = None, update: bool = False
    ) -> ComplexGrid:
        """
        Return the incident field.

        Parameters
        ----------
        ji : ndarray of complex, optional
            Current source.
        update : bool, optional
            Whether to update stored incident field. Default: False

        Returns
        -------
        ndarray of complex
            Incident electromagnetic field.
        """
        raise NotImplementedError

    @abstractmethod
    def set_ei(self, ei: ComplexGrid) -> None:
        """
        Set the incident field.

        Parameters
        ----------
        ei : ndarray of complex
            Incident electromagnetic field to store.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_QCQP(self, Pdiags: str = "global", verbose: float = 0) -> None:
        """
        Construct the QCQP instance for the current configuration.

        Parameters
        ----------
        Pdiags : str, optional
            Projector specification. Default: "global"
        verbose : float, optional
            Verbosity level. Default: 0
        """
        raise NotImplementedError

    @abstractmethod
    def bound_QCQP(
        self,
        method: str = "bfgs",
        init_lags: Optional[FloatNDArray] = None,
        opt_params: Optional[dict[str, Any]] = None,
    ) -> Tuple[float, FloatNDArray, FloatNDArray, Optional[FloatNDArray], ComplexArray]:
        """
        Solve the QCQP dual and return solver results.

        Parameters
        ----------
        method : str, optional
            Optimization method. Default: 'bfgs'
        init_lags : ndarray of float, optional
            Initial Lagrange multipliers.
        opt_params : dict, optional
            Optimization parameters.

        Returns
        -------
        tuple
            (dual_value, lagrange_multipliers, gradient, hessian, primal_variable)
        """
        raise NotImplementedError

    @abstractmethod
    def get_chi_inf(self) -> ComplexArray:
        """
        Compute inferred chi from the QCQP dual solution.

        Returns
        -------
        ndarray of complex
            Inferred susceptibility.
        """
        raise NotImplementedError

    def solve_current_dual_problem(
        self,
        method: str = "bfgs",
        init_lags: Optional[FloatNDArray] = None,
        opt_params: Optional[dict[str, Any]] = None,
    ) -> Tuple[float, FloatNDArray, FloatNDArray, Optional[FloatNDArray], ComplexArray]:
        """
        Delegate to bound_QCQP so callers can use a uniform method name.

        Subclasses only need to implement bound_QCQP.

        Parameters
        ----------
        method : str, optional
            Optimization method. Default: 'bfgs'
        init_lags : ndarray of float, optional
            Initial Lagrange multipliers.
        opt_params : dict, optional
            Optimization parameters.

        Returns
        -------
        tuple
            (dual_value, lagrange_multipliers, gradient, hessian, primal_variable)
        """
        return self.bound_QCQP(
            method=method, init_lags=init_lags, opt_params=opt_params
        )
