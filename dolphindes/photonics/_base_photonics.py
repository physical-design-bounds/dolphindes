"""
Base classes and hyperparameters for photonics problems.

This module contains abstract base classes and geometry specifications that
are inherited by concrete photonics solver implementations.
"""

__all__ = [
    "Photonics_FDFD",
]

import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from dolphindes.cvxopt import DenseSharedProjQCQP, SparseSharedProjQCQP
from dolphindes.geometry import GeometryHyperparameters
from dolphindes.types import (
    BoolGrid,
    ComplexArray,
    ComplexGrid,
    FloatNDArray,
)


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
        geometry: GeometryHyperparameters,
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
        flatten_order: Literal["C", "F"] = "C",
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
        self._flatten_order: Literal["C", "F"] = flatten_order

        self.Ginv: Optional[sp.csc_array] = None
        self.G: Optional[ComplexArray] = None
        self.M: Optional[sp.csc_array] = None
        self.EM_solver: Optional[Any] = None
        self.Ndes: Optional[int] = None
        self.Plist: Optional[list] = None
        self.dense_s0: Optional[ComplexArray] = None

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

    def _get_des_mask_flat(self) -> BoolGrid:
        """Get flattened design mask in appropriate order for this geometry."""
        assert self.des_mask is not None
        return self.des_mask.flatten(order=self._flatten_order)

    def set_objective(
        self,
        A0: Optional[Union[ComplexArray, sp.csc_array]] = None,
        s0: Optional[ComplexArray] = None,
        c0: Optional[float] = None,
        denseToSparse: bool = False,
    ) -> None:
        """
        Set QCQP objective parameters with appropriate basis transformations.

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
        des_mask_flat = self._get_des_mask_flat()
        areas = self.geometry.get_pixel_areas()[des_mask_flat]

        if self.sparseQCQP:
            if denseToSparse:
                assert self.Ginv is not None
                W_mat = sp.diags(areas)  # Full integration measure
                if A0 is not None:
                    term = W_mat @ sp.csc_array(A0)
                    self.A0 = self.Ginv.conj().T @ term @ self.Ginv
                if s0 is not None:
                    self.dense_s0 = s0
                    self.s0 = self.Ginv.conj().T @ (W_mat @ s0)
            else:
                # Direct, assumes the user knows what they are doing
                if A0 is not None:
                    self.A0 = A0
                if s0 is not None:
                    self.s0 = s0
                    self.dense_s0 = None
        else:
            if denseToSparse:
                raise ValueError("Cannot use denseToSparse=True when sparseQCQP=False.")

            # Transform physical inputs to weighted current basis (x = sqrt(W) J)
            sqrtW = np.sqrt(areas)
            invSqrtW = 1.0 / sqrtW

            if A0 is not None:
                if sp.issparse(A0):
                    A0_dense = A0.toarray()
                else:
                    A0_dense = A0
                self.A0 = sqrtW[:, None] * A0_dense * invSqrtW[None, :]

            if s0 is not None:
                # Vector Scaling: sqrtW * s0
                # Matches J' W s0 -> x' sqrtW s0
                self.dense_s0 = s0
                self.s0 = sqrtW * s0

        # Constant c0 is coordinate-independent
        if c0 is not None:
            self.c0 = c0

    def setup_QCQP(self, Pdiags: str = "global", verbose: float = 0) -> None:
        """
        Set up the quadratically constrained quadratic programming (QCQP) problem.

        Parameters
        ----------
        Pdiags : str or ndarray, optional
            Specification for projection matrix diagonals. If "global", creates
            global projectors with ones and -1j entries. Default: "global"
        verbose : float, optional
            Verbosity level for debugging output. Default: 0

        Notes
        -----
        For sparse QCQP, creates SparseSharedProjQCQP with transformed matrices.
        For dense QCQP, creates DenseSharedProjQCQP with original matrices.

        Raises
        ------
        AttributeError
            If required attributes (des_mask, A0, s0, c0) are not defined.
        ValueError
            If neither ji nor ei is specified, or if Pdiags specification is invalid.
        """
        from dolphindes.util import check_attributes

        check_attributes(self, "des_mask", "A0", "s0", "c0")
        assert self.des_mask is not None
        assert self.EM_solver is not None

        self.Ndes = int(np.sum(self.des_mask))
        des_mask_flat = self._get_des_mask_flat()

        if (self.ji is None) and (self.ei is None):
            raise AttributeError("an initial current ji or field ei must be specified.")
        if self.ji is not None and self.ei is not None:
            warnings.warn("If both ji and ei are specified then ji is ignored.")

        self.get_ei(self.ji, update=True)

        if Pdiags == "global":
            Id = sp.eye_array(self.Ndes, dtype=complex, format="csc")
            self.Plist = [Id, (-1j) * Id]
        else:
            raise ValueError("Not a valid Pdiags specification / needs implementation")

        assert self.chi is not None
        assert self.ei is not None

        ei_des = self.ei.flatten(order=self._flatten_order)[des_mask_flat]

        areas = self.geometry.get_pixel_areas()[des_mask_flat]
        sqrtW = np.sqrt(areas)
        invSqrtW = 1.0 / sqrtW

        if self.sparseQCQP:
            if (self.Ginv is None) or (self.M is None):
                self.setup_EM_operators()

            assert self.Ginv is not None
            W_mat = sp.diags(areas, format="csc")

            A2_sparse = self.Ginv
            term1 = self.Ginv.conj().T @ (
                W_mat @ (sp.eye(self.Ginv.shape[0]) * np.conj(1.0 / self.chi))
            )
            term2 = W_mat

            A1_sparse = term1 - term2
            s1_sparse = W_mat @ (ei_des / 2)

            self.QCQP = SparseSharedProjQCQP(
                self.A0,
                self.s0,
                self.c0,
                A1_sparse,
                A2_sparse,
                s1_sparse,
                self.Plist,
                verbose=int(verbose),
            )
        else:
            if self.G is None:
                self.setup_EM_operators()

            assert self.G is not None

            if sp.issparse(self.A0):
                self.A0 = self.A0.toarray()

            G_weighted = sqrtW[:, None] * self.G * invSqrtW[None, :]
            A1_dense = (
                np.conj(1.0 / self.chi) * np.eye(self.G.shape[0]) - G_weighted.conj().T
            )
            s1_qcqp = sqrtW * (ei_des / 2)
            A2_dense = sp.eye(self.Ndes, dtype=complex)

            self.QCQP = DenseSharedProjQCQP(
                self.A0,
                self.s0,
                self.c0,
                A1_dense,
                s1_qcqp,
                self.Plist,
                A2=A2_dense,
                verbose=int(verbose),
            )

    def get_ei(
        self, ji: Optional[ComplexGrid] = None, update: bool = False
    ) -> ComplexArray:
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
        assert self.EM_solver is not None
        ei: ComplexArray
        if self.ei is None:
            assert ji is not None or self.ji is not None, (
                "Either ji argument or self.ji must be specified to compute ei."
            )
            source = ji if ji is not None else self.ji
            ei = self.EM_solver.get_TM_field(source, self.chi_background)
        else:
            ei = self.ei
        if update:
            self.ei = ei
        return ei

    def set_ei(self, ei: ComplexArray) -> None:
        """Set the incident electromagnetic field."""
        self.ei = ei

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
        assert self.QCQP is not None
        return self.QCQP.solve_current_dual_problem(
            method=method, init_lags=init_lags, opt_params=opt_params
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

        des_mask_flat = self._get_des_mask_flat()

        P: ComplexArray
        Es: ComplexArray
        if self.sparseQCQP:
            assert self.QCQP.A2 is not None
            P = self.QCQP.A2 @ self.QCQP.current_xstar
            Es = self.QCQP.current_xstar
        else:
            assert self.G is not None
            P = self.QCQP.current_xstar
            Es = self.G @ P

        Etotal = self.get_ei().flatten(order=self._flatten_order)[des_mask_flat] + Es
        return P / Etotal

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

    @abstractmethod
    def setup_EM_solver(self, geometry: GeometryHyperparameters) -> None:
        """Initialize EM solver (geometry-specific)."""
        raise NotImplementedError

    @abstractmethod
    def setup_EM_operators(self) -> None:
        """Build EM operators G/Ginv, M (geometry-specific)."""
        raise NotImplementedError

    @abstractmethod
    def _get_dof_chigrid_M_es(
        self, dof: NDArray[np.floating]
    ) -> Tuple[ComplexGrid, sp.csc_array, ComplexArray]:
        """Compute chi grid and scattered field from DOFs (geometry-specific)."""
        raise NotImplementedError
