"""
Concrete implementations of photonics QCQP design bounds solvers.

Provides TM and TE polarization FDFD solvers that bridge the QCQP Dual Problem
Interface in cvxopt and the Maxwell Solvers in maxwell.
"""

__all__ = ["Photonics_TM_FDFD", "Photonics_TE_Yee_FDFD", "chi_to_feasible_rho"]

import warnings
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from dolphindes.cvxopt import DenseSharedProjQCQP, SparseSharedProjQCQP
from dolphindes.maxwell import TM_FDFD
from dolphindes.types import BoolGrid, ComplexArray, ComplexGrid, FloatNDArray
from dolphindes.util import check_attributes

from ._base_photonics import CartesianFDFDGeometry, Photonics_FDFD


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
        except AttributeError as e:
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

    def set_objective(
        self,
        A0: Optional[Union[ComplexArray, sp.csc_array]] = None,
        s0: Optional[ComplexArray] = None,
        c0: Optional[float] = None,
        denseToSparse: bool = False,
    ) -> None:
        """
        Set the QCQP objective function parameters.

        Not specifying a particular parameter leaves it unchanged.

        Parameters
        ----------
        A0 : ndarray of complex or scipy.sparse.csc_array, optional
            The matrix A0 in the QCQP objective function.
        s0 : ndarray of complex, optional
            The vector s0 in the QCQP objective function.
        c0 : float, optional
            The constant c0 in the QCQP objective function. Default: 0.0
        denseToSparse : bool, optional
            If True, treat input A0 and s0 as describing forms
            of the polarization p, and convert them to the equivalent forms
            of (Gp) before assigning to the class attributes. Default: False
        """
        if denseToSparse:
            if not self.sparseQCQP:
                raise ValueError(
                    "sparseQCQP needs to be True to use dense-to-sparse conversion."
                )
            if A0 is not None:
                assert self.Ginv is not None
                self.A0 = self.Ginv.T.conj() @ sp.csc_array(A0) @ self.Ginv
            if s0 is not None:
                assert self.Ginv is not None
                self.dense_s0 = s0
                self.s0 = self.Ginv.T.conj() @ s0
        else:
            if A0 is not None:
                self.A0 = A0
            if s0 is not None:
                self.s0 = s0
                self.dense_s0 = None

        if c0 is not None:
            self.c0 = c0

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

    def get_ei(
        self, ji: Optional[ComplexGrid] = None, update: bool = False
    ) -> ComplexGrid:
        """
        Get or compute the incident electromagnetic field.

        Parameters
        ----------
        ji : ndarray of complex, optional
            Current source for computing incident field. If None, uses self.ji.
        update : bool, optional
            Whether to update self.ei with the computed field. Default: False

        Returns
        -------
        ei : ndarray of complex
            The incident electromagnetic field. If self.ei exists, returns it directly.
            Otherwise computes it using the EM solver.
        """
        assert self.EM_solver is not None
        if self.ei is None:
            ei = (
                self.EM_solver.get_TM_field(ji, self.chi_background)
                if self.ji is None
                else self.EM_solver.get_TM_field(self.ji, self.chi_background)
            )
        else:
            ei = self.ei
        if update:
            self.ei = ei
        return ei

    def set_ei(self, ei: ComplexGrid) -> None:
        """
        Set the incident electromagnetic field.

        Parameters
        ----------
        ei : ndarray of complex
            The incident electromagnetic field to store.
        """
        self.ei = ei

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
        check_attributes(self, "des_mask", "A0", "s0", "c0")

        # number of field degrees of freedom / pixels in design region
        assert self.des_mask is not None
        self.Ndes = int(np.sum(self.des_mask))

        # generate initial field
        if (self.ji is None) and (self.ei is None):
            raise AttributeError("an initial current ji or field ei must be specified.")
        if not (self.ji is None) and not (self.ei is None):
            warnings.warn("If both ji and ei are specified then ji is ignored.")

        self.get_ei(self.ji, update=True)

        if Pdiags == "global":
            # Build projectors as a list of matrices (new API)
            # old behavior: two columns with [1, -1j] on the diagonal
            I = sp.eye_array(self.Ndes, dtype=complex, format="csc")
            self.Plist = [I, (-1j) * I]
        else:
            raise ValueError("Not a valid Pdiags specification / needs implementation")

        assert self.chi is not None
        assert self.ei is not None
        if (
            self.sparseQCQP
        ):  # rewrite later when sparse and dense QCQP classes are unified
            if (self.Ginv is None) or (self.M is None):
                self.setup_EM_operators()

            assert self.Ginv is not None
            A1_sparse = sp.csc_array(
                np.conj(1.0 / self.chi) * self.Ginv.conj().T - sp.eye(self.Ndes)
            )
            A2_sparse = sp.csc_array(self.Ginv)

            self.QCQP = SparseSharedProjQCQP(
                self.A0,
                self.s0,
                self.c0,
                A1_sparse,
                A2_sparse,
                self.ei[self.des_mask] / 2,
                self.Plist,
                verbose=int(verbose),
            )
        else:
            if self.G is None:
                self.setup_EM_operators()

            assert self.G is not None
            A1_dense = (
                np.conj(1.0 / self.chi) * np.eye(self.G.shape[0]) - self.G.conj().T
            )
            print(
                self.A0.shape,
                self.s0.shape,
                self.c0,
                A1_dense.shape,
                self.ei[self.des_mask].shape,
            )
            self.QCQP = DenseSharedProjQCQP(
                self.A0,
                self.s0,
                self.c0,
                A1_dense,
                self.ei[self.des_mask] / 2,
                self.Plist,
                verbose=int(verbose),
            )  # for dense QCQP formulation A2 is not needed

    def bound_QCQP(
        self,
        method: str = "bfgs",
        init_lags: Optional[FloatNDArray] = None,
        opt_params: Optional[dict[str, Any]] = None,
    ) -> Tuple[float, FloatNDArray, FloatNDArray, Optional[FloatNDArray], ComplexArray]:
        """
        Calculate a bound on the QCQP dual problem.

        Parameters
        ----------
        method : str, optional
            Optimization method to use. Options: 'bfgs', 'newton'. Default: 'bfgs'
        init_lags : ndarray of float, optional
            Initial Lagrange multipliers for optimization. If None, finds feasible point.
        opt_params : dict, optional
            Additional parameters for the optimization algorithm.

        Returns
        -------
        result : tuple
            Result from QCQP dual problem solver containing:
            (dual_value, lagrange_multipliers, gradient, hessian, primal_variable)
        """
        assert self.QCQP is not None
        return self.QCQP.solve_current_dual_problem(
            method=method, init_lags=init_lags, opt_params=opt_params
        )

    def get_chi_inf(self) -> ComplexArray:
        """
        Get the inferred susceptibility from the QCQP dual solution.

        Returns
        -------
        chi_inf : ndarray of complex
            The inferred susceptibility χ_inf = P / E_total, where P is the
            polarization current and E_total is the total electric field.

        Notes
        -----
        This represents the material susceptibility that would be required to
        achieve the optimal field distribution found by the QCQP solver.
        The inferred chi may not be physically feasible for nonzero duality gap.

        Raises
        ------
        AssertionError
            If QCQP has not been initialized or solved yet.
        """
        assert hasattr(self, "QCQP"), (
            "QCQP not initialized. Initialize and solve QCQP first."
        )
        assert self.QCQP is not None
        assert self.QCQP.current_xstar is not None, (
            "Inferred chi not available before solving QCQP dual"
        )
        assert self.des_mask is not None

        P: ComplexArray
        Es: ComplexArray
        if self.sparseQCQP:
            assert self.QCQP.A2 is not None
            P = self.QCQP.A2 @ self.QCQP.current_xstar  # Calculate polarization current
            Es = self.QCQP.current_xstar
        else:
            assert self.G is not None
            P = self.QCQP.current_xstar
            Es = self.G @ P

        Etotal = self.get_ei()[self.des_mask] + Es
        return P / Etotal

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
