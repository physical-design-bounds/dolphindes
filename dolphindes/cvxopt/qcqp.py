"""
Dual Problem Interface for Quadratically Constrained Quadratic Programming (QCQP).

This module provides interfaces for solving QCQP problems with shared projection
constraints using dual optimization methods. It includes both sparse and dense
implementations optimized for different matrix structures.
"""

__all__ = ["SparseSharedProjQCQP", "DenseSharedProjQCQP"]

import copy
from typing import Any, cast

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import sksparse.cholmod
from numpy.typing import ArrayLike

from dolphindes.types import ComplexArray, FloatNDArray

from ._base_qcqp import _SharedProjQCQP


class SparseSharedProjQCQP(_SharedProjQCQP):
    """Sparse QCQP with projector-structured and optional general quadratic constraints.

    This class specializes the generic _SharedProjQCQP implementation for the case
    where ALL quadratic matrices (A0, A1, A2, each projector block, and any B_j)
    are sparse. It supports:
      1. A family of "shared" (projection-structured) constraints defined through
         sparse projector matrices with shared sparsity structure.
      2. An optional list of additional (general) quadratic equality constraints
         parameterized by matrices B_j, vectors s_2j, and scalars c_2j.

    Primal problem (maximization form):
        maximize_x   - x^† A0 x + 2 Re(x^† s0) + c0
        subject to   Re( - x^† A1 P_j A2 x + 2 x^† A2^† P_j^† s1 ) = 0     (shared)
                     Re( - x^† A2^† B_j A2 x + 2 x^† A2^† s_2j + c_2j ) = 0 (general)

    Dual feasibility relies (heuristically) on at least one projector direction
    such that A0 + λ A1 P_j A2 becomes PSD for sufficiently large λ. This is not
    programmatically verified; users are responsible for supplying a suitable
    projector set (the second multiplier is chosen for this role).

    Parameters
    ----------
    A0 : sp.csc_array
        Objective quadratic matrix (Hermitian expected).
    s0 : ArrayLike
        Objective linear vector (complex allowed).
    c0 : float
        Objective constant term.
    A1 : sp.csc_array
        Left quadratic factor in projector constraints.
    A2 : sp.csc_array
        Right quadratic factor used in both projector and general constraints.
    s1 : ArrayLike
        Linear term coupled with projector constraints.
    Plist : list[ArrayLike]
        list of 2D arrays representing the projector matrices P_j.
    B_j : list[sp.csc_array] | None
        (Optional) list of general constraint middle matrices (between A2^† and A2).
    s_2j : list[ArrayLike] | None
        (Optional) list of linear term vectors for general constraints.
    c_2j : ArrayLike | None
        (Optional) array of constant terms for general constraints.
    verbose : int, default 0
        Verbosity level (≥1 prints preprocessing info).

    Attributes
    ----------
    A0, A1, A2 : scipy.sparse.csc_array
        Stored sparse matrices
    B_j : list[scipy.sparse.csc_array]
        General constraint matrices (empty if none supplied).
    s0, s1, s_2j : ComplexArray, ComplexArray, list[ComplexArray]
        Complex vectors for objective / constraints.
    c0 : float
        Objective constant.
    c_2j : FloatNDArray
        Real constants for general constraints (length matches B_j).
    Proj : :class:`dolphindes.util.Projectors`
        dolphindes Projectors object representing all projector matrices P_j.
    n_gen_constr : int
        Number of general constraints (len(B_j)).
    precomputed_As : list[sp.csc_array]
        Symmetrized matrices [Sym(A1 P_j A2)] for projectors followed by
        [Sym(A2^† B_j A2)] for general constraints (if any).
    Fs : ComplexArray
        Columns are A2^† P_j^† s1 (projector-only part used in derivatives).
    Acho : sksparse.cholmod.Factor | None
        Symbolic/numeric CHOLMOD factorization handle (updated per solve).
    current_dual : float | None
        Cached optimal dual value after solve_current_dual_problem().
    current_lags : FloatNDArray | None
        Cached Lagrange multipliers (projector first, then general).
    current_grad : FloatNDArray | None
        Gradient of dual at current_lags (if computed).
    current_hess : FloatNDArray | None
        Hessian of dual at current_lags (only when no general constraints).
    current_xstar : ComplexArray | None
        Primal maximizer associated with current_lags.
    use_precomp : bool
        Whether precomputation of constraint matrices/vectors is enabled.
    verbose : int
        Stored verbosity level.

    Performance Notes
    -----------------
    - CHOLMOD symbolic analysis is performed once (via _initialize_Acho) based on
      an example A(lags); subsequent factorizations reuse the sparsity pattern.
    - Precomputation accelerates repeated evaluations for moderate constraint counts.

    See Also
    --------
    DenseSharedProjQCQP : Dense analogue using LAPACK factorization.
    _SharedProjQCQP    : Base abstract class with core logic.
    """

    def __repr__(self) -> str:
        """Return a concise string summary (size and projector count)."""
        return (
            f"SparseSharedProjQCQP of size {self.A0.shape[0]}^2 with "
            f"{self.n_proj_constr} projectors."
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "SparseSharedProjQCQP":
        try:
            new_QCQP = cast("SparseSharedProjQCQP", super().__deepcopy__(memo))
        except TypeError:
            # autoreload (often used in notebooks) breaks inheritance detection
            new_QCQP = cast(
                "SparseSharedProjQCQP", _SharedProjQCQP.__deepcopy__(self, memo)
            )
        if getattr(new_QCQP, "_initialize_Acho", None) is not None:
            new_QCQP._initialize_Acho()
        return new_QCQP

    def compute_precomputed_values(self) -> None:
        """Precompute constraint data then initialize symbolic factorization."""
        super().compute_precomputed_values()
        self._initialize_Acho()

    def _initialize_Acho(self) -> sksparse.cholmod.Factor:
        """
        Symbolically analyze sparsity/fill pattern for Cholesky factorization.

        Returns
        -------
        sksparse.cholmod.Factor
            Analyzed (symbolic) factorization handle stored in self.Acho.
        """
        random_lags = np.random.rand(self.n_proj_constr + self.n_gen_constr)
        A = self._get_total_A(random_lags)
        assert sp.issparse(A)
        A = sp.csc_array(A)

        if self.verbose > 1:
            print(
                f"analyzing A of format and shape {type(A)}, {A.shape} "
                f"and # of nonzero elements '{A.nnz}"
            )
        self.Acho = sksparse.cholmod.analyze(A)
        return self.Acho

    def _update_Acho(self, A: sp.csc_array) -> None:
        """
        Update numerical Cholesky factorization for total matrix A.

        Parameters
        ----------
        A : sp.csc_array
            Current Hermitian (PSD / PD) system matrix.
        """
        assert self.Acho is not None
        self.Acho.cholesky_inplace(A)

    def _Acho_solve(self, b: ComplexArray) -> ComplexArray:
        """
        Solve A x = b using the current CHOLMOD factorization.

        Parameters
        ----------
        b : ComplexArray
            Right-hand side vector (or multiple RHS as columns).

        Returns
        -------
        ComplexArray
            Solution x = A^{-1} b.
        """
        assert self.Acho is not None
        return cast(ComplexArray, self.Acho.solve_A(b))

    def is_dual_feasible(self, lags: FloatNDArray) -> bool:
        """
        Check PSD feasibility of A(lags) via attempted Cholesky factorization.

        Parameters
        ----------
        lags : FloatNDArray
            Full Lagrange multiplier vector.

        Returns
        -------
        bool
            True if factorization succeeds (A is PSD), False otherwise.
        """
        assert self.Acho is not None
        A = self._get_total_A(lags)
        assert sp.issparse(A)
        A = sp.csc_array(A)
        try:
            tmp = self.Acho.cholesky(A)
            tmp = (
                tmp.L()
            )  # Have to access the factor for the decomposition to be actually done.
            return True
        except sksparse.cholmod.CholmodNotPositiveDefiniteError:
            return False


class DenseSharedProjQCQP(_SharedProjQCQP):
    """Dense QCQP with projector-structured constraints.

    Dense analogue of SparseSharedProjQCQP; uses scipy.linalg for
    Cholesky factorization. Inherits full problem specification from
    _SharedProjQCQP.

    Parameters
    ----------
    A0 : ArrayLike
        Objective quadratic term.
    s0 : ArrayLike
        Objective linear term.
    c0 : float
        Objective constant.
    A1 : ArrayLike
        Left quadratic factor in projector constraints.
    s1 : ArrayLike
        Projector constraint linear term.
    Plist : list[ArrayLike]
        list of 2D arrays representing the projector matrices P_j.
    A2 : ArrayLike | None, default None
        Right quadratic factor (defaults to identity if None).
    verbose : int, default 0
        Verbosity level.

    Notes
    -----
    - All quadratic matrices must be dense (or convertible) for this class.
    - General constraints can be supplied via the base constructor if extended.
    """

    def __init__(
        self,
        A0: ArrayLike,
        s0: ArrayLike,
        c0: float,
        A1: ArrayLike,
        s1: ArrayLike,
        Plist: list[ArrayLike],
        Pstruct: ArrayLike | None = None,
        A2: ArrayLike | None = None,
        B_j: list[ArrayLike] | None = None,
        s_2j: list[ArrayLike] | None = None,
        c_2j: ArrayLike | None = None,
        verbose: int = 0,
    ):
        if A2 is None:
            n = int(np.asarray(s0).shape[0])
            A2 = sp.eye_array(n, format="csc")
        super().__init__(
            A0,
            s0,
            c0,
            A1,
            A2,
            s1,
            Plist,
            Pstruct,
            B_j=B_j,
            s_2j=s_2j,
            c_2j=c_2j,
            verbose=verbose,
        )

    def __repr__(self) -> str:
        """Return a concise string summary (size and projector count)."""
        return (
            f"DenseSharedProjQCQP of size {self.A0.shape[0]}^2 with "
            f"{self.n_proj_constr} projectors."
        )

    def _update_Acho(self, A: ArrayLike) -> None:
        """
        Update dense Cholesky factorization of A.

        Parameters
        ----------
        A : ArrayLike
            Hermitian positive (semi)definite matrix to factor.
        """
        self.Acho = la.cho_factor(A)

    def _Acho_solve(self, b: ComplexArray) -> ComplexArray:
        """
        Solve A x = b using stored dense Cholesky factorization.

        Parameters
        ----------
        b : ComplexArray
            Right-hand side vector (or stacked RHS matrix).

        Returns
        -------
        ComplexArray
            Solution x = A^{-1} b.
        """
        return cast(ComplexArray, la.cho_solve(self.Acho, b))

    def is_dual_feasible(self, lags: FloatNDArray) -> bool:
        """
        Check PSD feasibility of A(lags) via dense Cholesky attempt.

        Parameters
        ----------
        lags : FloatNDArray
            Full Lagrange multiplier vector.

        Returns
        -------
        bool
            True if Cholesky succeeds, False otherwise.
        """
        A = self._get_total_A(lags)
        try:
            la.cho_factor(A)
            return True
        except la.LinAlgError:
            return False
