"""
Dual Problem Interface for Quadratically Constrained Quadratic Programming (QCQP).

This module provides interfaces for solving QCQP problems with shared projection 
constraints using dual optimization methods. It includes both sparse and dense 
implementations optimized for different matrix structures.
"""

__all__ = ["SparseSharedProjQCQP", "DenseSharedProjQCQP"]

import copy
from typing import (
    Any,
    Iterator,
    Optional,
    Tuple,
    cast,
)

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
         diagonal projector matrices P_j whose diagonals are the columns of Pdiags.
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
    Pdiags : ArrayLike
        2D array whose columns are the diagonals of projector matrices P_j.
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
    Pdiags : ComplexArray
        Complex-valued projector diagonals stacked column-wise.
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
            f"{self.Pdiags.shape[1]} projectors."
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "SparseSharedProjQCQP":
        """Copy this instance."""
        # custom __deepcopy__ because Acho is not pickle-able
        new_QCQP = SparseSharedProjQCQP.__new__(SparseSharedProjQCQP)
        for name, value in self.__dict__.items():
            if name != "Acho":
                setattr(new_QCQP, name, copy.deepcopy(value, memo))

        new_QCQP._initialize_Acho()  # Recompute the Cholesky factorization. 
        # If dense, will use self.current_lags.
        # TODO: update Acho with current_lags if applicable
        return new_QCQP

    def compute_precomputed_values(self) -> None:
        """Compute precomputed constraint data then initialize symbolic factorization."""
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
        # P = self._add_projectors(random_lags)
        A = self._get_total_A(random_lags)
        if self.verbose > 1:
            print(
                f"analyzing A of format and shape {type(A)}, {A.shape} "
                f"and # of nonzero elements '{A.count_nonzero()}"
            )
        self.Acho = sksparse.cholmod.analyze(A)

    def _update_Acho(self, A: sp.csc_array) -> None:
        """
        Update numerical Cholesky factorization for total matrix A.

        Parameters
        ----------
        A : sp.csc_array
            Current Hermitian (PSD / PD) system matrix.
        """
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
        return self.Acho.solve_A(b)

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
        A = self._get_total_A(lags)
        try:
            tmp = self.Acho.cholesky(A)
            tmp = (
                tmp.L()
            )  # Have to access the factor for the decomposition to be actually done.
            return True
        except sksparse.cholmod.CholmodNotPositiveDefiniteError:
            return False

    def refine_projectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Doubles the number of projectors, refining the number of constraints to smaller regions. Multipliers will be selected so that dual value remains constant and can be further optimized from existing point.

        For each projector P_j (columns of Pdiags) that doesn't project to a single pixel, split it into two projectors P1 and P2 such that P_j = P1 + P2 with half (or near half) the nonzero entries; j>2 (zero-th and first projector is always left as is). If there are only two projectors, keep both split projectors and original ones.

        Then, form a new Pdiags with the new projectors. Furthermore, extend lags. For each split projector P_j -> P_j, P_j+1, make lags[j] = lags[j], lags[j+1] = lags[j].

        Updates Pdiags and current_lags attributes. Verifies the dual value remains the same after refinement.

        Arguments
        ---------
        None

        Returns
        -------
        self.Pdiags, self.current_lags
        """
        raise NotImplementedError("Refine projectors not implemented in new refactor.")
        assert (
            self.current_lags is not None
        ), "Cannot refine projectors until an existing problem is solved. Run solve_current_dual_problem first."
        assert self.n_gen_constr == 0, "Cannot refine projectors when general constraints are present."
        assert np.all(
            self.Pdiags[:, 0] == 1
        ), "The zeroth projector must contain all ones (identity)"
        assert np.all(
            np.isclose(self.Pdiags[:, 1], -1j)
        ), "The second projector must contain all -1j values (-1j * identity)"

        new_Pdiags_cols = []
        new_lags_list = []

        # Handle the first projectors (j=0,1) - always kept as is
        new_Pdiags_cols.append(self.Pdiags[:, 0])
        new_lags_list.append(self.current_lags[0])

        new_Pdiags_cols.append(self.Pdiags[:, 1])
        new_lags_list.append(self.current_lags[1])

        # Iterate through the rest of the projectors (j > 0)
        split_limit = 0 if self.Pdiags.shape[1] == 2 else 2
        for j in range(split_limit, self.Pdiags.shape[1]):
            P_j_diag = self.Pdiags[:, j]
            current_lag_j = self.current_lags[j]

            # Find non-zero elements (pixels) in the current projector
            # Assuming projector diagonals are binary (0 or 1)
            nonzero_indices = np.where(P_j_diag != 0)[0]
            num_nonzero = len(nonzero_indices)

            if num_nonzero == 1:  # Projector is already a single pixel, keep it as is
                new_Pdiags_cols.append(P_j_diag)
                # new_lags_list.append(current_lag_j)
            elif num_nonzero > 1:  # Split the projector
                split_point = num_nonzero // 2
                indices1 = nonzero_indices[:split_point]
                indices2 = nonzero_indices[split_point:]

                # Create new projector P1
                P1_diag_new = np.zeros_like(P_j_diag)
                P1_diag_new[indices1] = P_j_diag[
                    indices1
                ]  # Copy values from original projector

                # Create new projector P2
                P2_diag_new = np.zeros_like(P_j_diag)
                P2_diag_new[indices2] = P_j_diag[
                    indices2
                ]  # Copy values from original projector

                if not np.all(P1_diag_new[1:] == 0):
                    new_Pdiags_cols.append(P1_diag_new)
                if not np.all(P2_diag_new[1:] == 0):
                    new_Pdiags_cols.append(P2_diag_new)

                if self.verbose > 1:
                    print(
                        f"Split projector {j} (lag: {current_lag_j:.2e}) with {num_nonzero} non-zeros into two projectors with {len(indices1)} and {len(indices2)} non-zeros."
                    )

            else:
                raise ValueError(
                    f"Unexpected number of non-zero elements in projector {j}: {num_nonzero}. Projector should have at least one non-zero element."
                )

        # Update Pdiags and current_lags
        # We need the overall P (and therefore the dual value) to stay constant upon projector refinement.
        # Here, we solve the system of equations to do that. It should always be solvable.
        new_Pdiags = np.column_stack(new_Pdiags_cols)
        new_Pdiags_real = np.vstack(
            [np.real(new_Pdiags), np.imag(new_Pdiags)]
        )  # Convert complex to real for least squares
        old_Pdiags_real = np.vstack(
            [np.real(self.Pdiags), np.imag(self.Pdiags)]
        )  # Convert complex to real for least squares
        self.current_lags, residuals, rank, s = np.linalg.lstsq(
            new_Pdiags_real, old_Pdiags_real @ self.current_lags, rcond=None
        )
        self.Pdiags = new_Pdiags

        self.compute_precomputed_values()  # Get new precomputed values for the comparison in the next run

        # Reset current dual, grad, hess, xstar as they are for the old problem structure
        new_dual, new_grad, new_hess, dual_aux = self.get_dual(
            self.current_lags, get_grad=True, get_hess=False
        )
        if self.verbose >= 1:
            print(
                f"previous dual: {self.current_dual}, new dual: {new_dual} (should be the same)"
            )
        assert np.isclose(
            new_dual, self.current_dual, rtol=1e-2
        ), "Dual value should be the same after refinement."

        self.current_dual = new_dual
        self.current_grad = new_grad
        self.current_hess = new_hess
        self.current_xstar = self.current_xstar

        return self.Pdiags, self.current_lags

    def iterative_splitting_step(
        self, method: str = "bfgs", max_cstrt_num: int = np.inf
    ) -> Iterator[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Iterative splitting step generator function that continues until pixel-level constraints are reached.

        Each call:
            1. Doubles the number of projectors using refine_projectors
            2. Solves the new dual problem and yields the results
            3. Continues until all projectors are "one-hot" matrices (pixel-level constraints)

        Parameters
        ----------
        method : str
            Optimizer used for solving each iterative splitting step
        max_cstrt_num : int or np.inf
            termination condition based on maximum number of constraints.
            If np.inf, defaults to pixel level constraints

        Yields
        ------
        tuple
            Result of solve_current_dual_problem:
            (current_dual, current_lags, current_grad, current_hess, current_xstar)
        """
        raise NotImplementedError("Iterative splitting not implemented in new refactor.")
        assert self.n_gen_constr == 0, "Cannot iteratively split projectors when general constraints are present."
        max_cstrt_num = min(max_cstrt_num, 2 * self.Pdiags.shape[0])
        # Check if we're already at termination condition
        if self.Pdiags.shape[1] >= max_cstrt_num:
            if self.verbose > 0:
                print("Projector number already above specified max or pixel level.")
            return

        # Continue splitting until number of constraints exceeds or equals max_cstrt_num
        while self.Pdiags.shape[1] < max_cstrt_num:
            if self.verbose > 0:
                if self.Pdiags.shape[1] == 2:
                    print(
                        f"Splitting projectors: {self.Pdiags.shape[1]} → {self.Pdiags.shape[1] + self.Pdiags.shape[1]}"
                    )
                else:
                    print(
                        f"Splitting projectors: {self.Pdiags.shape[1]} → {self.Pdiags.shape[1] + self.Pdiags.shape[1] - 2}"
                    )

            # Refine projectors to get finer constraints
            self.refine_projectors()

            # Solve the dual problem with the new projectors
            result = self.solve_current_dual_problem(
                method, init_lags=self.current_lags
            )

            # Yield the result to the caller
            yield result

            # Check if we've reached pixel level after this iteration
            if self.Pdiags.shape[1] >= max_cstrt_num:
                if self.verbose > 0:
                    print(
                        "Reached max number of projectors or pixel-level projectors. Refinement complete."
                    )
                break


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
    Pdiags : ArrayLike
        Columns are diagonals of projector matrices P_j.
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
        Pdiags: ArrayLike,
        A2: ArrayLike | None = None,
        B_j: list[ArrayLike] | None = None,
        s_2j: list[ArrayLike] | None = None,
        c_2j: ArrayLike | None = None,
        verbose: int = 0,
    ):

        if A2 is None:
            A2 = sp.eye_array(len(s0), format="csc")

        super().__init__(
            A0, s0, c0, A1, A2, s1, Pdiags,
            B_j=B_j, s_2j=s_2j, c_2j=c_2j, verbose=verbose
        )

    def __repr__(self) -> str:
        """Return a concise string summary (size and projector count)."""
        return (
            f"DenseSharedProjQCQP of size {self.A0.shape[0]}^2 with "
            f"{self.Pdiags.shape[1]} projectors."
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
        return la.cho_solve(self.Acho, b)

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
