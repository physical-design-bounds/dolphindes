import copy
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Tuple, cast

import numpy as np
import scipy.linalg as la  # Added for dense eigenvalue checks
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import ArrayLike, NDArray

from dolphindes.types import ComplexArray, FloatNDArray, SparseDense
from dolphindes.util import Projectors, Sym

from .optimization import BFGS, Alt_Newton_GD, OptimizationHyperparameters, _Optimizer

if TYPE_CHECKING:
    from .gcd import GCDHyperparameters


class _SharedProjQCQP(ABC):
    """Represents a quadratically constrained quadratic program (QCQP).

    QCQP has a set of projection-structured (shared) constraints and (optionally) a
    separate list of general quadratic constraints.

    Primal problem (maximization form):
        maximize_x   - x^† A0 x + 2 Re(x^† s0) + c0
        subject to   Re( - x^† A1 P_j A2 x + 2 x^† A2^† P_j^† s1 ) = 0
                     Re( - x^† A2^† B_j A2 x + 2 x^† A2^† s_2j + c_2j ) = 0

    where the P_j are projection matrices with shared sparsity structure.
    The matrices used internally for the shared (projector) constraints are
    symmetrized via Sym(A1 P_j A2) to ensure Hermitian structure.

    Dual feasibility relies on the existence of (at least) one projector column
    such that A0 + λ A1 P_j A2 becomes positive semidefinite for sufficiently
    large λ > 0 (by convention this is often column index 1, but the code does
    not enforce or check which column satisfies this).

    Attributes
    ----------
    A0 : sp.csc_array | ArrayLike
        Quadratic matrix in the objective.
    s0 : ArrayLike
        Linear term vector in the objective.
    c0 : float
        Constant term in the objective.
    A1 : sp.csc_array | ArrayLike
        Left quadratic factor in projector constraints.
    A2 : sp.csc_array | ArrayLike
        Right quadratic factor in projector and general constraints.
    s1 : ArrayLike
        Linear term vector for projector constraints.
    B_j : list[sp.csc_array | ArrayLike]
        List of general constraint middle matrices (between A2^† and A2).
    s_2j : list[ArrayLike]
        Linear term vectors for general constraints.
    c_2j : ArrayLike
        Constant terms for general constraints.
    Proj : :class:`dolphindes.util.Projectors`
        dolphindes Projectors object representing all projector matrices P_j.
    verbose : int
        Verbosity level (0 = silent).
    Acho : Any | None
        Cholesky (or other) factorization object of the current total A matrix.
    current_dual : float | None
        Cached dual value after solve_current_dual_problem().
    current_lags : FloatNDArray | None
        Cached optimal Lagrange multipliers.
    current_grad : FloatNDArray | None
        Gradient of the dual at the cached solution.
    current_hess : FloatNDArray | None
        Hessian of the dual at the cached solution (if computed).
    current_xstar : ComplexArray | None
        Primal maximizer x* corresponding to current_lags.
    use_precomp : bool
        Whether precomputed constraint matrices (A_k) and Fs vectors are used.
    precomputed_As : list[sp.csc_array | ComplexArray]
        Symmetrized matrices A_k = Sym(A1 P_k A2) plus general constraint blocks.
    Fs : ComplexArray
        Matrix with columns (A2^† P_k^† s1) for each projector constraint k.

    Notes
    -----
    - General (B_j) constraints are appended after projector constraints in
      precomputed_As order.
    """

    def __init__(
        self,
        A0: ArrayLike | sp.csc_array,
        s0: ArrayLike,
        c0: float,
        A1: ArrayLike | sp.csc_array,
        A2: ArrayLike | sp.csc_array,
        s1: ArrayLike,
        Plist: list[SparseDense],
        Pstruct: SparseDense | None = None,
        B_j: list[SparseDense] | None = None,
        s_2j: list[ArrayLike] | None = None,
        c_2j: ArrayLike | None = None,
        verbose: int = 0,
        strong_duality_checker: Callable[[ComplexArray, float], bool] | None = None,
        A2dagger_inv_s0_func: Callable[[ComplexArray, ComplexArray], ComplexArray]
        | None = None,
    ) -> None:
        if B_j is None:
            all_mat_sp = [sp.issparse(A0), sp.issparse(A1)]
        else:
            all_mat_sp = [sp.issparse(Bj) for Bj in B_j] + [
                sp.issparse(A0),
                sp.issparse(A1),
            ]
        # A2 may be sparse even if using dense formulation
        all_sparse = np.all(all_mat_sp)
        all_dense = not np.any(all_mat_sp)
        assert all_sparse or all_dense, (
            "All quadratic matrices must be either sparse or dense."
        )

        if all_sparse:
            self.A0 = sp.csc_array(A0)
            self.A1 = sp.csc_array(A1)
            if B_j is None:
                self.B_j = []
            else:
                self.B_j = [sp.csc_array(Bj) for Bj in B_j]
        elif all_dense:
            self.A0 = np.asarray(A0, dtype=complex)
            self.A1 = np.asarray(A1, dtype=complex)
            if B_j is None:
                self.B_j = []
            else:
                self.B_j = [np.asarray(Bj, dtype=complex) for Bj in B_j]

        if sp.issparse(A2):
            self.A2 = sp.csc_array(A2)
        else:
            self.A2 = np.asarray(A2, dtype=complex)

        # Cast vectors to ComplexArray
        self.s0 = np.asarray(s0, dtype=complex)
        self.s1 = np.asarray(s1, dtype=complex)
        if s_2j is None:
            self.s_2j = []
        else:
            self.s_2j = [np.asarray(s2j, dtype=complex) for s2j in s_2j]
        if c_2j is None:
            self.c_2j = np.array([], dtype=float)
        else:
            self.c_2j = np.asarray(c_2j, dtype=float)

        if Pstruct is None:
            # capture Pstruct as the superset of all sparsity structures in Plist
            # Ensure complex dtype by default
            P0 = Plist[0]
            Pstruct = (
                P0.astype(complex).copy()
                if sp.issparse(P0)
                else np.asarray(P0, dtype=complex).copy()
            )
            for P in Plist:
                # Use complex scalar to guarantee complex promotion
                coef = (np.random.rand() + 0.01) + 0j
                Pcomplex = (
                    P.astype(complex)
                    if sp.issparse(P)
                    else np.asarray(P, dtype=complex)
                )
                Pstruct += coef * Pcomplex

        # Ensure complex dtype when converting to CSC
        Pstruct = sp.csc_array(Pstruct, dtype=complex)

        self.Proj = Projectors(Plist, Pstruct)
        self.n_proj_constr = len(self.Proj)
        self.n_gen_constr = len(self.B_j)

        assert len(self.c_2j) == len(self.s_2j), (
            "Length of c_2j must match length of s_2j."
        )
        assert len(self.c_2j) == len(self.B_j), (
            "Length of c_2j must match number of general constraints."
        )

        self.c0 = c0
        self.verbose = verbose
        self.Acho: Optional[Any] = None
        self.current_dual: Optional[float] = None
        self.current_grad: Optional[FloatNDArray] = None
        self.current_hess: Optional[FloatNDArray] = None
        self.current_lags: Optional[FloatNDArray] = None
        self.current_xstar: Optional[ComplexArray] = None
        self.use_precomp = True

        # Optional function callbacks
        self.strong_duality_checker: Optional[
            Callable[[ComplexArray, float], Tuple[float, bool]]
        ] = None
        # Optional PSD constraint projector (to re-append when needed)
        self.psd_projector: Optional[SparseDense] = None

        if self.use_precomp:
            self.compute_precomputed_values()

    def __deepcopy__(self, memo: dict[int, Any]) -> "_SharedProjQCQP":
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k == "Acho":
                setattr(new_obj, k, None)
            else:
                setattr(new_obj, k, copy.deepcopy(v, memo))
        return new_obj

    def reinitialize_A_operators(
        self,
        A0: ArrayLike | sp.csc_array,
        A1: ArrayLike | sp.csc_array,
        A2: ArrayLike | sp.csc_array,
    ) -> None:
        """Replace the A0, A1, and A2 operators.

        Only re-computes necessary quantities. Used in Verlan scheme.
        """
        existing_sparse = sp.issparse(self.A0)
        incoming_sparse = sp.issparse(A0)
        if existing_sparse != incoming_sparse:
            raise ValueError("Reinitialization must preserve sparse/dense storage.")

        if existing_sparse:
            self.A0 = sp.csc_array(A0, dtype=complex)
            self.A1 = sp.csc_array(A1, dtype=complex)
        else:
            self.A0 = np.asarray(A0, dtype=complex)
            self.A1 = np.asarray(A1, dtype=complex)

        self.A2 = (
            sp.csc_array(A2, dtype=complex)
            if sp.issparse(A2)
            else np.asarray(A2, dtype=complex)
        )

        self.Acho = None
        self.current_dual = None
        self.current_grad = None
        self.current_hess = None
        self.current_lags = None
        self.current_xstar = None

        if self.use_precomp:
            self.compute_precomputed_values()
        elif hasattr(self, "_initialize_Acho"):
            self._initialize_Acho()

    def set_is_strongly_dual(
        self, func: Callable[[ComplexArray, float], Tuple[float, bool]]
    ) -> None:
        """Set a custom function to check for strong duality."""
        self.strong_duality_checker = func  # type: ignore

    def is_strongly_dual(
        self, xstar: ComplexArray | None, tol: float = 1e-3
    ) -> Tuple[float, bool]:
        """
        Check if the problem satisfies strong duality conditions (default: A is PD).

        Parameters
        ----------
        xstar : ComplexArray
            The primal solution (unused in default check).
        tol : float
            Tolerance for checking strict positivity of eigenvalues.

        Returns
        -------
        violation : float
            Amount of violation (tol - min_eig) if not PD, else 0.0.
        is_sd : bool
            True if strongly dual (PD implies unique primal), False otherwise.
        """
        if self.strong_duality_checker is not None and xstar is not None:
            return self.strong_duality_checker(xstar, tol)  # type: ignore

        if self.current_lags is None:
            raise ValueError("Undefined lags, cannot check strong duality.")

        A = self._get_total_A(self.current_lags)
        min_eig: float

        if sp.issparse(A):
            try:
                vals = spla.eigsh(
                    A, k=1, sigma=0.0, which="LM", return_eigenvectors=False
                )
                min_eig = float(vals[0])
            except (spla.ArpackNoConvergence, spla.ArpackError) as e:
                if self.verbose > 0:
                    print(f"Warning: eigsh failed in is_strongly_dual: {e}")
                return tol, False
        else:
            vals = la.eigvalsh(A, subset_by_index=[0, 0])
            min_eig = float(vals[0])

        if min_eig < -1e-6:
            warnings.warn(
                f"Found negative eigenvalue {min_eig:.3e} in is_strongly_dual check."
            )

        is_sd = min_eig > tol
        violation = max(0.0, tol - min_eig) if not is_sd else 0.0

        return violation, is_sd

    def compute_precomputed_values(self) -> None:
        """
        Precompute per-constraint symmetrized matrices and projector-source terms.

        Precomputes:
          - precomputed_As are the quadratic forms for x in the constraints
            = Sym(A1 P_k A2) for each projector constraint k
            = Sym(A2^† B_j A2) for each general constraint j
          - Fs are the linear forms for x in the constraints
            = A2^† P_k^† s1 for each projector constraint k
            = A2^† s_2j for each general constraint j

        This speeds up repeated assembly of A(lags) and derivative-related
        operations when the number of constraints is moderate.
        """
        self.precomputed_As = []
        for i in range(self.n_proj_constr):
            Ak = Sym(self.A1 @ self.Proj[i] @ self.A2)
            self.precomputed_As.append(Ak)
        for i in range(len(self.B_j)):
            self.precomputed_As.append(Sym(self.A2.conj().T @ self.B_j[i] @ self.A2))

        self.Fs: ComplexArray = np.zeros(
            (self.A2.shape[1], len(self.precomputed_As)), dtype=complex
        )
        # For diagonal P: allP_at_v(self.s1, dagger=True) == (Pdiags.conj().T * s1).T
        if self.n_proj_constr > 0:
            Pv = self.Proj.allP_at_v(self.s1, dagger=True)  # shape (n, k)
            self.Fs[:, : self.n_proj_constr] = self.A2.conj().T @ Pv  # shape (m, k)
        if self.n_gen_constr > 0:
            self.Fs[:, self.n_proj_constr :] = self.A2.conj().T @ np.column_stack(
                self.s_2j
            )

        if self.verbose > 0:
            print(
                f"Precomputed {self.n_proj_constr + self.n_gen_constr}"
                " A matrices and Fs vectors."
            )

    def get_number_constraints(self) -> int:
        """Return total number of constraints (projector + general)."""
        return self.n_proj_constr + self.n_gen_constr

    def _add_projectors(self, lags: FloatNDArray) -> ComplexArray:
        """Form the diagonal of sum_j λ_j P_j (diagonal projectors only)."""
        # Retained for compatibility where a diagonal vector is explicitly needed.
        if hasattr(self.Proj, "Pdiags"):
            return cast(ComplexArray, self.Proj.Pdiags @ lags[: self.n_proj_constr])
        raise NotImplementedError("Combined diagonal only defined for diagonal proj.")

    def _get_total_A(self, lags: FloatNDArray) -> sp.csc_array | ComplexArray:
        """Return A(lags) = A0 + Σ_j lags[j] * Sym(A1 P_j A2) (+ general parts).

        Uses precomputed matrices if enabled; otherwise assembles on the fly.
        """
        return (
            self._get_total_A_precomp(lags)
            if self.use_precomp
            else self._get_total_A_noprecomp(lags)
        )

    def _get_total_A_precomp(self, lags: FloatNDArray) -> sp.csc_array | ComplexArray:
        """Return total A using precomputed_As (fast path for few constraints)."""
        return self.A0 + sum(lags[i] * self.precomputed_As[i] for i in range(len(lags)))

    def _get_total_A_noprecomp(self, lags: FloatNDArray) -> sp.csc_array | ComplexArray:
        """Return total A without precomputation (better for many constraints)."""
        raise NotImplementedError(
            "_get_total_A_noprecomp not implemented; use precomputation."
        )

    def _get_total_S(self, lags: FloatNDArray) -> ComplexArray:
        """Return S(lags), the linear form of x in the Lagrangian.

        S = s0 + A2^† (Σ_j λ_j P_j^† s1) + Σ_general μ_j (A2^† s_2j).

        Parameters
        ----------
        lags : FloatNDArray
            Lagrange multipliers for all constraints
            (length = n_proj_constr + n_gen_constr).

        Returns
        -------
        ComplexArray
            The combined linear term S used in A x = S.
        """
        S: ComplexArray
        if hasattr(self, "Fs"):
            S = self.s0 + self.Fs @ lags
        else:
            # Σ λ_j P_j^† s1
            proj_lags = lags[: self.n_proj_constr]
            y = self.Proj.weighted_sum_on_vector(self.s1, proj_lags, dagger=True)
            S = self.s0 + self.A2.conj().T @ y
            Blags = lags[self.n_proj_constr :]
            S += sum(
                Blags[i] * (self.A2.conj().T @ self.s_2j[i])
                for i in range(len(self.B_j))
            )

        return S

    def _get_total_C(self, lags: FloatNDArray) -> float:
        """Return Σ_general μ_j c_2j (0 if no general constraints)."""
        return cast(float, np.sum(lags[self.n_proj_constr :] * self.c_2j))

    @abstractmethod
    def _update_Acho(self, A: sp.csc_array | ComplexArray) -> None:
        """
        Update the Cholesky factorization to be that of the input matrix A.

        Needs to be implemented by subclasses.

        Parameters
        ----------
        A : sp.csc_array | ComplexArray
            New total Hermitian matrix to factorize (or store for dense solve).
        """
        pass

    @abstractmethod
    def _Acho_solve(self, b: ComplexArray) -> ComplexArray:
        """
        Solve A x = b using the current factorization.

        Parameters
        ----------
        b : ComplexArray
            Right-hand side vector (or stacked RHS columns).

        Returns
        -------
        ComplexArray
            Solution vector(s) x = A^{-1} b.
        """
        pass

    @abstractmethod
    def is_dual_feasible(self, lags: FloatNDArray) -> bool:
        """
        Check positive semidefiniteness of A(lags).

        Parameters
        ----------
        lags : FloatNDArray
            Full Lagrange multiplier vector (projector part first, then general).

        Returns
        -------
        bool
            True if A(lags) is PSD (dual feasible).
        """
        pass

    def find_feasible_lags(
        self, start: float = 0.1, limit: float = 1e8, idx: int = 1
    ) -> FloatNDArray:
        """
        Heuristically find a dual feasible (PSD) set of Lagrange multipliers.

        Assumes scaling up (typically) the second projector multiplier eventually
        yields a PSD A matrix.

        Parameters
        ----------
        start : float, default 0.1
            Initial value assigned to lags[idx].
        limit : float, default 1e8
            Upper bound before giving up.
        idx : int, default 1
            Index of the projector constraint to scale.

        Returns
        -------
        FloatNDArray
            Feasible initial lags (projector first, then zeros for general constraints).
        """
        if self.current_lags is not None:
            if self.is_dual_feasible(self.current_lags):
                return self.current_lags

        # Start with small positive lags
        init_lags = np.random.random(self.n_proj_constr) * 1e-6
        init_lags = np.append(init_lags, len(self.B_j) * [0.0])

        init_lags[idx] = start
        while self.is_dual_feasible(init_lags) is False:
            init_lags[idx] *= 1.5
            if init_lags[idx] > limit:
                raise ValueError(
                    "Could not find a feasible point for the dual problem."
                )

        if self.verbose > 0:
            print(
                f"Found feasible point for dual problem: {init_lags} with "
                f"dualvalue {self.get_dual(init_lags)[0]}"
            )
        return init_lags

    def _get_PSD_penalty(self, lags: FloatNDArray) -> Tuple[ComplexArray, float]:
        """
        Return (v, λ_min) where λ_min is the extremal eigenvalue closest to 0.

        Uses shift-invert (eigsh with sigma=0.0) to approximate the smallest
        magnitude eigenvalue/eigenvector of A(lags) for PSD boundary penalization.

        Parameters
        ----------
        lags : FloatNDArray
            Lagrange multipliers.

        Returns
        -------
        v : ComplexArray
            Eigenvector associated with the returned eigenvalue.
        lam : float
            Corresponding eigenvalue (should be ≥ 0 at feasibility).
        """
        A = self._get_total_A(lags)
        eigw, eigv = spla.eigsh(A, k=1, sigma=0.0, which="LM", return_eigenvectors=True)
        return eigv[:, 0], eigw[0]

    def _get_xstar(self, lags: FloatNDArray) -> Tuple[ComplexArray, float]:
        """
        Solve A(lags) x* = S(lags); return x* and x*^† A x* (dual contribution).

        Parameters
        ----------
        lags : FloatNDArray
            Lagrange multipliers (projector + general).

        Returns
        -------
        x_star : ComplexArray
            Primal maximizing vector for current lags.
        xAx : float
            Value x*^† A x* (real scalar).
        """
        A = self._get_total_A(lags)
        S = self._get_total_S(lags)
        self._update_Acho(A)
        x_star: ComplexArray = self._Acho_solve(S)
        xAx: float = np.real(np.vdot(x_star, A @ x_star))

        return x_star, xAx

    def get_dual(
        self,
        lags: FloatNDArray,
        get_grad: bool = False,
        get_hess: bool = False,
        penalty_vectors: Optional[list[ComplexArray]] = None,
    ) -> Tuple[float, Optional[FloatNDArray], Optional[FloatNDArray], Any]:
        """
        Evaluate dual function and optional derivatives at lags.

        Parameters
        ----------
        lags : FloatNDArray
            Lagrange multipliers (projector first, then general).
        get_grad : bool, default False
            If True, compute gradient w.r.t. lags.
        get_hess : bool, default False
            If True, compute Hessian (only supported when no general constraints).
        penalty_vectors : list[ComplexArray] | None
            Optional PSD-boundary penalty vectors.

        Returns
        -------
        dualval : float
            Dual objective value (with penalties if provided).
        grad : FloatNDArray | None
            Gradient (None if not requested).
        hess : FloatNDArray | None
            Hessian (None if not requested).
        dual_aux : namedtuple
            Auxiliary components (raw + penalty-separated parts).

        Notes
        -----
        - Penalty adjustment adds Σ_j v_j^† A^{-1} v_j and its derivatives.
        """
        if penalty_vectors is None:
            penalty_vectors = []

        # Make Optional explicit for mypy
        grad: Optional[FloatNDArray]
        hess: Optional[FloatNDArray]
        grad, hess = None, None
        # Default penalty containers so DualAux is always defined
        grad_penalty: FloatNDArray = np.zeros(0, dtype=float)
        hess_penalty: FloatNDArray = np.zeros((0, 0), dtype=float)

        xstar, dualval = self._get_xstar(lags)
        dualval += self.c0 + self._get_total_C(lags)

        if get_hess:
            try:
                # useful intermediate computations
                # (Fx)_k = -A_k @ x_star
                # where A_k is quadratic form of constraints

                Fx = np.zeros((len(xstar), len(self.precomputed_As)), dtype=complex)
                for k, Ak in enumerate(self.precomputed_As):
                    Fx[:, k] = -Ak @ xstar

                # get_hess implies get_grad also
                grad = np.real(xstar.conj() @ (Fx + 2 * self.Fs))
                grad[self.n_proj_constr :] += self.c_2j

                Ftot = Fx + self.Fs
                hess = 2 * np.real(Ftot.conj().T @ self._Acho_solve(Ftot))
            except AttributeError:
                # this assumes that in the future we may consider making
                # precomputed_As optional can also compute the Hessian without
                # precomputed_As, leave for future implementation if useful
                raise AttributeError("precomputed_As needed for computing Hessian")

        elif get_grad:
            # Generic projector gradient (works for diagonal and general P)
            # Let y := A2 @ x*, u := A1^H @ x*
            # For diagonal projectors only (legacy), we had:
            #   term1_diag = -Re((x*^H A1) @ (Pdiags * y[:, None]))
            #              = -Re(((x*^H A1) ⊙ y^T) @ Pdiags)
            #              = -Re((x_conj_A1 * y) @ Pdiags)
            #   term2_diag =  2 Re(y^H @ (Pdiags^* ⊙ s1))
            #              =  2 Re((y^* ⊙ s1)^T @ Pdiags^*)
            # New unified expressions using Projectors:
            #   Py       = [P_1 y, ..., P_k y]            (n x k)
            #   Ps1_dag  = [P_1^† s1, ..., P_k^† s1]      (n x k)
            #   term1    = -Re(u^H Py)                    (k,)
            #   term2    =  2 Re(y^H Ps1_dag)             (k,)
            # For diagonal P, Py == Pdiags ⊙ y and Ps1_dag == Pdiags^* ⊙ s1, so the
            # new expressions reduce exactly to the legacy vectorized formulas above.
            A2_xstar = self.A2 @ xstar  # (n,)  aka y
            u = self.A1.conj().T @ xstar  # (n,)  aka A1^H x*
            Py = self.Proj.allP_at_v(A2_xstar)  # (n, k)  columns: P_j y

            term1_proj = -np.real(u.conj() @ Py)  # (k,) float64
            term2_proj = 2.0 * np.real(
                xstar.conj() @ self.Fs[:, : self.n_proj_constr]
            )  # (k,) float64
            proj_grad = term1_proj + term2_proj  # (k,) float64

            if self.n_gen_constr > 0:
                gen_constr_grad = np.zeros(self.n_gen_constr, dtype=float)
                for i in range(self.n_gen_constr):
                    gen_constr_grad[i] = np.real(
                        -xstar.conj().T
                        @ self.A2.conj().T
                        @ self.B_j[i]
                        @ self.A2
                        @ xstar
                        + 2 * xstar.conj().T @ self.A2.conj().T @ self.s_2j[i]
                        + self.c_2j[i]
                    )
                grad_full = np.concatenate((proj_grad, gen_constr_grad))
                grad = cast(FloatNDArray, grad_full)
            else:
                grad = cast(FloatNDArray, proj_grad)

        # Boundary penalty for the PSD boundary
        dualval_penalty = 0.0
        if len(penalty_vectors) > 0:
            penalty_matrix = np.column_stack(penalty_vectors).astype(complex)
            A_inv_penalty = self._Acho_solve(penalty_matrix)
            dualval_penalty += np.sum(
                np.real(A_inv_penalty.conj() * penalty_matrix)
            )  # multiplies columns with columns, sums all at once

            if get_hess:
                grad = cast(FloatNDArray, grad)
                penalty_matrix = cast(ComplexArray, penalty_matrix)
                # get_hess implies get_grad also
                grad_penalty = np.zeros(grad.shape[0])
                hess_penalty = np.zeros((grad.shape[0], grad.shape[0]))

                Fv = np.zeros((penalty_matrix.shape[0], len(grad)), dtype=complex)
                for j in range(penalty_matrix.shape[1]):
                    for k, Ak in enumerate(self.precomputed_As):
                        # yes this is a double for loop, hessian for fake sources
                        # is likely a speed bottleneck
                        Fv[:, k] = Ak @ A_inv_penalty[:, j]

                grad_penalty += np.real(-A_inv_penalty[:, j].conj().T @ Fv)
                hess_penalty += 2 * np.real(Fv.conj().T @ self._Acho_solve(Fv))

            elif get_grad:
                grad = cast(FloatNDArray, grad)
                P = self.n_proj_constr
                G = self.n_gen_constr

                proj_grad_penalty = np.zeros(P)
                for j in range(penalty_matrix.shape[1]):
                    pj = A_inv_penalty[:, j]
                    yj = self.A2 @ pj
                    uj = self.A1.conj().T @ pj
                    Pyj = self.Proj.allP_at_v(yj)  # (n, k)
                    proj_grad_penalty += -np.real(uj.conj() @ Pyj)  # (k,)

                if G > 0:
                    gen_constr_grad_penalty = np.zeros(G)
                    for i in range(G):
                        for j in range(penalty_matrix.shape[1]):
                            gen_constr_grad_penalty[i] += np.real(
                                -A_inv_penalty[:, j].conj().T
                                @ self.A2.conj().T
                                @ self.B_j[i]
                                @ self.A2
                                @ A_inv_penalty[:, j]
                            )
                    grad_penalty = np.zeros(P + G)
                    grad_penalty[:P] = proj_grad_penalty
                    grad_penalty[P:] = gen_constr_grad_penalty
                else:
                    grad_penalty = proj_grad_penalty

        DualAux = namedtuple(
            "DualAux",
            [
                "dualval_real",
                "dualgrad_real",
                "dualval_penalty",
                "grad_penalty",
                "hess_penalty",
            ],
        )
        dual_aux = DualAux(
            dualval_real=dualval,
            dualgrad_real=grad,
            dualval_penalty=dualval_penalty,
            grad_penalty=grad_penalty,
            hess_penalty=hess_penalty,
        )

        if len(penalty_vectors) > 0:
            grad_out: Optional[FloatNDArray] = (
                None if grad is None else grad + grad_penalty
            )
            hess_out: Optional[FloatNDArray] = (
                None if hess is None else hess + hess_penalty
            )
            return (
                dualval + dualval_penalty,
                grad_out,
                hess_out,
                dual_aux,
            )
        else:
            return dualval, grad, hess, dual_aux

    def check_constraint(self, delta: float, idx: int) -> bool:
        """Check if a constraint holds within tolerance delta.

        Arguments
        ---------
        delta : float
            Tolerance for checking if the constraint gradient is zero.
        idx : int
            Index of the constraint to check.

        Returns
        -------
        bool
            True if the constraint holds within tolerance.
        """
        gradient = self.current_grad
        if gradient is None:
            raise ValueError("QCQP has not been solved yet.")
        compact_gradient = gradient[idx]
        return bool(np.linalg.norm(compact_gradient) < delta)

    def solve_current_dual_problem(
        self,
        method: str,
        opt_params: Optional[OptimizationHyperparameters] = None,
        init_lags: Optional[FloatNDArray] = None,
    ) -> Tuple[float, FloatNDArray, FloatNDArray, Optional[FloatNDArray], ComplexArray]:
        """
        Optimize the dual problem using 'newton' or 'bfgs'.

        Parameters
        ----------
        method : str
            'newton' (alternating Newton / GD) or 'bfgs'.
        opt_params : OptimizationHyperparameters | None
            Optimization hyperparameters (if None, default values are used).
        init_lags : ArrayLike | None
            Initial feasible lags; if None, a feasible point is searched.

        Returns
        -------
        current_dual : float
            Optimal dual value.
        current_lags : FloatNDArray
            Optimal Lagrange multipliers.
        current_grad : FloatNDArray
            Gradient at optimum.
        current_hess : FloatNDArray | None
            Hessian (if computed by Newton variant).
        current_xstar : ComplexArray
            Primal maximizer corresponding to current_lags.
        """
        is_convex = True

        if opt_params is None:
            opt_params = OptimizationHyperparameters(
                opttol=1e-2,
                gradConverge=False,
                min_inner_iter=5,
                max_restart=np.inf,
                penalty_ratio=1e-2,
                penalty_reduction=0.1,
                break_iter_period=20,
                verbose=int(self.verbose - 1),
            )

        if init_lags is None:
            init_lags = self.find_feasible_lags()
        init_lags = np.array(init_lags, float)

        optfunc = self.get_dual
        feasibility_func = self.is_dual_feasible
        penalty_vector_func = self._get_PSD_penalty

        optimizer: _Optimizer
        if method == "newton":
            optimizer = Alt_Newton_GD(
                optfunc, feasibility_func, penalty_vector_func, is_convex, opt_params
            )
        elif method == "bfgs":
            optimizer = BFGS(
                optfunc, feasibility_func, penalty_vector_func, is_convex, opt_params
            )
        else:
            raise ValueError(
                f"Unknown method '{method}' for solving the dual problem. "
                "Use newton or bfgs."
            )

        self.current_lags, self.current_dual, self.current_grad, self.current_hess = (
            optimizer.run(init_lags)
        )
        self.current_xstar = self._get_xstar(self.current_lags)[0]

        return (
            self.current_dual,
            self.current_lags,
            self.current_grad,
            self.current_hess,
            self.current_xstar,
        )

    def merge_lead_constraints(self, merged_num: int = 2) -> None:
        """
        Merge first 'merged_num' projector constraints into one (GCD utility).

        Parameters
        ----------
        merged_num : int, default 2
            Number of leading projector constraints to merge.
        """
        from . import gcd as _gcd

        _gcd.merge_lead_constraints(self, merged_num=merged_num)

    def add_constraints(
        self, added_Pdata_list: list[ComplexArray], orthonormalize: bool = True
    ) -> None:
        """
        Append additional projector constraints.

        Parameters
        ----------
        added_Pdiag_list : list[ComplexArray]
            List of new projector diagonals.
        orthonormalize : bool, default True
            Whether to orthonormalize constraint set after insertion.
        """
        from . import gcd as _gcd

        _gcd.add_constraints(
            self, added_Pdata_list=added_Pdata_list, orthonormalize=orthonormalize
        )

    def run_gcd(
        self,
        gcd_params: "Optional[GCDHyperparameters]" = None,
    ) -> None:
        """Run GCD to approach tightest dual bound for this QCQP.

        See module-level run_gcd() for details. Modifies the existing QCQP object.

        Parameters
        ----------
        gcd_params : GCDHyperparameters | None
            GCD hyperparameters; if None, defaults are used.
        """
        if gcd_params is None:
            from .gcd import GCDHyperparameters as _GCDHyperparameters

            gcd_params = _GCDHyperparameters()
        from . import gcd as _gcd

        _gcd.run_gcd(self, gcd_params)

    def refine_projectors(self) -> Tuple[Any, NDArray[np.float64]]:
        """
        Refine projector constraints by splitting each projector into two projectors.

        Strategy (diagonal):
          - Identify non-empty column support of P (columns with nnz > 0).
          - If support size <= 1: keep P as is.
          - Else, build two diagonal column masks S1, S2 that partition the support
            roughly in half, and define P1 = P @ S1, P2 = P @ S2 so that P = P1 + P2.
          - Duplicate the parent multiplier on both children so Σ λ P remains unchanged.
        General constraints (B_j, s_2j, c_2j) are left unchanged.

        Returns
        -------
        self.Proj, self.current_lags
        """
        if self.current_lags is None:
            raise AssertionError(
                "Cannot refine projectors until an existing problem is solved. "
                "Run solve_current_dual_problem first."
            )
        if self.Proj.is_diagonal() is False:
            raise NotImplementedError(
                "Refinement only implemented for diagonal projectors."
            )

        curr_lags = self.current_lags

        # Preserve current dual for verification
        old_dual = self.get_dual(curr_lags, get_grad=False)[0]

        # Extract old projectors and lags
        old_proj_count = self.n_proj_constr
        old_proj_lags = np.array(curr_lags[:old_proj_count], float)
        old_gen_lags = (
            np.array(curr_lags[old_proj_count:], float)
            if self.n_gen_constr > 0
            else np.array([], float)
        )

        new_Plist = []
        new_proj_lags_list = []

        for j in range(old_proj_count):
            Pj = self.Proj[j]  # sp.csc_array
            # Column support: columns with any nnz
            col_nnz = np.diff(Pj.indptr)
            support = np.where(col_nnz > 0)[0]
            if support.size <= 1:
                # Keep as-is
                new_Plist.append(Pj)
                new_proj_lags_list.append(old_proj_lags[j])
                continue

            # Split support into two halves
            mid = support.size // 2
            idx1, idx2 = support[:mid], support[mid:]

            # Build diagonal masks S1, S2 (0-1 on selected columns)
            n = Pj.shape[1]
            mask1 = np.zeros(n, dtype=float)
            mask1[idx1] = 1.0
            mask2 = np.zeros(n, dtype=float)
            mask2[idx2] = 1.0
            S1 = sp.diags_array(mask1, format="csc")
            S2 = sp.diags_array(mask2, format="csc")

            # Column partition: P = P S1 + P S2 exactly
            P1 = Pj @ S1
            P2 = Pj @ S2

            # Append children; duplicate parent's lag on both so contribution equals old
            new_Plist.append(P1)
            new_Plist.append(P2)
            new_proj_lags_list.append(old_proj_lags[j])
            new_proj_lags_list.append(old_proj_lags[j])

        # Replace projectors and update counts
        self.Proj = Projectors(new_Plist, self.Proj.Pstruct)
        self.n_proj_constr = len(new_Plist)

        # New lags: concatenated projector lags + unchanged general lags
        new_lags = np.concatenate([np.array(new_proj_lags_list, float), old_gen_lags])
        self.current_lags = new_lags

        self.compute_precomputed_values()

        assert self.current_lags is not None
        new_dual = self.get_dual(self.current_lags, get_grad=False)[0]
        if self.verbose >= 1:
            print(f"previous dual: {old_dual}, new dual: {new_dual} (should match)")
        assert np.isclose(new_dual, old_dual, rtol=1e-2, atol=1e-8), (
            "Dual value should be unchanged after refinement."
        )

        # Keep cached values consistent
        self.current_dual = new_dual
        # current_grad/hess/xstar are now stale; recompute lazily when next requested

        return self.Proj, self.current_lags

    def iterative_splitting_step(
        self, method: str = "bfgs", max_proj_cstrt_num: int | float = np.inf
    ) -> Iterator[
        Tuple[float, FloatNDArray, FloatNDArray, Optional[FloatNDArray], ComplexArray]
    ]:
        """
        Generate successive solutions by refining projectors and solving the dual.

        Stop when:
        - each projector has at most one non-zero column (pixel level for diagonal), or
        - the number of constraints reaches max_proj_cstrt_num.

        Yields
        ------
        tuple
            (current_dual, current_lags, current_grad, current_hess, current_xstar)
        """

        def projector_column_support_sizes() -> NDArray[np.int_]:
            sizes = []
            for j in range(self.n_proj_constr):
                Pj = self.Proj[j]
                col_nnz = np.diff(Pj.indptr)
                sizes.append(int(np.count_nonzero(col_nnz)))
            return np.array(sizes, int)

        # Early exit if already above cap or pixel-level
        max_proj_cstrt_num = int(min(max_proj_cstrt_num, 2 * (self.A0.shape[0])))
        if self.n_proj_constr >= max_proj_cstrt_num:
            if self.verbose > 0:
                print("Projector count already at or above specified maximum.")
            return

        while True:
            sizes = projector_column_support_sizes()
            if self.n_proj_constr >= max_proj_cstrt_num or np.all(sizes <= 1):
                if self.verbose > 0:
                    print("Reached maximum projectors or pixel-level constraints.")
                break

            if self.verbose > 0:
                print(f"Splitting projectors: {self.n_proj_constr} → ", end="")
            # Refine (updates self.Proj/self.current_lags and recomputes precomputation)
            self.refine_projectors()
            if self.verbose > 0:
                print(f"{self.n_proj_constr}")

            # Solve with the new refined projector set, warm-start with current_lags
            result = self.solve_current_dual_problem(
                method, init_lags=self.current_lags
            )
            yield result
