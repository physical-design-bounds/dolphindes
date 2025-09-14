from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Optional, Tuple, cast

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import ArrayLike

from dolphindes.cvxopt import gcd
from dolphindes.types import ComplexArray, FloatNDArray
from dolphindes.util import Sym

from .optimization import BFGS, Alt_Newton_GD, _Optimizer


class _SharedProjQCQP(ABC):
    """Represents a quadratically constrained quadratic program (QCQP).
    
    QCQP has a set of projection-structured (shared) constraints and (optionally) a
    separate list of general quadratic constraints.

    Primal problem (maximization form):
        maximize_x   - x^† A0 x + 2 Re(x^† s0) + c0
        subject to   Re( - x^† A1 P_j A2 x + 2 x^† A2^† P_j^† s1 ) = 0
                     Re( - x^† A2^† B_j A2 x + 2 x^† A2^† s_2j + c_2j ) = 0

    where each P_j is a (diagonal) projector represented only through its diagonal
    entries (the columns of Pdiags). The matrices used internally for the shared
    (projector) constraints are symmetrized via Sym(A1 P_j A2) to ensure Hermitian
    structure.

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
    Pdiags : ArrayLike
        Matrix whose columns are the diagonals of the projector matrices P_j.
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
    - Hessian computation is only implemented when there are NO general
      constraints (n_gen_constr == 0); requesting it otherwise raises
      NotImplementedError.
    """

    def __init__(
        self,
        A0: ArrayLike | sp.csc_array,
        s0: ArrayLike,
        c0: float,
        A1: ArrayLike | sp.csc_array,
        A2: ArrayLike | sp.csc_array,
        s1: ArrayLike,
        Pdiags: ArrayLike,
        B_j: list[ArrayLike | sp.csc_array] | None = None,
        s_2j: list[ArrayLike] | None = None,
        c_2j: ArrayLike | None = None,
        verbose: int = 0,
    ) -> None:
        if B_j is None:
            all_mat_sp = [sp.issparse(A0), sp.issparse(A1)]
        else:
            all_mat_sp = (
                [sp.issparse(Bj) for Bj in B_j]
                + [sp.issparse(A0), sp.issparse(A1)]
            )
        # A2 may be sparse even if using dense formulation
        all_sparse = np.all(all_mat_sp)
        all_dense = not np.any(all_mat_sp)
        assert (
            all_sparse or all_dense
        ), "All quadratic matrices must be either sparse or dense."

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

        self.Pdiags = np.asarray(Pdiags, dtype=complex)
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

        if self.use_precomp:
            self.compute_precomputed_values()

    def compute_precomputed_values(self) -> None:
        """
        Precompute per-constraint symmetrized matrices and projector-source terms.

        Precomputes:
          - precomputed_As[k] = Sym(A1 P_k A2) for each projector constraint k
          - then appends Sym(A2^† B_j A2) for each general constraint j
          - Fs[:, k] = A2^† P_k^† s1 for projector constraints (no general part)

        This speeds up repeated assembly of A(lags) and derivative-related
        operations when the number of constraints is moderate.
        """
        self.precomputed_As = []
        for i in range(self.Pdiags.shape[1]):
            Ak = Sym(
                self.A1 @ sp.diags_array(self.Pdiags[:, i], format="csr") @ self.A2
            )
            self.precomputed_As.append(Ak)
        for i in range(len(self.B_j)):
            self.precomputed_As.append(Sym(self.A2.conj().T @ self.B_j[i] @ self.A2))

        if self.verbose > 0:
            print(
                f"Precomputed {self.Pdiags.shape[1] + len(self.B_j)}"
                " A matrices for the projectors."
            )

        # (Fs)_k = A_2^dagger P_k^dagger s1
        self.Fs = self.A2.conj().T @ (self.Pdiags.conj().T * self.s1).T

    def get_number_constraints(self) -> int:
        """Return total number of constraints (projector + general)."""
        return self.Pdiags.shape[1] + len(self.B_j)

    def _add_projectors(self, lags: FloatNDArray) -> ComplexArray:
        """Form the diagonal of sum_j λ_j P_j using ONLY projector multipliers.

        Parameters
        ----------
        lags : FloatNDArray
            Full Lagrange multiplier vector (projector first, then general).
            Only the first P = Pdiags.shape[1] entries are used here.

        Returns
        -------
        ComplexArray
            Combined projector diagonal entries Σ_j λ_j P_j (as a vector).
        """
        return cast(ComplexArray, self.Pdiags @ lags[:self.Pdiags.shape[1]])

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
        # TODO(alessio): P_diag is usually already computed before calling
        # get_total_A. Keeping it like this to not have to change the passed
        # arguments, but should fix it at some point.
        P_diag = self._add_projectors(lags)
        return self.A0 + Sym(self.A1 @ sp.diags_array(P_diag, format="csr") @ self.A2)

    def _get_total_S(self, Pdiag: ComplexArray, Blags: FloatNDArray) -> ComplexArray:
        """Return S(lags) = s0 + A2^† ((Σ_j λ_j P_j)^† s1) + Σ_general μ_j (A2^† s_2j).

        Parameters
        ----------
        Pdiag : ComplexArray
            Diagonal of Σ_j λ_j P_j (projector part only).
        Blags : FloatNDArray
            Lagrange multipliers for general constraints (may be empty).

        Returns
        -------
        ComplexArray
            The combined linear term S used in A x = S.
        """
        S = cast(ComplexArray, self.s0 + self.A2.T.conj() @ (Pdiag.conj() * self.s1))
        # Could be optimized further by precomputing A2^dagger s_2j
        S += sum(
            Blags[i] * (self.A2.conj().T @ self.s_2j[i]) for i in range(len(self.B_j))
        )
        return S

    def _get_total_C(self, Blags: FloatNDArray) -> float:
        """Return Σ_general μ_j c_2j (0 if no general constraints)."""
        return cast(float, np.sum(Blags * self.c_2j))

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
        self, start: float = 0.1, limit: float = 1e8
    ) -> FloatNDArray:
        """
        Heuristically find a dual feasible (PSD) set of Lagrange multipliers.

        Assumes scaling up (typically) the second projector multiplier eventually
        yields a PSD A matrix.

        Parameters
        ----------
        start : float, default 0.1
            Initial value assigned to lags[1] (must have ≥ 2 projector constraints).
        limit : float, default 1e8
            Upper bound before giving up.

        Returns
        -------
        FloatNDArray
            Feasible initial lags (projector first, then zeros for general constraints).
        """
        if (self.current_lags is not None):
            if self.is_dual_feasible(self.current_lags):
                return self.current_lags

        # Start with small positive lags
        init_lags = np.random.random(int(self.Pdiags.shape[1])) * 1e-6
        init_lags = np.append(init_lags, len(self.B_j) * [0.0])
        
        init_lags[1] = start
        while self.is_dual_feasible(init_lags) is False:
            init_lags[1] *= 1.5
            if init_lags[1] > limit:
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
        Blags = lags[-self.n_gen_constr:] if self.n_gen_constr > 0 else np.array([])
        P_diag = self._add_projectors(lags)
        A = self._get_total_A(lags)
        S = self._get_total_S(P_diag, Blags)
        self._update_Acho(A)  # update the Cholesky factorization

        x_star: ComplexArray = self._Acho_solve(S)
        xAx: float = np.real(np.vdot(x_star, A @ x_star))

        return x_star, xAx

    def get_dual(
        self,
        lags: FloatNDArray,
        get_grad: bool = False,
        get_hess: bool = False,
        penalty_vectors: Optional[list[FloatNDArray]] = None,
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
        penalty_vectors : list[FloatNDArray] | None
            Optional PSD-boundary penalty vectors.

        Returns
        -------
        dualval : float
            Dual objective value (with penalties if provided).
        grad : FloatNDArray | None
            Gradient (None if not requested).
        hess : FloatNDArray | None
            Hessian (None if not requested / unsupported).
        dual_aux : namedtuple
            Auxiliary components (raw + penalty-separated parts).

        Notes
        -----
        - Hessian currently unsupported with general constraints.
        - Penalty adjustment adds Σ_j v_j^† A^{-1} v_j and its derivatives.
        - Potential shape issue if get_hess=True and general constraints present
          (Fx + 2*Fs width mismatch) occurs before raising NotImplementedError.
        """
        if penalty_vectors is None:
            penalty_vectors = []

        grad, hess = None, None
        grad_penalty, hess_penalty = np.array([]), np.array([[]])

        xstar, dualval = self._get_xstar(lags)
        Blags = lags[-self.n_gen_constr:] if self.n_gen_constr > 0 else np.array([])
        dualval += (self.c0 + self._get_total_C(Blags))

        if get_hess:
            if not hasattr(self, "precomputed_As"):
                raise AttributeError("precomputed_As needed for computing Hessian")
                # this assumes that in the future we may consider making
                # precomputed_As optional can also compute the Hessian without
                # precomputed_As, leave for future implementation if useful

            # useful intermediate computations
            # (Fx)_k = -Sym(A_1 P_k A2) x_star = -A_k @ x_star

            Fx = np.zeros((len(xstar), len(self.precomputed_As)), dtype=complex)
            for k, Ak in enumerate(self.precomputed_As):
                Fx[:, k] = -Ak @ xstar

            # get_hess implies get_grad also
            grad = np.real(xstar.conj() @ (Fx + 2 * self.Fs))

            if self.n_gen_constr == 0:
                Ftot = Fx + self.Fs
                hess = 2 * np.real(Ftot.conj().T @ self._Acho_solve(Ftot))
            else:
                # TODO: implement Hessian with general constraints
                # Because there may be cross terms beteween normal and general
                # constraints, we need to compute it with a double for loop.
                # Is there a way to compute the diagonal terms for the shared
                # constraints faster? Then only compute cross terms + general
                # diagonal components with a double for loop?
                raise NotImplementedError(
                    "Hessian computation with general constraints not implemented."
                )

        elif get_grad: 
            # This is grad_lambda (not grad_x); elif since get_hess computes grad
            # First term: -Re(xstar.conj() @ self.A1 @ (self.Pdiags[:, i] *
            # (self.A2 @ xstar))). Second term: 2*Re(xstar.conj() @
            # self.A2.T.conj() @ (self.Pdiags[:, i].conj() * self.s1))
            # self.Pdiags has shape (N_diag, N_projectors), A2_xstar has shape (N_diag,)
            # We want to multiply each column of Pdiags elementwise with A2_xstar.
            # However, we know that sum_i w_i A_ij v_i = sum_i (w_i * v_i) A_ij.
            # LHS is expression right below, RHS is below so we avoid dense
            # intermediate matrices.
            A2_xstar = self.A2 @ xstar  # Shape: (N_p,)
            # Shape: (N_p,) where N_p = self.A1.shape[1]
            x_conj_A1 = (xstar.conj() @ self.A1)  
            
            # term1 = -np.real((xstar.conj() @ self.A1) @ (self.Pdiags *
            # A2_xstar[:, np.newaxis]))  # Shape: (N_diag, N_projectors)
            term1 = -np.real((x_conj_A1 * A2_xstar) @ self.Pdiags)

            # term2 = 2 * np.real(A2_xstar.conj() @ (self.Pdiags.conj() *
            # self.s1[:, np.newaxis])) #. Same as above.
            term2 = 2 * np.real((A2_xstar.conj() * self.s1) @ self.Pdiags.conj())
            grad = term1 + term2

            if self.n_gen_constr > 0:
                gen_constr_grad = np.zeros(self.n_gen_constr)
                for i in range(self.n_gen_constr):
                    gen_constr_grad[i] = np.real(
                        -xstar.conj().T @ self.A2.conj().T @ self.B_j[i] @ 
                        self.A2 @ xstar
                        + 2 * xstar.conj().T @ self.A2.conj().T @ self.s_2j[i]
                        + self.c_2j[i]
                    )
                grad = np.concatenate((grad, gen_constr_grad))

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
                if self.n_gen_constr == 0:
                    Fv = np.zeros((penalty_matrix.shape[0], len(grad)), dtype=complex)
                    for j in range(penalty_matrix.shape[1]):
                        for k, Ak in enumerate(self.precomputed_As):
                            # yes this is a double for loop, hessian for fake sources
                            # is likely a speed bottleneck
                            Fv[:, k] = Ak @ A_inv_penalty[:, j]

                        grad_penalty += np.real(-A_inv_penalty[:, j].conj().T @ Fv)
                        hess_penalty += 2 * np.real(Fv.conj().T @ self._Acho_solve(Fv))
                else:
                    raise NotImplementedError(
                        "Hessian computation with general constraints not implemented."
                    )
                

            elif get_grad:
                grad = cast(FloatNDArray, grad)
                P = self.Pdiags.shape[1]
                G = self.n_gen_constr

                proj_grad_penalty = np.zeros(P)
                for j in range(penalty_matrix.shape[1]):
                    # slow: for i in range(len(grad)): grad_penalty[i] +=
                    # -np.real(A_inv_penalty[:, j].conj().T @ self.A1 @
                    # (self.Pdiags[:, i] * (self.A2 @ A_inv_penalty[:, j])))
                    # # for loop method
                    # fast: grad_penalty += -np.real((A_inv_penalty[:, j].conj().T
                    # @ self.A1) @ (self.Pdiags * (self.A2 @
                    # A_inv_penalty[:, j])[:, np.newaxis]))

                    # Same as above (fastest)
                    A_inv_penalty_j_A1 = A_inv_penalty[:, j].conj().T @ self.A1
                    A2_A_inv_penalty_j = self.A2 @ A_inv_penalty[:, j]  # (N_p,)
                    proj_grad_penalty += -np.real(
                        (A_inv_penalty_j_A1 * A2_A_inv_penalty_j) @ self.Pdiags
                    )

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
            ["dualval_real", "dualgrad_real", "dualval_penalty", "grad_penalty"],
        )
        dual_aux = DualAux(
            dualval_real=dualval,
            dualgrad_real=grad,
            dualval_penalty=dualval_penalty,
            grad_penalty=grad_penalty,
        )

        if len(penalty_vectors) > 0:
            return (
                dualval + dualval_penalty,
                grad + grad_penalty,
                hess + hess_penalty,
                dual_aux,
            )
        else:
            return dualval, grad, hess, dual_aux

    def solve_current_dual_problem(
        self,
        method: str,
        opt_params: Optional[dict[str, Any]] = None,
        init_lags: Optional[ArrayLike] = None,
    ) -> Tuple[float, ArrayLike, ArrayLike, Optional[ArrayLike], ArrayLike]:
        """
        Optimize the dual problem using 'newton' or 'bfgs'.

        Parameters
        ----------
        method : str
            'newton' (alternating Newton / GD) or 'bfgs'.
        opt_params : dict | None
            Override optimization parameters (see optimization.py defaults).
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

        OPT_PARAMS_DEFAULTS = {
            "opttol": 1e-2,
            "gradConverge": False,
            "min_inner_iter": 5,
            "max_restart": np.inf,
            "penalty_ratio": 1e-2,
            "penalty_reduction": 0.1,
            "break_iter_period": 20,
            "verbose": self.verbose - 1,
            "penalty_vector_list": [],
        }
        if opt_params is None:
            opt_params = {}
        opt_params = {
            **OPT_PARAMS_DEFAULTS,
            **opt_params,
        }  # override defaults with user specifications

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

        Raises
        ------
        NotImplementedError
            If general constraints are present.
        """
        if self.n_gen_constr > 0:
            raise NotImplementedError(
                "Merging constraints not implemented for general constraints."
            )
        gcd.merge_lead_constraints(self, merged_num=merged_num)

    def add_constraints(
        self, added_Pdiag_list: list[ComplexArray], orthonormalize: bool = True
    ) -> None:
        """
        Append additional projector constraints.

        Parameters
        ----------
        added_Pdiag_list : list[ComplexArray]
            List of new projector diagonals.
        orthonormalize : bool, default True
            Whether to orthonormalize constraint set after insertion.

        Raises
        ------
        NotImplementedError
            If general constraints are present.
        """
        if self.n_gen_constr > 0:
            raise NotImplementedError(
                "Adding constraints not implemented for QCQPs with general constraints."
            )
        gcd.add_constraints(
            self, added_Pdiag_list=added_Pdiag_list, orthonormalize=orthonormalize
        )

    def run_gcd(
        self,
        max_cstrt_num: int = 10,
        orthonormalize: bool = True,
        opt_params: Optional[dict[str, Any]] = None,
        max_gcd_iter_num: int = 50,
        gcd_iter_period: int = 5,
        gcd_tol: float = 1e-2,
    ) -> None:
        """Run GCD to approach tightest dual bound for this QCQP.
        
        See module-level run_gcd() for details. Modifies the existing QCQP object.

        Parameters
        ----------
        max_cstrt_num : int
            Maximum number of projector constraints to keep.
        orthonormalize : bool
            Whether to orthonormalize constraints during updates.
        opt_params : dict | None
            Optimization parameters for inner dual solves.
        max_gcd_iter_num : int
            Maximum outer GCD iterations.
        gcd_iter_period : int
            Period for GCD iterations.
        gcd_tol : float
            Tolerance for GCD convergence.
        """
        if self.n_gen_constr > 0:
            raise NotImplementedError(
                "GCD not implemented for QCQPs with general constraints."
            )
        gcd.run_gcd(
            self,
            max_cstrt_num=max_cstrt_num,
            orthonormalize=orthonormalize,
            opt_params=opt_params,
            max_gcd_iter_num=max_gcd_iter_num,
            gcd_iter_period=gcd_iter_period,
            gcd_tol=gcd_tol,
        )
