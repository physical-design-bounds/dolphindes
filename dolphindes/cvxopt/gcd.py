"""
Generalized Constraint Descent (GCD).

GCD tightens dual bounds for shared projection QCQPs by iteratively:
1. Adding new shared projection constraints likely to tighten the bound.
2. Merging older constraints to keep the total count small.

For usage examples see:
- examples/limits/LDOS.ipynb
- examples/verlan/LDOS_verlan.ipynb

Mathematical details: Appendix B of https://arxiv.org/abs/2504.10469
"""

from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from dolphindes.cvxopt._base_qcqp import _SharedProjQCQP
from dolphindes.cvxopt.optimization import OptimizationHyperparameters
from dolphindes.types import ComplexArray, FloatNDArray, SparseDense
from dolphindes.util import CRdot, Sym

OrthoMetric = Literal["euclidean", "hilbert_schmidt"]


def _frob_inner(A: Any, B: Any) -> float:
    """Real Frobenius inner product Re tr(A^H B) for dense or sparse A, B.

    For Hermitian A, B this equals Re tr(A B), the quantity that measures how
    parallel two constraint quadratic forms are.
    """
    if sp.issparse(A):
        val = A.conj().multiply(B).sum()
    else:
        val = np.vdot(np.asarray(A), np.asarray(B))
    return float(np.real(val))


def _hs_inner(
    ops_i: tuple[SparseDense, ComplexArray],
    ops_j: tuple[SparseDense, ComplexArray],
) -> float:
    """Augmented Hilbert-Schmidt inner product between two constraints.

    Each constraint is represented by its quadratic form ``A = Sym(A1 P A2)`` and
    linear form ``F = A2^H P^H s1``.
    The inner product is defined as

        <(A_i, F_i), (A_j, F_j)> = Re tr(A_i A_j) + Re(F_i^H F_j),

    a positive combination of the Hilbert-Schmidt inner product of the quadratic
    forms and the inner product of the linear forms. It is the Frobenius product
    of the homogenized constraint ``M = [[A, F], [F^H, 0]]`` up to the
    quadratic/linear weighting, which is treated as a fixed choice here: the A-F
    cross-blocks are off-diagonal and drop out of the trace, while the exact
    Frobenius product would weight the linear term by 2. It is real, symmetric and
    positive semidefinite, so it defines a valid metric on the real vector space
    of Lagrange multipliers; any positive weighting works, and the dual is
    preserved regardless of the choice.
    """
    A_i, F_i = ops_i
    A_j, F_j = ops_j
    return _frob_inner(A_i, A_j) + float(np.real(np.vdot(F_i, F_j)))


def _to_dense(M: SparseDense) -> ComplexArray:
    """Return ``M`` as a dense complex ndarray (no-op for dense input)."""
    dense = cast(Any, M).toarray() if sp.issparse(M) else M
    return np.asarray(dense, dtype=complex)


def _restrict(M: SparseDense, ix: "np.ndarray[Any, Any]") -> SparseDense:
    """Restrict ``M`` to the rows/cols in ``ix`` (dense or sparse), preserving type."""
    if sp.issparse(M):
        return cast(SparseDense, sp.csc_array(sp.csr_array(M)[ix, :])[:, ix])
    return cast(SparseDense, np.asarray(M)[np.ix_(ix, ix)])


def _hs_analytic_available(QCQP: _SharedProjQCQP) -> bool:
    """Whether the closed-form HS Gram / inner product applies to this QCQP.

    The reduction in :func:`_hs_kernels` expresses every HS inner product as a
    bilinear form in the projector *diagonals*, so it requires only that the
    projectors are diagonal. It applies to both formulations: the kernels are
    dense for a dense ``A1`` and sparse for a sparse ``A1`` (where ``A1, A2`` are
    themselves sparse, so their products stay sparse), and the projector
    diagonals contract against them either way.
    """
    return QCQP.Proj.is_diagonal()


def _hs_kernels(
    QCQP: _SharedProjQCQP,
) -> tuple[SparseDense, SparseDense, SparseDense]:
    """Build (and cache) the support-restricted HS kernels ``(L, K, N)``.

    For diagonal projectors ``P_j = diag(p_j)`` the augmented HS inner product is
    bilinear in the diagonals:

        <(A_i, F_i), (A_j, F_j)>
            = 1/2 Re(p_i^T L p_j) + 1/2 Re(p_i^H K p_j) + Re(p_i^T N conj(p_j))

    with kernels that depend only on ``(A1, A2, s1)``, not on the projector data:

        C  = A2 A1,                   L = C (.) C^T
        G1 = A1^H A1,  G2 = A2 A2^H,  K = G1 (.) G2^T
        N  = conj(diag s1) G2 diag(s1)

    where ``(.)`` is the elementwise (Hadamard) product. The ``1/2`` factors and
    the split into the ``L`` term (from ``tr(M_i M_j)``) and the ``K`` term (from
    ``tr(M_i^H M_j)``) both come from the symmetrization ``A_k = Sym(A1 P_k A2)``.

    The kernels are kept dense for a dense ``A1`` and sparse for a sparse ``A1``
    (there ``A1, A2`` and hence ``C, G1, G2`` are all sparse). Only the projector
    diagonals on ``Pstruct``'s support ever enter, so the kernels are restricted
    to those rows/cols (``Pstruct.indices``), matching the ``(nnz, k)`` layout of
    ``Projectors.get_Pdata_column_stack``. ``(A1, A2, s1, Pstruct)`` are fixed for
    the whole GCD run, so the result is cached on QCQP.
    """
    cached = getattr(QCQP, "_hs_kernel_cache", None)
    if cached is not None:
        return cast("tuple[SparseDense, SparseDense, SparseDense]", cached)

    s1 = np.asarray(QCQP.s1, dtype=complex)

    if sp.issparse(QCQP.A1):
        A1 = sp.csc_array(QCQP.A1)
        A2 = sp.csc_array(QCQP.A2)
        C = A2 @ A1
        G1 = A1.conj().T @ A1
        G2 = A2 @ A2.conj().T
        L: SparseDense = sp.csr_array(C.multiply(C.T))
        K: SparseDense = sp.csr_array(G1.multiply(G2.T))
        S = sp.diags_array(s1)
        N: SparseDense = sp.csr_array(S.conj() @ G2 @ S)
    else:
        A1d = _to_dense(QCQP.A1)
        A2d = _to_dense(QCQP.A2)
        C = A2d @ A1d
        G1 = A1d.conj().T @ A1d
        G2 = A2d @ A2d.conj().T
        L = C * C.T
        K = G1 * G2.T
        N = np.conj(s1)[:, None] * G2 * s1[None, :]

    ix = QCQP.Proj.Pstruct.indices
    kernels = (_restrict(L, ix), _restrict(K, ix), _restrict(N, ix))
    QCQP._hs_kernel_cache = kernels  # type: ignore[attr-defined]
    return kernels


def _constraint_ops(
    QCQP: _SharedProjQCQP, P: SparseDense
) -> tuple[SparseDense, ComplexArray]:
    """Return (Sym(A1 P A2), A2^H P^H s1) for a single projector matrix P."""
    A = Sym(QCQP.A1 @ P @ QCQP.A2)
    F = QCQP.A2.conj().T @ (P.conj().T @ QCQP.s1)
    return A, F


def _constraint_ops_for_data(
    QCQP: _SharedProjQCQP, Pdata: ComplexArray
) -> tuple[SparseDense, ComplexArray]:
    """Build the constraint operators for a projector given its Pstruct data."""
    P = QCQP.Proj.Pstruct.astype(complex, copy=True)
    P.data = np.asarray(Pdata, dtype=complex)
    return _constraint_ops(QCQP, P)


def _hs_gram_matrix(QCQP: _SharedProjQCQP) -> FloatNDArray:
    """Real (k, k) Gram matrix of the projector constraints in the HS metric.

    When the closed form applies (:func:`_hs_analytic_available`), the Gram
    matrix is computed analytically in ``O(n^2 k)`` via
    :func:`_hs_gram_matrix_analytic`.
    Otherwise this falls back to the ``O(k^2 n^2)`` Frobenius double loop over the
    cached ``precomputed_As`` (quadratic forms) and ``Fs`` (linear forms), so no
    operators are rebuilt. Only the projector constraints (the first
    ``n_proj_constr``) are included; general constraints are untouched.
    """
    if _hs_analytic_available(QCQP):
        return _hs_gram_matrix_analytic(QCQP)

    k = QCQP.n_proj_constr
    As = QCQP.precomputed_As
    Fs = QCQP.Fs
    G = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i, k):
            val = _frob_inner(As[i], As[j]) + float(
                np.real(np.vdot(Fs[:, i], Fs[:, j]))
            )
            G[i, j] = val
            G[j, i] = val
    return G


def _hs_gram_matrix_analytic(QCQP: _SharedProjQCQP) -> FloatNDArray:
    """Closed-form HS Gram matrix of the projector constraints.

    Replaces the ``O(k^2 n^2)`` Frobenius double loop of :func:`_hs_gram_matrix`
    with three tall-skinny BLAS products against the cached kernels
    (:func:`_hs_kernels`), i.e. ``O(n^2 k)`` and dominated by dense gemms. Only
    valid when :func:`_hs_analytic_available`.
    """
    L, K, N = _hs_kernels(QCQP)
    P = QCQP.Proj.get_Pdata_column_stack()  # (nnz, k)
    G = (
        0.5 * np.real(P.T @ (L @ P))
        + 0.5 * np.real(P.conj().T @ (K @ P))
        + np.real(P.T @ (N @ P.conj()))
    )
    # symmetrize to clear asymmetric rounding error (G is symmetric in exact math)
    return cast(FloatNDArray, 0.5 * (G + G.T))


def _orthonormalize_hs(QCQP: _SharedProjQCQP) -> None:
    """Re-orthonormalize all projector constraints in the HS metric.

    This is the Hilbert-Schmidt analogue of the real-extended QR step: we form
    the HS Gram matrix ``G`` of the current constraints and Cholesky-factor it as
    ``G = R^T R`` with ``R`` upper triangular. The new projector data
    ``Q = Pdata @ R^{-1}`` is HS-orthonormal, and since ``Pdata = Q R`` the dual
    value is preserved by mapping the multipliers ``lambda <- R @ lambda`` (the
    same bookkeeping the Euclidean QR path uses).
    """
    k = QCQP.n_proj_constr
    if k < 1:
        return
    assert QCQP.current_lags is not None
    G = _hs_gram_matrix(QCQP)
    try:
        R = la.cholesky(G, lower=False)
    except la.LinAlgError:
        # Constraints are (numerically) HS-dependent; add a tiny relative ridge
        # so we still obtain a valid triangular factor. The dual is preserved for
        # any invertible R, so this only perturbs the resulting basis slightly.
        ridge = 1e-12 * (np.trace(G) / k)
        R = la.cholesky(G + ridge * np.eye(k), lower=False)

    Pdata = QCQP.Proj.get_Pdata_column_stack()
    Rinv = la.solve_triangular(R, np.eye(k), lower=False)
    QCQP.Proj.set_Pdata_column_stack(Pdata @ Rinv)
    QCQP.current_lags[:k] = R @ QCQP.current_lags[:k]
    QCQP.compute_precomputed_values()


def _orthonormalize_new_hs_analytic(
    QCQP: _SharedProjQCQP, added_Pdata_list: list[ComplexArray]
) -> None:
    """Run modified Gram-Schmidt on new constraints in the HS metric.

    Operates purely on the projector diagonals via the cached kernels (no ``A_k``
    operators are built).
    Equivalent to :func:`_orthonormalize_new_hs` but ``O(n^2)`` per candidate
    (kernel matvecs) instead of the ``O(n^3)`` constraint-operator rebuilds.
    Assumes the existing projector constraints are already HS-orthonormal, as GCD
    maintains. Only valid when :func:`_hs_analytic_available`.
    """
    L, K, N = _hs_kernels(QCQP)

    def hs_inner(p: ComplexArray, q: ComplexArray) -> float:
        return float(
            0.5 * np.real(p @ (L @ q))
            + 0.5 * np.real(p.conj() @ (K @ q))
            + np.real(p @ (N @ q.conj()))
        )

    existing = QCQP.Proj.get_Pdata_column_stack()  # (nnz, n_proj_constr)
    basis = [existing[:, j] for j in range(existing.shape[1])]

    for m in range(len(added_Pdata_list)):
        a = np.array(added_Pdata_list[m], dtype=complex)
        init_norm_sq = hs_inner(a, a)
        for q in basis:
            a = a - hs_inner(q, a) * q
        norm_sq = hs_inner(a, a)
        if norm_sq > 1e-14 * max(init_norm_sq, 1.0):
            a = a / np.sqrt(norm_sq)
        # else: the constraint is (numerically) already in the span, so the
        # residual is an ~zero operator. Leaving it un-normalized keeps it inert
        # (it enters with a ~zero gradient and stays at multiplier 0).
        added_Pdata_list[m] = a
        basis.append(a.copy())


def _orthonormalize_new_hs(
    QCQP: _SharedProjQCQP, added_Pdata_list: list[ComplexArray]
) -> None:
    """Orthonormalize new projector data against the existing HS-orthonormal set."""
    if _hs_analytic_available(QCQP):
        _orthonormalize_new_hs_analytic(QCQP, added_Pdata_list)
        return

    proj_cstrt_num = QCQP.n_proj_constr
    existing_Pdata = QCQP.Proj.get_Pdata_column_stack()

    # (Pdata column, (A, F)) for each HS-orthonormal constraint accumulated so far
    basis = [
        (existing_Pdata[:, j], (QCQP.precomputed_As[j], QCQP.Fs[:, j]))
        for j in range(proj_cstrt_num)
    ]

    for m in range(len(added_Pdata_list)):
        a = added_Pdata_list[m]
        A_a, F_a = _constraint_ops_for_data(QCQP, a)
        init_norm_sq = _hs_inner((A_a, F_a), (A_a, F_a))
        for q_data, q_ops in basis:
            coef = _hs_inner(q_ops, (A_a, F_a))
            a -= coef * q_data
            A_a = A_a - coef * q_ops[0]
            F_a = F_a - coef * q_ops[1]
        norm_sq = _hs_inner((A_a, F_a), (A_a, F_a))
        if norm_sq > 1e-14 * max(init_norm_sq, 1.0):
            scale = np.sqrt(norm_sq)
            a /= scale
            A_a = A_a / scale
            F_a = F_a / scale
        # else: the constraint is (numerically) already in the span, so the
        # residual is an ~zero operator. Leaving it un-normalized keeps it inert
        # (it enters with a ~zero gradient and stays at multiplier 0).
        basis.append((a.copy(), (A_a, F_a)))


@dataclass(frozen=True)
class GCDHyperparameters:
    """Hyperparameters for GCD algorithm.

    Attributes
    ----------
    max_proj_cstrt_num : int
        Maximum number of projector constraints to keep during GCD.
    orthonormalize : bool
        Whether to keep projector constraints orthonormalized.
    ortho_metric : str
        Inner product used for orthonormalization. Either ``"euclidean"`` (the
        default: orthonormalize the raw projector data ``vec(P_j)``) or
        ``"hilbert_schmidt"`` (orthonormalize the constraints themselves, in the
        augmented Hilbert-Schmidt metric that reflects how each constraint enters
        the dual). See :func:`_hs_gram_matrix`. Ignored if ``orthonormalize`` is
        False.
    opt_params : OptimizationHyperparameters | None
        Optimization hyperparameters used for the internal dual solve at each GCD
        iteration. If None, GCD uses defaults suitable for frequent re-solves
        (notably `max_restart=1`).
    max_gcd_iter_num : int
        Maximum number of GCD iterations.
    gcd_iter_period : int
        Period for checking GCD convergence.
    gcd_tol : float
        Relative tolerance for GCD convergence.
    """

    max_proj_cstrt_num: int = 10
    orthonormalize: bool = True
    ortho_metric: OrthoMetric = "hilbert_schmidt"
    opt_params: OptimizationHyperparameters | None = None
    max_gcd_iter_num: int = 50
    gcd_iter_period: int = 5
    gcd_tol: float = 1e-2

    def __post_init__(self) -> None:
        """Validate that ortho_metric is one of the supported metrics."""
        valid = get_args(OrthoMetric)
        if self.ortho_metric not in valid:
            raise ValueError(
                f"ortho_metric must be one of {valid}, got {self.ortho_metric!r}."
            )


def merge_lead_constraints(
    QCQP: _SharedProjQCQP, merged_num: int = 2, metric: OrthoMetric = "hilbert_schmidt"
) -> None:
    """
    Merge the first m shared projection constraints of QCQP into a single one.

    Also, adjust the Lagrange multipliers so the dual value is the same.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        QCQP for which we merge the leading constraints.
    merged_num : int (optional, default 2)
        Number of leading constraints that we are merging together; must be at least 2.
    metric : str, optional
        Metric used to normalize the merged constraint. ``"euclidean"`` (default)
        normalizes by the projector-data norm; ``"hilbert_schmidt"`` normalizes by
        the constraint's HS norm, which keeps the constraint set HS-orthonormal
        when GCD runs with ``ortho_metric="hilbert_schmidt"``. The choice does not
        affect the dual value (the multiplier is rescaled to compensate).

    Raises
    ------
    ValueError
        If merged_num < 2 or if there are insufficient constraints for merging.
    """
    proj_cstrt_num = len(QCQP.Proj)
    if merged_num < 2:
        raise ValueError("Need at least 2 constraints for merging.")
    if proj_cstrt_num < merged_num:
        raise ValueError("Number of constraints insufficient for size of merge.")

    if QCQP.current_lags is None:
        raise ValueError("Cannot merge constraints: QCQP.current_lags is None.")

    new_P = QCQP.Proj.Pstruct.astype(complex, copy=True)
    new_P.data[:] = 0.0
    for i in range(merged_num):
        # keep in mind the sharedProj multipliers come first in current_lags
        new_P += QCQP.current_lags[i] * QCQP.Proj[i]

    # Normalize the merged constraint. Any nonzero factor preserves the dual (the
    # multiplier below is set to compensate); the HS norm additionally keeps the
    # constraint set HS-orthonormal for ortho_metric="hilbert_schmidt".
    if metric == "hilbert_schmidt":
        merged_ops = _constraint_ops(QCQP, new_P)
        Pnorm = np.sqrt(_hs_inner(merged_ops, merged_ops))
    else:
        Pnorm = la.norm(new_P.data)
    new_P /= Pnorm

    QCQP.Proj[merged_num - 1] = new_P
    QCQP.Proj.erase_leading(merged_num - 1)

    # update QCQP
    if hasattr(QCQP, "precomputed_As"):
        # updated precomputed_As
        QCQP.precomputed_As[merged_num - 1] *= QCQP.current_lags[merged_num - 1]
        for i in range(merged_num - 1):
            QCQP.precomputed_As[merged_num - 1] += (
                QCQP.precomputed_As[i] * QCQP.current_lags[i]
            )
        QCQP.precomputed_As[merged_num - 1] /= Pnorm
        del QCQP.precomputed_As[: merged_num - 1]

    if hasattr(QCQP, "Fs"):
        QCQP.Fs = QCQP.Fs[:, merged_num - 1 :]
        QCQP.Fs[:, 0] = QCQP.A2.conj().T @ (new_P.conj().T @ QCQP.s1)

    QCQP.current_lags = QCQP.current_lags[merged_num - 1 :]
    QCQP.current_lags[0] = Pnorm
    QCQP.n_proj_constr = len(QCQP.Proj)

    QCQP.current_grad = QCQP.current_hess = None

    # precomputed_As/Fs were edited in place, so cached factorizations are stale.
    QCQP._invalidate_factor_cache()


def add_constraints(
    QCQP: _SharedProjQCQP,
    added_Pdata_list: list[ComplexArray],
    orthonormalize: bool = True,
    metric: OrthoMetric = "euclidean",
) -> None:
    """
    Add new shared projection constraints into an existing QCQP.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        QCQP for which the new constraints are added in.
    added_Pdata_list : list
        List of 1d numpy arrays representing the sparse entries of
        the new constraints to be added in, with sparsity structure QCQP.Proj.Pstruct
    orthonormalize : bool, optional
        If true, assume that QCQP has orthonormal constraints and keeps it that way
    metric : str, optional
        Metric for orthonormalization: ``"euclidean"`` (default) uses the
        projector-data inner product; ``"hilbert_schmidt"`` uses the augmented
        HS metric on the constraints. Only used when ``orthonormalize`` is True.
    """
    x_size = QCQP.Proj.Pstruct.size
    proj_cstrt_num = QCQP.n_proj_constr
    added_Pdata_num = len(added_Pdata_list)

    if QCQP.current_lags is not None:
        new_lags = np.zeros(
            proj_cstrt_num + added_Pdata_num + QCQP.n_gen_constr, dtype=float
        )
        new_lags[:proj_cstrt_num] = QCQP.current_lags[:proj_cstrt_num]
        new_lags[proj_cstrt_num + added_Pdata_num :] = QCQP.current_lags[
            proj_cstrt_num:
        ]
        QCQP.current_lags = new_lags

    if orthonormalize and metric == "hilbert_schmidt":
        _orthonormalize_new_hs(QCQP, added_Pdata_list)
    elif orthonormalize:
        # in this case assume that existing Pdata is already orthonormalized
        new_Pdata = np.zeros((x_size, proj_cstrt_num + added_Pdata_num), dtype=complex)
        new_Pdata[:, :proj_cstrt_num] = QCQP.Proj.get_Pdata_column_stack()

        for m in range(added_Pdata_num):
            # do (modified) Gram-Schmidt orthogonalization for each added Pdata
            for j in range(proj_cstrt_num + m):
                added_Pdata_list[m] -= (
                    CRdot(new_Pdata[:, j], added_Pdata_list[m]) * new_Pdata[:, j]
                )
            added_Pdata_list[m] /= la.norm(added_Pdata_list[m])

            new_Pdata[:, proj_cstrt_num + m] = added_Pdata_list[m]

    # update QCQP
    for m, added_Pdata in enumerate(added_Pdata_list):
        Pnew = QCQP.Proj.Pstruct.astype(complex, copy=True)
        Pnew.data = added_Pdata
        QCQP.Proj.append(Pnew)

        if hasattr(QCQP, "precomputed_As"):
            # updated precomputed_As
            QCQP.precomputed_As.insert(
                proj_cstrt_num + m, Sym(QCQP.A1 @ Pnew @ QCQP.A2)
            )

    if hasattr(QCQP, "Fs"):
        new_Fs = np.zeros(
            (QCQP.Fs.shape[0], len(QCQP.Proj) + QCQP.n_gen_constr), dtype=complex
        )
        new_Fs[:, : len(QCQP.Proj)] = QCQP.A2.conj().T @ QCQP.Proj.allP_at_v(
            QCQP.s1, dagger=True
        )
        new_Fs[:, len(QCQP.Proj) :] = QCQP.Fs[:, proj_cstrt_num:]
        QCQP.Fs = new_Fs

    QCQP.n_proj_constr = len(QCQP.Proj)
    QCQP.current_grad = QCQP.current_hess = None

    # precomputed_As/Fs were edited in place, so cached factorizations are stale.
    QCQP._invalidate_factor_cache()


def run_gcd(
    QCQP: _SharedProjQCQP,
    gcd_params: GCDHyperparameters = GCDHyperparameters(),
) -> None:
    """
    Perform generalized constraint descent to gradually refine dual bound on QCQP.

    At each GCD iteration, add two new constraints:
    1.a constraint generated so the corresponding dual derivative is large,
    to hopefully tighten the dual bound
    2. a constraint generated so the corresponding derivative of the smallest
    Lagrangian quadratic form eigenvalue is large, to help the dual optimization
    navigate the semi-definite boundary

    If the total number of constraints is larger than max_gcd_proj_cstrt_num combine
    the earlier constraints to keep the total number of constraints fixed. Setting
    max_proj_cstrt_num large enough will eventually result in evaluating the dual bound
    with all possible constraints, which gives the tightest bound but may be extremely
    expensive. The goal of GCD is to approximate this tightest bound with greatly
    reduced computational cost.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        The SharedProjQCQP for which we compute and refine dual bounds.
    max_proj_cstrt_num : int, optional
        The maximum projection constraint number for QCQP. The default is 10.
    orthonormalize : bool, optional
        Whether or not to orthonormalize the constraint projectors. The default is True.
    opt_params : OptimizationHyperparameters, optional
        Optimization hyperparameters for the internal dual solve at every GCD
        iteration.
    max_gcd_iter_num : int, optional
        Maximum number of GCD iterations, by default 50.
    gcd_iter_period : int, optional
        Period for checking convergence, by default 5.
    gcd_tol : float, optional
        Tolerance for convergence, by default 1e-2.

    Notes
    -----
    TODO: formalize optimization and convergence parameters.
    """
    # Since GCD constantly changes constraints, there is typically little value in
    # running multiple outer penalty-reduction restarts for each intermediate solve.
    # Default to a single outer iteration (max_restart=1).
    if gcd_params.opt_params is None:
        opt_params = OptimizationHyperparameters(
            opttol=1e-2,
            gradConverge=False,
            min_inner_iter=5,
            max_restart=1,
            penalty_ratio=1e-2,
            penalty_reduction=0.1,
            break_iter_period=20,
            verbose=int(QCQP.verbose - 1),
        )
    else:
        opt_params = gcd_params.opt_params

    # get to feasible point
    # TODO: revamp find_feasible_lags
    QCQP.current_lags = QCQP.find_feasible_lags()
    assert QCQP.current_lags is not None

    orthonormalize = gcd_params.orthonormalize
    ortho_metric = gcd_params.ortho_metric
    max_proj_cstrt_num = gcd_params.max_proj_cstrt_num
    max_gcd_iter_num = gcd_params.max_gcd_iter_num
    gcd_iter_period = gcd_params.gcd_iter_period
    gcd_tol = gcd_params.gcd_tol

    if orthonormalize and ortho_metric == "hilbert_schmidt":
        _orthonormalize_hs(QCQP)
    elif orthonormalize:
        # orthonormalize QCQP
        # informally checked for correctness
        x_size = QCQP.Proj.Pstruct.size
        proj_cstrt_num = QCQP.n_proj_constr
        Pdata = QCQP.Proj.get_Pdata_column_stack()
        realext_Pdata = np.zeros((2 * x_size, proj_cstrt_num), dtype=float)
        realext_Pdata[:x_size, :] = np.real(Pdata)
        realext_Pdata[x_size:, :] = np.imag(Pdata)
        realext_Pdata_Q, realext_Pdata_R = la.qr(realext_Pdata, mode="economic")

        QCQP.Proj.set_Pdata_column_stack(
            realext_Pdata_Q[:x_size, :] + 1j * realext_Pdata_Q[x_size:, :]
        )
        QCQP.current_lags[: QCQP.n_proj_constr] = (
            realext_Pdata_R @ QCQP.current_lags[: QCQP.n_proj_constr]
        )
        QCQP.compute_precomputed_values()

    ## gcd loop
    gcd_iter_num = 0
    gcd_prev_dual: float = np.inf
    while True:
        gcd_iter_num += 1
        # solve current dual problem
        # print('at gcd iter num', gcd_iter_num)
        # print('QCQP.current_lags', QCQP.current_lags)
        # print('QCQP.Fs.shape', QCQP.Fs.shape)
        QCQP.solve_current_dual_problem(
            "newton", init_lags=QCQP.current_lags, opt_params=opt_params
        )
        assert QCQP.current_dual is not None
        assert QCQP.current_xstar is not None

        print(
            f"At GCD iteration #{gcd_iter_num}, best dual bound found is \
            {QCQP.current_dual}."
        )

        ## termination conditions
        if gcd_iter_num > max_gcd_iter_num:
            break
        if gcd_iter_num % gcd_iter_period == 0:
            if gcd_prev_dual - QCQP.current_dual < gcd_tol * abs(gcd_prev_dual):
                break
            gcd_prev_dual = QCQP.current_dual

        ## generate new constraints
        new_Pdata_list = []
        Pstruct_rows, Pstruct_cols = QCQP.Proj.Pstruct.nonzero()
        ## generate max dualgrad constraint
        maxViol_Pdiag = (2 * QCQP.s1 - (QCQP.A1.conj().T @ QCQP.current_xstar))[
            Pstruct_rows
        ] * (QCQP.A2 @ QCQP.current_xstar).conj()[Pstruct_cols]

        if la.norm(maxViol_Pdiag) >= 1e-14:
            new_Pdata_list.append(maxViol_Pdiag)
            # skip this new constraint if maxViol_Pdiag is uniformly 0
            # can happen if there are no linear forms in objective and all constraints

        ## generate min A eig constraint
        minAeigv, minAeigw = QCQP._get_PSD_penalty(QCQP.current_lags)
        minAeig_Pdiag = (QCQP.A1.conj().T @ minAeigv)[Pstruct_rows] * (
            QCQP.A2 @ minAeigv
        ).conj()[Pstruct_cols]

        minAeig_Pdiag /= np.sqrt(np.real(minAeig_Pdiag.conj() * minAeig_Pdiag))
        # minAeig_Pdiag * np.sqrt(np.real(maxViol_Pdiag.conj() * maxViol_Pdiag))
        # use the same relative weights for minAeig_Pdiag as maxViol_Pdiag
        # informally checked that minAeigw increases when increasing multiplier of
        # minAeig_Pdiag
        new_Pdata_list.append(minAeig_Pdiag)

        ## add new constraints
        QCQP.add_constraints(
            new_Pdata_list, orthonormalize=orthonormalize, metric=ortho_metric
        )
        # informally checked that new constraints are added in orthonormal fashion

        ## merge old constraints if necessary
        if len(QCQP.Proj) > max_proj_cstrt_num:
            QCQP.merge_lead_constraints(
                merged_num=len(QCQP.Proj) - max_proj_cstrt_num + 1,
                metric=ortho_metric,
            )
