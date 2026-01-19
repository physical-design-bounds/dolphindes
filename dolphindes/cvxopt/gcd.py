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
from typing import Sequence

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

from dolphindes.cvxopt._base_qcqp import _SharedProjQCQP
from dolphindes.cvxopt.optimization import OptimizationHyperparameters
from dolphindes.types import ComplexArray
from dolphindes.util import CRdot, Projectors, Sym


@dataclass(frozen=True)
class GCDHyperparameters:
    """Hyperparameters for GCD algorithm.

    Attributes
    ----------
    max_proj_cstrt_num : int
        Maximum number of projector constraints to keep during GCD.
    orthonormalize : bool
        Whether to keep projector constraints orthonormalized.
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
    protected_proj_indices : Sequence[int] | None
        Projector constraint indices that GCD will not merge/compress.

        Notes on semantics:
        - Indices refer to the current projector constraint ordering at the
            start of GCD.
        - Protection applies to GCD merging/compression only; GCD may still
            append new (unprotected) constraints.
    """

    max_proj_cstrt_num: int = 10
    orthonormalize: bool = True
    opt_params: OptimizationHyperparameters | None = None
    max_gcd_iter_num: int = 50
    gcd_iter_period: int = 5
    gcd_tol: float = 1e-2
    verbose: int = 1
    protected_proj_indices: Sequence[int] | None = None


def _normalize_unique_indices(indices: Sequence[int], n: int) -> list[int]:
    """Make indices for protected_proj_indices valid.

    Normalize Python-style negative indices: e.g. -1 becomes n-1, etc.
    This lets users write for example protected_proj_indices=[-1]
    to protect the last constraint.

    Notes
    -----
    This works in two steps:
    1. Checks 0 <= j < n.
       This matters because Projectors.__getitem__ internally uses idx = key % self._k,
       so out-of-range indices would otherwise wrap around and accidentally “protect”
       the wrong constraint.
    2. De-duplicate and sort.
    """
    out: set[int] = set()
    for idx in indices:
        j = int(idx)
        if j < 0:
            j = n + j
        if j < 0 or j >= n:
            raise IndexError(f"Protected projector index {idx} out of range for {n}.")
        out.add(j)
    return sorted(out)


def _get_gcd_protected_mask(
    QCQP: _SharedProjQCQP, indices: Sequence[int] | None
) -> NDArray[np.bool_]:
    """Convert the indices of protected projector constraints into a boolean mask."""
    n = QCQP.n_proj_constr
    mask: NDArray[np.bool_] = np.zeros(n, dtype=bool)
    if indices is not None:
        for j in _normalize_unique_indices(indices, n):
            mask[j] = True
    setattr(QCQP, "_gcd_protected_proj_mask", mask)
    return mask


def _assert_protected_projectors_orthogonal(
    QCQP: _SharedProjQCQP,
    protected_mask: NDArray[np.bool_],
    tol: float = 1e-10,
) -> None:
    """Assert protected projectors are pairwise orthogonal.

    Orthogonality is measured using the real-field inner product `CRdot` on the
    complex Pdata vectors. The check is scale-invariant: we test normalized
    correlations so protected projectors need not be unit-normalized.
    """
    if not np.any(protected_mask):
        return

    proj_cstrt_num = QCQP.n_proj_constr
    if protected_mask.shape != (proj_cstrt_num,):
        raise ValueError("protected_mask length mismatch with n_proj_constr.")

    protected_idx = np.where(protected_mask)[0]
    if protected_idx.size <= 1:
        return

    Pdata = QCQP.Proj.get_Pdata_column_stack()

    max_corr = 0.0
    for a_i, a in enumerate(protected_idx):
        P1 = Pdata[:, a]
        norm1 = CRdot(P1, P1)
        assert norm1 > 0.0, "Protected projector has zero real-extended norm."

        for b in protected_idx[a_i + 1 :]:
            P2 = Pdata[:, b]
            norm2 = CRdot(P2, P2)
            assert norm2 > 0.0, "Protected projector has zero real-extended norm."
            denom = float(np.sqrt(norm1 * norm2))
            corr = abs(CRdot(P1, P2)) / denom
            if corr > max_corr:
                max_corr = corr

    if max_corr > tol:
        raise ValueError(
            "Protected projectors must be pairwise orthogonal. "
            f"Max normalized correlation is {max_corr:.3e} (tol={tol:.1e})."
        )


def _freeze_protected_orthonormalize_others(
    QCQP: _SharedProjQCQP,
    protected_mask: NDArray[np.bool_],
) -> None:
    """Orthonormalize unprotected constraints w.r.t. all constraints.

    Updates QCQP.current_lags projector part so that the weighted projector sum
    is preserved under the real-field orthonormalization transform.
    """
    if QCQP.current_lags is None:
        raise ValueError("QCQP.current_lags is None; cannot orthonormalize.")

    x_size = QCQP.Proj.Pstruct.size
    proj_cstrt_num = QCQP.n_proj_constr
    if protected_mask.shape != (proj_cstrt_num,):
        raise ValueError("protected_mask length mismatch with n_proj_constr.")

    protected_idx = np.where(protected_mask)[0]
    unprotected_idx = np.where(~protected_mask)[0]
    if unprotected_idx.size == 0:
        return

    # Column-stacked sparse entries (nnz, k)
    Pdata = QCQP.Proj.get_Pdata_column_stack()
    real_proj_data = np.zeros((2 * x_size, proj_cstrt_num), dtype=float)
    real_proj_data[:x_size, :] = np.real(Pdata)
    real_proj_data[x_size:, :] = np.imag(Pdata)

    A_unprotected = real_proj_data[:, unprotected_idx]
    proj_multipliers = np.array(QCQP.current_lags[:proj_cstrt_num], dtype=float)
    proj_multipliers_unprotected = proj_multipliers[unprotected_idx]

    if protected_idx.size > 0:
        # By construction we require protected columns to be mutually orthogonal.
        Ap = real_proj_data[:, protected_idx]
        gp = np.sum(Ap * Ap, axis=0)

        # Remove protected component from unprotected block:
        # Au = Ap D + B, where D = (Ap^T Au) / diag(Ap^T Ap) since Ap columns are
        # orthogonal (not necessarily normalized).
        D = (Ap.T @ A_unprotected) / gp[:, None]
        B = A_unprotected - Ap @ D

        # Update protected multipliers so that Σ λ P is preserved.
        proj_multipliers_protected = proj_multipliers[protected_idx] + (
            D @ proj_multipliers_unprotected
        )
    else:
        B = A_unprotected
        proj_multipliers_protected = np.zeros(0, dtype=float)

    Qu, Ru = la.qr(B, mode="economic")
    proj_multipliers_unprotected_new = Ru @ proj_multipliers_unprotected

    # Write back updated projectors (protected columns unchanged).
    new_real_proj_data = real_proj_data.copy()
    new_real_proj_data[:, unprotected_idx] = Qu
    new_Pdata = new_real_proj_data[:x_size, :] + 1j * new_real_proj_data[x_size:, :]
    QCQP.Proj.set_Pdata_column_stack(new_Pdata)

    # Write back updated lags in original order.
    proj_multipliers_new = proj_multipliers.copy()
    if protected_idx.size > 0:
        proj_multipliers_new[protected_idx] = proj_multipliers_protected
    proj_multipliers_new[unprotected_idx] = proj_multipliers_unprotected_new
    QCQP.current_lags[:proj_cstrt_num] = proj_multipliers_new

    QCQP.compute_precomputed_values()
    QCQP.current_grad = QCQP.current_hess = None
    QCQP.current_dual = None
    QCQP.current_xstar = None


def _merge_selected_projector_constraints(
    QCQP: _SharedProjQCQP, merge_indices: Sequence[int]
) -> None:
    """Merge an arbitrary set of projector constraints into one.

    The merged projector is normalized and placed at the largest index among
    merge_indices; all other merged constraints are removed. General constraints
    remain unchanged.
    """
    if QCQP.current_lags is None:
        raise ValueError("Cannot merge constraints: QCQP.current_lags is None.")

    k = QCQP.n_proj_constr
    if k == 0:
        raise ValueError("No projector constraints to merge.")

    merge_idx = _normalize_unique_indices(list(merge_indices), k)
    if len(merge_idx) < 2:
        raise ValueError("Need at least 2 constraints to merge.")

    protected_mask = getattr(QCQP, "_gcd_protected_proj_mask", None)
    if protected_mask is None:
        protected_mask = np.zeros(k, dtype=bool)
    if bool(np.any(protected_mask[merge_idx])):
        raise ValueError("Attempted to merge a protected constraint.")

    # Build merged column in Pdata space.
    Pdata = QCQP.Proj.get_Pdata_column_stack()  # (nnz, k)
    proj_lags = np.array(QCQP.current_lags[:k], dtype=float)
    merged_col = np.zeros((Pdata.shape[0],), dtype=complex)
    for j in merge_idx:
        merged_col += proj_lags[j] * Pdata[:, j]

    Pnorm = la.norm(merged_col)
    if Pnorm <= 0:
        raise ValueError("Cannot merge constraints: merged projector has zero norm.")
    merged_col /= Pnorm

    keep = max(merge_idx)
    keep_set = set(merge_idx)

    new_cols: list[ComplexArray] = []
    new_proj_lags: list[float] = []
    new_mask: list[bool] = []

    for j in range(k):
        if j in keep_set:
            if j == keep:
                new_cols.append(merged_col)
                new_proj_lags.append(float(Pnorm))
                new_mask.append(False)
        else:
            new_cols.append(Pdata[:, j])
            new_proj_lags.append(float(proj_lags[j]))
            new_mask.append(bool(protected_mask[j]))

    # Rebuild Projectors from scratch
    # TODO: More efficient way to do this?
    Plist = []
    for col in new_cols:
        Pnew = QCQP.Proj.Pstruct.astype(complex, copy=True)
        Pnew.data = np.asarray(col, dtype=complex)
        Plist.append(Pnew)
    QCQP.Proj = Projectors(Plist, QCQP.Proj.Pstruct)
    QCQP.n_proj_constr = len(QCQP.Proj)

    # Update lags: [proj lags..., general lags...]
    gen_lags = (
        np.array(QCQP.current_lags[k:], dtype=float)
        if QCQP.n_gen_constr > 0
        else np.array([], dtype=float)
    )
    QCQP.current_lags = np.concatenate([np.array(new_proj_lags, float), gen_lags])

    # Update protected mask
    setattr(QCQP, "_gcd_protected_proj_mask", np.array(new_mask, dtype=bool))

    # Recompute caches (simpler + robust)
    QCQP.compute_precomputed_values()
    QCQP.current_grad = QCQP.current_hess = None
    QCQP.current_dual = None
    QCQP.current_xstar = None


def merge_lead_constraints(QCQP: _SharedProjQCQP, merged_num: int = 2) -> None:
    """
    Merge the first m shared projection constraints of QCQP into a single one.

    Also, adjust the Lagrange multipliers so the dual value is the same.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        QCQP for which we merge the leading constraints.
    merged_num : int (optional, default 2)
        Number of leading constraints that we are merging together; must be at least 2.

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


def add_constraints(
    QCQP: _SharedProjQCQP,
    added_Pdata_list: list[ComplexArray],
    orthonormalize: bool = True,
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
    """
    x_size = QCQP.Proj.Pstruct.size
    proj_cstrt_num = QCQP.n_proj_constr
    added_Pdata_num = len(added_Pdata_list)

    if orthonormalize:
        # If a protected mask exists, do not assume the full existing set is
        # orthonormal; only assume the unprotected subset is kept orthonormal
        # w.r.t. the protected span and each other
        existing_Pdata = QCQP.Proj.get_Pdata_column_stack()  # (nnz, k)
        protected_mask = getattr(QCQP, "_gcd_protected_proj_mask", None)

        if protected_mask is not None and np.any(protected_mask):
            protected_idx = np.where(protected_mask[:proj_cstrt_num])[0]
            unprotected_idx = np.where(~protected_mask[:proj_cstrt_num])[0]

            # Protected projectors are assumed mutually orthogonal (checked at start
            # of GCD) so we can project out protected span without QR.
            if protected_idx.size > 0:
                Ap_c = existing_Pdata[:, protected_idx]
                Ap_re = np.zeros((2 * x_size, Ap_c.shape[1]), dtype=float)
                Ap_re[:x_size, :] = np.real(Ap_c)
                Ap_re[x_size:, :] = np.imag(Ap_c)
                gp = np.sum(Ap_re * Ap_re, axis=0)
                if bool(np.any(gp <= 0.0)):
                    raise ValueError("Protected projector has zero real-extended norm.")
            else:
                Ap_re = None
                gp = None

            unprot_basis = existing_Pdata[:, unprotected_idx]
            new_cols: list[ComplexArray] = []
            filtered_added: list[ComplexArray] = []
            for m in range(added_Pdata_num):
                v0 = np.array(added_Pdata_list[m], dtype=complex, copy=True)
                v = v0
                if Ap_re is not None and gp is not None:
                    realext_v = np.concatenate([np.real(v), np.imag(v)])
                    coeff = (Ap_re.T @ realext_v) / gp
                    realext_v = realext_v - Ap_re @ coeff
                    v = realext_v[:x_size] + 1j * realext_v[x_size:]

                # Modified Gram-Schmidt against existing unprotected + previously added
                for j in range(unprot_basis.shape[1]):
                    v = v - CRdot(unprot_basis[:, j], v) * unprot_basis[:, j]
                for col in new_cols:
                    v = v - CRdot(col, v) * col

                vnorm = la.norm(v)
                assert vnorm >= 1e-16, "Numerical issue in add_constraints."
                v = v / vnorm
                filtered_added.append(v)
                new_cols.append(v)

            added_Pdata_list = filtered_added
        else:
            # Legacy path assume existing constraints are orthonormal.
            # Faster when there are no protected constraints
            new_Pdata = np.zeros(
                (x_size, proj_cstrt_num + added_Pdata_num), dtype=complex
            )
            new_Pdata[:, :proj_cstrt_num] = existing_Pdata

            for m in range(added_Pdata_num):
                for j in range(proj_cstrt_num + m):
                    added_Pdata_list[m] -= (
                        CRdot(new_Pdata[:, j], added_Pdata_list[m]) * new_Pdata[:, j]
                    )
                added_Pdata_list[m] /= la.norm(added_Pdata_list[m])
                new_Pdata[:, proj_cstrt_num + m] = added_Pdata_list[m]

    # Expand lags after possible skipping of near-zero vectors above.
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

    # Keep protected mask aligned with projector indices.
    protected_mask = getattr(QCQP, "_gcd_protected_proj_mask", None)
    if protected_mask is not None:
        setattr(
            QCQP,
            "_gcd_protected_proj_mask",
            np.concatenate([protected_mask, np.zeros(added_Pdata_num, dtype=bool)]),
        )


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
    max_proj_cstrt_num = gcd_params.max_proj_cstrt_num
    max_gcd_iter_num = gcd_params.max_gcd_iter_num
    gcd_iter_period = gcd_params.gcd_iter_period
    gcd_tol = gcd_params.gcd_tol

    protected_mask = _get_gcd_protected_mask(QCQP, gcd_params.protected_proj_indices)
    if orthonormalize and np.any(protected_mask):
        _assert_protected_projectors_orthogonal(QCQP, protected_mask)

    if orthonormalize:
        if np.any(protected_mask):
            _freeze_protected_orthonormalize_others(QCQP, protected_mask)
        else:
            # Original behavior: full orthonormalization of all constraints.
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
        QCQP.add_constraints(new_Pdata_list, orthonormalize=orthonormalize)
        # informally checked that new constraints are added in orthonormal fashion

        # merge old constraints if necessary
        if len(QCQP.Proj) > max_proj_cstrt_num:
            excess = len(QCQP.Proj) - max_proj_cstrt_num
            merge_count = excess + 1

            protected_mask_any = getattr(QCQP, "_gcd_protected_proj_mask", None)
            if protected_mask_any is None:
                protected_mask = np.zeros(QCQP.n_proj_constr, dtype=bool)
            else:
                protected_mask = np.asarray(protected_mask_any, dtype=bool)

            merge_candidates: list[int] = []
            for j in range(QCQP.n_proj_constr):
                if not bool(protected_mask[j]):
                    merge_candidates.append(j)
                    if len(merge_candidates) >= merge_count:
                        break

            if len(merge_candidates) < merge_count:
                raise ValueError(
                    "Cannot compress to max_proj_cstrt_num without merging protected "
                    "constraints. Increase max_proj_cstrt_num or reduce "
                    "protected_proj_indices."
                )

            _merge_selected_projector_constraints(QCQP, merge_candidates)

            # Maintain the orthonormalization invariant after merging.
            if orthonormalize:
                protected_mask_any = getattr(QCQP, "_gcd_protected_proj_mask", None)
                if protected_mask_any is not None and np.any(protected_mask_any):
                    _freeze_protected_orthonormalize_others(
                        QCQP, np.asarray(protected_mask_any, dtype=bool)
                    )
                else:
                    x_size = QCQP.Proj.Pstruct.size
                    proj_cstrt_num = QCQP.n_proj_constr
                    Pdata = QCQP.Proj.get_Pdata_column_stack()
                    realext_Pdata = np.zeros((2 * x_size, proj_cstrt_num), dtype=float)
                    realext_Pdata[:x_size, :] = np.real(Pdata)
                    realext_Pdata[x_size:, :] = np.imag(Pdata)
                    realext_Pdata_Q, realext_Pdata_R = la.qr(
                        realext_Pdata, mode="economic"
                    )

                    QCQP.Proj.set_Pdata_column_stack(
                        realext_Pdata_Q[:x_size, :] + 1j * realext_Pdata_Q[x_size:, :]
                    )
                    assert QCQP.current_lags is not None
                    QCQP.current_lags[: QCQP.n_proj_constr] = (
                        realext_Pdata_R @ QCQP.current_lags[: QCQP.n_proj_constr]
                    )
                    QCQP.compute_precomputed_values()
