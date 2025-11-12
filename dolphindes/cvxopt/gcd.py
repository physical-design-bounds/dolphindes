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
from typing import Any

import numpy as np
import scipy.linalg as la

from dolphindes.cvxopt._base_qcqp import _SharedProjQCQP
from dolphindes.cvxopt.optimization import OptimizationHyperparameters
from dolphindes.types import ComplexArray
from dolphindes.util import CRdot, Sym


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
    """

    max_proj_cstrt_num: int = 10
    orthonormalize: bool = True
    opt_params: OptimizationHyperparameters | None = None
    max_gcd_iter_num: int = 50
    gcd_iter_period: int = 5
    gcd_tol: float = 1e-2
    verbose: int = 1


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

    if QCQP.current_lags is not None:
        new_lags = np.zeros(
            proj_cstrt_num + added_Pdata_num + QCQP.n_gen_constr, dtype=float
        )
        new_lags[:proj_cstrt_num] = QCQP.current_lags[:proj_cstrt_num]
        new_lags[proj_cstrt_num + added_Pdata_num :] = QCQP.current_lags[
            proj_cstrt_num:
        ]
        QCQP.current_lags = new_lags

    if orthonormalize:
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

    if orthonormalize:
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
        QCQP.add_constraints(new_Pdata_list, orthonormalize=orthonormalize)
        # informally checked that new constraints are added in orthonormal fashion

        ## merge old constraints if necessary
        if len(QCQP.Proj) > max_proj_cstrt_num:
            QCQP.merge_lead_constraints(
                merged_num=len(QCQP.Proj) - max_proj_cstrt_num + 1
            )
