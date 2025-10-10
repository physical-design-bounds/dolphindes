"""
Generalized Constraint Descent (GCD) methods below.

GCD is a method for tightening dual bounds on a shared projection QCQP
based on iteratively
1. adding of new shared projection constraints that tighten the bound
2. merging of old constraints that does not change the bound to keep total constraint number small.
For usage examples, see the notebooks examples/LDOS_gcd and examples/verlan/LDOS_verlan
For more mathematical details, see Appendix B of https://arxiv.org/abs/2504.10469
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from dolphindes.util import CRdot, Sym

# import the base class type for annotations
from dolphindes.cvxopt._base_qcqp import _SharedProjQCQP

def merge_lead_constraints(QCQP: _SharedProjQCQP, merged_num: int = 2) -> None:
    """
    Merge the first m constraints of QCQP into a single constraint.
    Also adjust the Lagrange multipliers so the dual value is the same.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        QCQP for which we merge the leading constraints.
    merged_num : int (optional, default 2)
        Number of leading constraints that we are merging together; should be at least 2.

    Raises
    ------
    ValueError
        If merged_num < 2 or if there are insufficient constraints for merging.
    """

    cstrt_num = len(QCQP.Proj)
    if merged_num < 2:
        raise ValueError("Need at least 2 constraints for merging.")
    if cstrt_num < merged_num:
        raise ValueError("Number of constraints insufficient for size of merge.")

    new_P = QCQP.Proj.Pstruct.astype(complex, copy=True)
    new_P.data[:] = 0.0
    for i in range(merged_num):
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
        QCQP.Fs = QCQP.Fs[:, merged_num-1:]
        if sp.issparse(new_P):
            QCQP.Fs[:,0] = QCQP.A2.conj().T @ (new_P.conj().T @ QCQP.s1)
        else:
            QCQP.Fs[:,0] = QCQP.A2.conj().T @ (new_P.conj() * QCQP.s1)

    QCQP.current_lags = QCQP.current_lags[merged_num-1:]
    QCQP.current_lags[0] = Pnorm
    
    QCQP.current_grad = QCQP.current_hess = (
        None  # in principle can merge dual derivatives but leave it undone for now
    )


def add_constraints(
    QCQP: _SharedProjQCQP, added_Pdiag_list: list, orthonormalize: bool = True
) -> None:
    """
    Method that adds new shared projection constraints into an existing QCQP.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        QCQP for which the new constraints are added in.
    added_Pdiag_list : list
        List of 1d numpy arrays that are the new constraints to be added in.
    orthonormalize : bool, optional
        If true, assume that QCQP has orthonormal constraints and keeps it that way.
    """
    x_size, cstrt_num = QCQP.Pdiags.shape
    added_Pdiag_num = len(added_Pdiag_list)

    if QCQP.current_lags is not None:
        new_lags = np.zeros(cstrt_num + added_Pdiag_num, dtype=float)
        new_lags[:cstrt_num] = QCQP.current_lags
    else:
        new_lags = None

    new_Pdiags = np.zeros((x_size, cstrt_num + added_Pdiag_num), dtype=complex)
    new_Pdiags[:, :cstrt_num] = QCQP.Pdiags
    if not orthonormalize:
        for m, added_Pdiag in enumerate(added_Pdiag_list):
            new_Pdiags[:, cstrt_num + m] = added_Pdiag

    else:
        for m, added_Pdiag in enumerate(added_Pdiag_list):
            # do Gram-Schmidt orthogonalization for each added Pdiag
            for j in range(cstrt_num + m):
                added_Pdiag -= CRdot(new_Pdiags[:, j], added_Pdiag) * new_Pdiags[:, j]
            added_Pdiag /= la.norm(added_Pdiag)

            new_Pdiags[:, cstrt_num + m] = added_Pdiag

    # update QCQP
    if hasattr(QCQP, "precomputed_As"):
        # updated precomputed_As
        for added_Pdiag in added_Pdiag_list:
            QCQP.precomputed_As.append(
                Sym(QCQP.A1 @ sp.diags_array(added_Pdiag, format="csr") @ QCQP.A2)
            )

    if hasattr(QCQP, "Fs"):
        new_Fs = np.zeros((x_size, cstrt_num + added_Pdiag_num), dtype=complex)
        new_Fs[:, :cstrt_num] = QCQP.Fs
        new_Fs[:, cstrt_num:] = (
            QCQP.A2.conj().T @ (new_Pdiags[:, cstrt_num:].conj().T * QCQP.s1).T
        )
        QCQP.Fs = new_Fs

    QCQP.Pdiags = new_Pdiags
    QCQP.current_lags = new_lags
    QCQP.current_grad = QCQP.current_hess = None


def run_gcd(
    QCQP: _SharedProjQCQP,
    max_cstrt_num: int = 10,
    orthonormalize: bool = True,
    opt_params=None,
    max_gcd_iter_num=50,
    gcd_iter_period=5,
    gcd_tol=1e-2,
):
    """
    Perform generalized constraint descent to gradually refine dual bound on QCQP.

    At each GCD iteration, add two new constraints:
    1.a constraint generated so the corresponding dual derivative is large,
    to hopefully tighten the dual bound
    2. a constraint generated so the corresponding derivative of the smallest
    Lagrangian quadratic form eigenvalue is large, to help the dual optimization
    navigate the semi-definite boundary

    If the total number of constraints is larger than max_gcd_cstrt_num combine
    the earlier constraints to keep the total number of constraints fixed. Setting
    max_cstrt_num large enough will eventually result in evaluating the dual bound
    with all possible constraints, which gives the tightest bound but may be extremely
    expensive. The goal of GCD is to approximate this tightest bound with greatly reduced
    computational cost.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        The SharedProjQCQP for which we compute and refine dual bounds.
    max_cstrt_num : int, optional
        The maximum constraint number for QCQP. The default is 10.
    orthonormalize : bool, optional
        Whether or not to orthonormalize the constraint projectors. The default is True.
    opt_params : dict, optional
        The opt_params for the internal _Optimizer run at every GCD iteration.
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
    # since GCD is constantly changing the constraints, no need for many fake source iterations
    OPT_PARAMS_DEFAULTS = {"max_restart": 1}
    if opt_params is None:
        opt_params = {}
    opt_params = {**OPT_PARAMS_DEFAULTS, **opt_params}

    # get to feasible point
    QCQP.current_lags = QCQP.find_feasible_lags()
    if orthonormalize:
        # orthonormalize QCQP
        # informally checked for correctness
        x_size, cstrt_num = QCQP.Pdiags.shape
        realext_Pdiags = np.zeros((2 * x_size, cstrt_num), dtype=float)
        realext_Pdiags[:x_size, :] = np.real(QCQP.Pdiags)
        realext_Pdiags[x_size:, :] = np.imag(QCQP.Pdiags)
        realext_Pdiags_Q, realext_Pdiags_R = la.qr(realext_Pdiags, mode="economic")
        QCQP.Pdiags = realext_Pdiags_Q[:x_size, :] + 1j * realext_Pdiags_Q[x_size:, :]
        QCQP.current_lags = realext_Pdiags_R @ QCQP.current_lags
        QCQP.compute_precomputed_values()

    ## gcd loop
    gcd_iter_num = 0
    gcd_prev_dual = np.inf
    while True:
        gcd_iter_num += 1
        # solve current dual problem
        QCQP.solve_current_dual_problem(
            "newton", init_lags=QCQP.current_lags, opt_params=opt_params
        )
        print(
            f"At GCD iteration #{gcd_iter_num}, best dual bound found is {QCQP.current_dual}."
        )

        ## termination conditions
        if gcd_iter_num > max_gcd_iter_num:
            break
        if gcd_iter_num % gcd_iter_period == 0:
            if gcd_prev_dual - QCQP.current_dual < gcd_tol * abs(gcd_prev_dual):
                break
            gcd_prev_dual = QCQP.current_dual

        ## generate max dualgrad constraint
        maxViol_Pdiag = (2 * QCQP.s1 - (QCQP.A1.conj().T @ QCQP.current_xstar)) * (
            QCQP.A2 @ QCQP.current_xstar
        ).conj()

        ## generate min A eig constraint
        minAeigv, minAeigw = QCQP._get_PSD_penalty(QCQP.current_lags)
        minAeig_Pdiag = (QCQP.A1.conj().T @ minAeigv) * (QCQP.A2 @ minAeigv).conj()
        # use the same relative weights for minAeig_Pdiag as maxViol_Pdiag
        minAeig_Pdiag /= np.sqrt(np.real(minAeig_Pdiag.conj() * minAeig_Pdiag))
        minAeig_Pdiag * np.sqrt(np.real(maxViol_Pdiag.conj() * maxViol_Pdiag))
        # informally checked that minAeigw increases when increasing multiplier of minAeig_Pdiag

        ## add new constraints
        QCQP.add_constraints([maxViol_Pdiag, minAeig_Pdiag])
        # informally checked that new constraints are added in orthonormal fashion

        ## merge old constraints if necessary
        if QCQP.Pdiags.shape[1] > max_cstrt_num:
            QCQP.merge_lead_constraints(
                merged_num=QCQP.Pdiags.shape[1] - max_cstrt_num + 1
            )
