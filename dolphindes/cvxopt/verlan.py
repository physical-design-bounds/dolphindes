"""Compute initializations for QCQPs using Verlan algorithm with scraping.

See https://arxiv.org/abs/2504.14083 for mathematical details.
"""

import copy
from dataclasses import dataclass
from typing import Callable, Tuple

import scipy.sparse as sp

from dolphindes.cvxopt import GCDHyperparameters
from dolphindes.cvxopt._base_qcqp import _SharedProjQCQP
from dolphindes.types import ComplexArray
from dolphindes.util import math_utils


@dataclass(frozen=True)
class VerlanHyperparameters:
    """Hyperparameters for Verlan algorithm.

    Parameters
    ----------
    method : str
        Method to use for Verlan. Options are "gcd" (default).
        More options may be added in the future.
    delta : float
        Tolerance for checking strong duality. Default is 1e-3.
        Bigger is less tolerant!
    max_iter : int | float
        Maximum number of Verlan iterations to perform. Default is np.inf.
    theta : float
        Angle in degrees to rotate s0 towards xstar during scraping. Default is 5.0.
    n_theta : int
        Number of scraping attempts per Verlan iteration. Default is 10.
        If n_theta is reached without finding strong duality,
        the algorithm reduces delta_t and retries.
    delta_t : float
        Starting step size to increase t when strong duality is found.
    min_delta_t: float
        Minimum step size to increase t when strong duality is found.
    delta_t_growth_factor : float
        Multiplicative factor to restore delta_t after successful iterations.
    max_verlan_iters : int
        Maximum number of Verlan iterations before giving up.
    verbose : float
        Verbosity level, either 0 (silent), 1 (Verlan prints only), or
        2 (Verlan + solve prints)
    """

    method: str = "gcd"
    sd_search_tol: float = 0.1
    delta: float = 1e-3
    theta: float = 5.0  # degrees
    n_theta: int = 10
    delta_t: float = 0.1
    t_low: float = 1e-5
    t_high: float = 0.9
    min_delta_t: float = 1e-6
    delta_t_growth_factor: float = 1.5
    max_verlan_iters: int = 50
    gcd_params: GCDHyperparameters = GCDHyperparameters()
    verbose: float = 1.0


def _single_solve(
    QCQP: _SharedProjQCQP,
    gcd_params: GCDHyperparameters,
    t: float,
    At: Callable[
        [float],
        Tuple[
            ComplexArray | sp.csc_array,
            ComplexArray | sp.csc_array,
            ComplexArray | sp.csc_array,
        ],
    ],
) -> None:
    A0, A1, A2 = At(t)

    # Preserve a previously feasible warm-start if it still works after reinit
    prev_lags = QCQP.current_lags
    QCQP.reinitialize_A_operators(A0, A1, A2)

    # Helper to add PSD constraint if needed
    if prev_lags is not None and QCQP.is_dual_feasible(prev_lags):
        QCQP.current_lags = prev_lags
    else:
        raise NotImplementedError("Reinitialization logic disabled for now.")
        # psd_idx = QCQP.add_single_constraint(None)
        # QCQP.current_lags = QCQP.find_feasible_lags(idx=psd_idx)

    QCQP.run_gcd(gcd_params)
    QCQP.solve_current_dual_problem(
        method="bfgs", init_lags=QCQP.current_lags, opt_params=None
    )


def scrape(QCQP: _SharedProjQCQP, theta: float) -> None:
    """Based on an existing solve, rotate the objective s0 towards xstar."""
    A2dagger_xstar = QCQP.get_A2_dagger_x(QCQP.current_xstar)
    QCQP.s0 = math_utils.rotate_toward(QCQP.s0, A2dagger_xstar, theta)


def run_verlan(
    QCQP: "_SharedProjQCQP",
    At: Callable[
        [float],
        Tuple[
            ComplexArray | sp.csc_array,
            ComplexArray | sp.csc_array,
            ComplexArray | sp.csc_array,
        ],
    ],
    verlan_params: VerlanHyperparameters = VerlanHyperparameters(),
) -> "_SharedProjQCQP":
    assert QCQP is not None, "QCQP must be initialized to run verlan."
    tmp_QCQP = copy.deepcopy(QCQP)

    gcd_params = verlan_params.gcd_params
    verlan_tol = verlan_params.delta

    _single_solve(tmp_QCQP, gcd_params, t=1.0, At=At)
    viol, SD = tmp_QCQP.is_strongly_dual(tmp_QCQP.current_xstar, verlan_tol)
    current_t = 1.0
    current_sd = SD
    current_delta_t = verlan_params.delta_t
    initial_delta_t = verlan_params.delta_t
    growth_factor = verlan_params.delta_t_growth_factor
    if SD:
        if verlan_params.verbose >= 1:
            print("Original problem is strongly dual! Done.")
        return tmp_QCQP
    if verlan_params.verbose >= 1:
        print("Original problem is not strongly dual. Violation: ", viol)

    return None
    # def check_strong_duality_wrapper(
    #     some_QCQP: "_SharedProjQCQP", t_val: float
    # ) -> bool:
    #     nonlocal current_t, current_sd, viol
    #     _single_solve(some_QCQP, gcd_params, t_val, At)
    #     viol_check, sd_local = some_QCQP.is_strongly_dual(
    #         some_QCQP.current_xstar, tol=verlan_tol
    #     )
    #     if sd_local:
    #         viol = viol_check
    #     if verlan_params.verbose >= 1:
    #         print(f"t = {t_val:.3e}, SD Violation = {viol_check:.3e}.")
    #     current_t = t_val
    #     current_sd = sd_local
    #     return sd_local

    # # Step 1: find initial t with strong duality
    # if verlan_params.verbose >= 1:
    #     print_underline("Finding t such that small problem is strongly dual...")
    # t_low = verlan_params.t_low
    # t_high = verlan_params.t_high
    # t_SD, success = math_utils.bool_binary_search(
    #     func=lambda t_val: check_strong_duality_wrapper(tmp_QCQP, t_val),
    #     low=t_low,
    #     high=t_high,
    #     tol=verlan_params.sd_search_tol,
    # )
    # assert success, (
    #     "Failed! That shouldn't happen. Increase amount of loss in A(t) / lower t_low."
    # )
    # if verlan_params.verbose >= 1:
    #     print(f"Found t_SD ≈ {t_SD:.3e} with strong duality. Violation: {viol:.3e}")
    #     print_underline("Starting Verlan iterations...")

    # check_strong_duality_wrapper(tmp_QCQP, t_SD)
    # SD_QCQP = copy.deepcopy(tmp_QCQP)
    # # Track the t parameter of the last strongly dual solution for backtracking (logic fix)
    # last_good_t = current_t

    # # Step 2: Verlan iterations
    # # Scrape and increase t until t=1 with strong duality
    # verlan_iter = 0
    # # Track how many times we've hit min delta_t at the same t
    # last_t_min_marker = None
    # min_at_t_hits = 0

    # while True:
    #     # 2.1: Check for convergence
    #     if current_t >= 1.0 and current_sd:
    #         if verlan_params.verbose >= 1:
    #             print("Reached t = 1 with strong duality.")
    #         break
    #     if verlan_iter >= verlan_params.max_verlan_iters:
    #         if verlan_params.verbose >= 1:
    #             print("Reached maximum Verlan iterations without t = 1 strong duality.")
    #         break

    #     # Check scrape condition at the start of loop
    #     if not current_sd:
    #         # If not strongly dual, attempt scraping n_theta times until strong duality.
    #         scraped_success = False
    #         for scrape_attempt in range(verlan_params.n_theta):
    #             scrape(tmp_QCQP, verlan_params.theta)
    #             sd_found = check_strong_duality_wrapper(tmp_QCQP, current_t)
    #             if sd_found:
    #                 scraped_success = True
    #                 SD_QCQP = copy.deepcopy(tmp_QCQP)
    #                 last_good_t = current_t
    #                 if verlan_params.verbose >= 1:
    #                     print(
    #                         f"Found strong duality after {scrape_attempt + 1} "
    #                         f"attempt(s)."
    #                     )
    #                 break
    #         if not scraped_success:
    #             if verlan_params.verbose >= 1:
    #                 print(
    #                     f"  Failed to find strong duality after "
    #                     f"{verlan_params.n_theta} scraping attempts."
    #                 )

    #             # Logic Fix: Only backtrack if scraping consistently fails
    #             # Revert to last good state
    #             tmp_QCQP = copy.deepcopy(SD_QCQP)
    #             current_t = last_good_t
    #             check_strong_duality_wrapper(
    #                 tmp_QCQP, current_t
    #             )  # Restore nonlocal state
    #             current_sd = True

    #             current_delta_t = max(current_delta_t * 0.5, verlan_params.min_delta_t)
    #             if verlan_params.verbose >= 1:
    #                 print(
    #                     f"Backtracking to t={current_t:.3e} and reducing delta_t to {current_delta_t:.3e}."
    #                 )

    #             # If we've hit the minimum delta_t twice at this t, give up.
    #             if current_delta_t == verlan_params.min_delta_t:
    #                 if last_t_min_marker == current_t:
    #                     min_at_t_hits += 1
    #                 else:
    #                     last_t_min_marker = current_t
    #                     min_at_t_hits = 1
    #                 if min_at_t_hits >= 2:
    #                     if verlan_params.verbose >= 1:
    #                         print(
    #                             f"delta_t hit minimum twice at t = {current_t:.3e}; giving up."
    #                         )
    #                     break
    #             continue
    #         else:
    #             if verlan_params.verbose >= 1:
    #                 print("Scraped successfully to strong duality.")
    #                 print(f"Current t = {current_t:.3e}, violation = {viol:.3e}.")

    #     # Advance state
    #     t_new = min(current_t + current_delta_t, 1.0)
    #     sd_after_increase = check_strong_duality_wrapper(tmp_QCQP, t_new)

    #     if not sd_after_increase:
    #         if verlan_params.verbose >= 1:
    #             print(
    #                 f"Failed to keep strong duality at t = {t_new:.3e}. Attempting to scrape..."
    #             )
    #         # Logic Fix: Do NOT backtrack here. Loop around to attempt scraping at t_new.
    #         continue
    #     else:
    #         SD_QCQP = copy.deepcopy(tmp_QCQP)
    #         last_good_t = current_t
    #         if current_delta_t < initial_delta_t:
    #             increased_delta_t = min(
    #                 current_delta_t * growth_factor, initial_delta_t
    #             )
    #             if verlan_params.verbose >= 1:
    #                 print(f"Increasing delta_t to {increased_delta_t:.3e}.")
    #             current_delta_t = increased_delta_t
    #         # Successful t increase; moving t resets the per-t minimum-hit logic.
    #         last_t_min_marker = current_t
    #         min_at_t_hits = 0

    #     verlan_iter += 1
    #     if verlan_params.verbose >= 1:
    #         print(
    #             f"Verlan iteration {verlan_iter}: Successfully increased t to {t_new:.3e}. "
    #             f"Violation: {viol:.3e}."
    #         )
    #     print()

    # # Finally, return with QCQP solved at t=1 (or best strongly dual solution reached)
    # return SD_QCQP
