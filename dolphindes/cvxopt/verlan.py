"""Computer structures using Verlan algorithm."""

import copy
from dataclasses import dataclass
from typing import Callable, Tuple, cast

import numpy as np

from dolphindes.cvxopt import GCDHyperparameters, gcd
from dolphindes.cvxopt._base_qcqp import _SharedProjQCQP
from dolphindes.types import ComplexArray
from dolphindes.util import math_utils

# write a is_strongly_dual function for QCQPs. This must be overriden in general - can't do it generally.
# write a function to compute A2inv^dagger s0 for QCQPs. Abstract method. For photonics, override it with the answer.
# For dense, A2 will be identity, so just return s0.

# Plan:
# 1. I want verlan code that works on just QCQPs, but this is hard for multiple reasons:
#    a. I need a function that takes a QCQP and returns True if it is strongly dual
#    b. I need A2inv^dagger s0, but I don't want to invert A2 inside QCQP unless absolutely necessary
#       Maybe I can have a function that does this in QCQP, and you can override it if you know it?
# 2. So instead I will write a general verlan scheme that works for QCQPs, but leaves some things not implemented.
#    Then I will subclass it for Photonics and implement the missing pieces.
# 3. Hopefully this means Sean or the others can use it for other problems too, such as subset sum.

# So what are the components?
# First,

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
    loss_peak : float
        Peak loss (imaginary part of chi) at t = 0.5. Default is 0.1.
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
    """

    method: str = "gcd"
    delta: float = 1e-3
    loss_peak: float = 0.1
    max_iter: int | float = np.inf
    theta: float = 5.0  # degrees
    n_theta: int = 10
    delta_t: float = 0.1
    gcd_params: GCDHyperparameters = GCDHyperparameters()


def run_verlan(
    s0: ComplexArray,
    QCQP: "_SharedProjQCQP",
    verlan_params: VerlanHyperparameters = VerlanHyperparameters(),
    At,
):
    norm_s0 = float(np.linalg.norm(s0))
    return norm_s0
    
def _single_solve(
    QCQP: _SharedProjQCQP,
    t: float,
    At: Callable
):
    pass

class VerlanProblem:
    def __init__(
        self, PhotonicProblem, verlan_params: VerlanHyperparameters
    ) -> None:
        self.PhotonicProblem = copy.deepcopy(PhotonicProblem)

        self.dense_s0, self.norm_dense_s0 = self._get_dense_s0()
        self.dense_s0_original = self.dense_s0.copy()
        self.chi_original = copy.copy(self.PhotonicProblem.chi)
        self.params = verlan_params
        self.current_t = 1

    def _get_chi_t(self, t: float, loss_peak: float) -> complex:
        """Get chi at interpolation parameter t.

        t = 0 -> chi = 0
        t = 1 -> chi = chi_original
        chi(t) is a quadratic function of t such that Im(chi) has a peak
        at t = 0.5 of height loss_peak.
        """
        real_chi = t * np.real(self.chi_original)
        imag_chi = (2 * np.imag(self.chi_original) - 4 * loss_peak) * (t**2) + (
            4 * loss_peak - np.imag(self.chi_original)
        ) * t
        return complex(real_chi, imag_chi)

    def _set_chi(self, chi: complex) -> None:
        self.PhotonicProblem.chi = chi
        self.PhotonicProblem.setup_QCQP()

    def check_strong_duality(self, delta: float) -> bool:
        """Check if strong duality holds for the photonic problem for given tolerance.

        Arguments
        ---------
        delta : float
            Tolerance for checking if the compact constraint gradient is zero.

        Returns
        -------
        bool
            True if strong duality holds within tolerance delta.

        Notes
        -----
        In photonics, this is true if the compact constraint
            Asym[p† (chi^-† - G_0†)^{-1} p - 2 Re (p† s0)] = 0
        where s0 is the linear objective parameter.
        We assume that this is index 1 of the QCQP so we can just check
        its gradient.
        """
        gradient = self.PhotonicProblem.QCQP.current_grad
        if gradient is None:
            raise ValueError("QCQP has not been solved yet.")
        compact_gradient = gradient[1]
        return bool(np.linalg.norm(compact_gradient) < delta)

    def _single_solve(
        self,
        PhotonicProblem: Photonics_FDFD,
        t: float,
    ) -> None:
        """Solve the photonic problem at interpolation parameter t.

        Modifies the PhotonicProblem and PhotonicProblem.QCQP in place.
        """
        PhotonicProblem.chi = self._get_chi_t(t, self.params.loss_peak)
        PhotonicProblem.setup_QCQP()
        if self.params.method == "gcd":
            gcd.run_gcd(PhotonicProblem.QCQP, self.params.gcd_params)
        else:
            raise ValueError(f"Unknown Verlan method {self.params.method}.")
        return

    # def find_initial_strongly_dual_point(self, PhotonicProblem: Photonics_FDFD, strategy: str = "binary_search") -> None:
    #     """Attempt to find the biggest 0 < t < 1 such that strong duality holds."""
    #     pass

    # def _scrape(self, PhotonicProblem: Photonics_FDFD, theta: float) -> Photonics_FDFD:
    #     """Based on an existing solve, rotate the objective s0 towards xstar.

    #     Arguments
    #     ---------
    #     theta : float
    #         Angle in degrees to rotate s0 towards xstar.
    #     PhotonicProblem : Photonics_FDFD
    #         The photonic problem to modify.

    #     Notes
    #     -----
    #     This modifies self.dense_s0 and updates the PhotonicProblem objective.
    #     """
    #     xstar = PhotonicProblem.QCQP.current_xstar
    #     if xstar is None:
    #         raise ValueError("QCQP has not been solved yet.")

    #     # If the problem is sparse, convert xstar to dense
    #     if PhotonicProblem.sparseQCQP:
    #         xstar = PhotonicProblem.Ginv @ xstar

    #     # Rescale xstar to have the same norm as s0
    #     # Rotate xstar towards s0 by angle theta
    #     xstar = xstar * self.norm_dense_s0 / np.linalg.norm(xstar)
    #     s0_rotated = math_utils.rotate_toward(PhotonicProblem.dense_s0, xstar, theta)
    #     if PhotonicProblem.sparseQCQP:
    #         PhotonicProblem.set_objective(s0=s0_rotated, denseToSparse=True)
    #     else:
    #         PhotonicProblem.set_objective(s0=s0_rotated, denseToSparse=False)
    #     PhotonicProblem.setup_QCQP()
    #     return PhotonicProblem

    # def _scrape_until_strong_duality(
    #     self, PhotonicProblem: Photonics_FDFD, theta: float
    # ) -> Tuple[Photonics_FDFD, bool]:
    #     """Scrape the objective until strong duality is found.

    #     Arguments
    #     ---------
    #     PhotonicProblem : Photonics_FDFD
    #         The photonic problem to modify.
    #     theta : float
    #         Angle in degrees to rotate s0 towards xstar during scraping.

    #     Returns
    #     -------
    #     Photonic_FDFD
    #         The modified photonic problem after scraping.
    #     bool
    #         True if strong duality is found, False otherwise.

    #     Notes
    #     -----
    #     Terminates if
    #     1. Strong duality is found
    #     2. n_theta attempts are made without finding strong duality
    #     3. xstar has converged without finding strong duality
    #     """
    #     return PhotonicProblem, False

    # def verlan_iteration(self) -> None:
    #     """Perform a single iteration of the Verlan algorithm.

    #     This is a single increase in t followed by scraping until strong duality is found.
    #     If strong duality is not found in enough scraping attempts, the function returns without modifying t or s0.
    #     """
    #     pass

    # def verlan_run(self) -> None:
    #     """Run Verlan algorithm until strong duality is found with t=1.

    #     Arguments
    #     ---------

    #     Notes
    #     -----
    #     This modifies self.current_t, self.dense_s0, and the PhotonicProblem chi.
    #     """
    #     # 1. Use binary search to find initial t with strong duality
    #     t = self.find_initial_strongly_dual_point(strategy="binary_search")

    #     # 2. Verlan iterations
    #     verlan_iter = 0
    #     success = True
    #     failure_counter = 0
    #     t_new = t
    #     delta_t = self.params.delta_t
    #     n_theta = self.params.n_theta
    #     delta = self.params.delta
