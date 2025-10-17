"""Photonics simulation and optimization package."""

from ._base_photonics import (
    CartesianFDFDGeometry,
    GeometryHyperparameters,
    Photonics_FDFD,
)
from .photonics import Photonics_TE_Yee_FDFD, Photonics_TM_FDFD, chi_to_feasible_rho

# from .verlan import VerlanHyperparameters, VerlanProblem

__all__ = [
    "GeometryHyperparameters",
    "CartesianFDFDGeometry",
    "Photonics_FDFD",
    "Photonics_TM_FDFD",
    "Photonics_TE_Yee_FDFD",
    "chi_to_feasible_rho",
    # "VerlanProblem",
    # "VerlanHyperparameters",
]
