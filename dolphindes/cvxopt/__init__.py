"""Routines for optimization."""

from . import gcd
from ._base_qcqp import _SharedProjQCQP
from .gcd import GCDHyperparameters
from .optimization import BFGS, Alt_Newton_GD, OptimizationHyperparameters
from .qcqp import (
    DenseSharedProjQCQP,
    SparseSharedProjQCQP,
)

__all__ = [
    "_SharedProjQCQP",
    "SparseSharedProjQCQP",
    "DenseSharedProjQCQP",
    "BFGS",
    "Alt_Newton_GD",
    "OptimizationHyperparameters",
    "gcd",
    "GCDHyperparameters",
]
