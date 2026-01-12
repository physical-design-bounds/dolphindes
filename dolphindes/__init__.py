"""
DolphinDes.

Provides
  1. An interface to relax Quadratically Constrained Quadratic Programs with constraints
  that differ only by projection operators to their dual problem
  2. Convex optimization routines to solve these dual problems
  3. An interface for writing photonic optimization problems in the form of (1) for
  calculating limits to photonic performance
  4. Basic code to solve Maxwell's equations and compute Maxwell Green's functions in 2D
  5. Basic utilities for photonic inverse design for comparison with bounds

Available subpackages
---------------------
cvxopt
    Optimization problems and routines
photonics
    Photonic dual optimization interface
maxwell
    Maxwell solver
util
    Utilities
"""

from . import cvxopt, geometry, maxwell, photonics, util

__version__ = "0.1.0"

__all__ = ["photonics", "cvxopt", "geometry", "maxwell", "util", "__version__"]
