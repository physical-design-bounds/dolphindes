"""Utility functions and classes for dolphindes.

This package provides various utility functions including class utilities,
mathematical operations, and geometry utilities.
"""

from . import geometry_utils
from .class_utils import check_attributes
from .math_utils import CRdot, Sym
from .projectors import Projectors

__all__ = ["check_attributes", "CRdot", "Sym", "geometry_utils", "Projectors"]
