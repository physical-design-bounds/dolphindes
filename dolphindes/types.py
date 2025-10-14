"""Typing helpers (internal)."""

from typing import TypeAlias, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

# Public-facing (input) wide types
ArrayLikeFloat: TypeAlias = ArrayLike

FloatNDArray: TypeAlias = NDArray[np.float64]

ComplexArray: TypeAlias = NDArray[np.complexfloating]
ComplexGrid: TypeAlias = NDArray[np.complexfloating]
BoolGrid: TypeAlias = NDArray[np.bool_]
SparseDense: TypeAlias = Union[ComplexGrid, sp.sparray]

__all__ = [
    "ArrayLikeFloat",
    "FloatNDArray",
    "ComplexArray",
    "ComplexGrid",
    "BoolGrid",
    "SparseDense",
]
