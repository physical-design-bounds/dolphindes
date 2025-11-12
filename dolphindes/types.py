"""Typing helpers (internal)."""

from typing import TypeAlias, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

ArrayLikeFloat: TypeAlias = ArrayLike

FloatNDArray: TypeAlias = NDArray[np.float64]
IntNDArray: TypeAlias = NDArray[np.intp]

ComplexArray: TypeAlias = NDArray[np.complexfloating]
ComplexGrid: TypeAlias = NDArray[np.complexfloating]
BoolGrid: TypeAlias = NDArray[np.bool_]
SparseDense: TypeAlias = Union[ComplexGrid, sp.sparray]

__all__ = [
    "FloatNDArray",
    "IntNDArray",
    "ComplexArray",
    "ComplexGrid",
    "BoolGrid",
    "SparseDense",
]
