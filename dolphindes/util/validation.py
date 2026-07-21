"""Runtime input validation for public entry points.

These helpers enforce the type, dtype, and shape invariants that the numerical
routines assume, so that user mistakes surface as clear ``TypeError`` /
``ValueError`` messages at the library boundary.

Only array validation lives here, because it bundles several easy-to-get-wrong
steps (isinstance, dtype kind, shape/size) behind one clear message.

Convention (mirroring SciPy): public functions validate their own inputs; private
``_``-prefixed compute routines trust their callers and use plain ``assert`` only
for internal invariants that cannot be triggered by user input.
"""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from dolphindes.types import BoolGrid

__all__ = [
    "validate_bool_mask",
    "validate_numeric_array",
]


def validate_bool_mask(
    mask: object,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    size: int | None = None,
) -> BoolGrid:
    """
    Validate that ``mask`` is a boolean NumPy array of the expected extent.

    Parameters
    ----------
    mask : object
        The value to validate.
    name : str
        Name of the argument, used in error messages.
    shape : tuple of int, optional
        If given, the exact shape ``mask`` must have.
    size : int, optional
        If given, the exact number of elements ``mask`` must have. Useful when
        the rank is allowed to vary but the total extent is fixed.

    Returns
    -------
    numpy.ndarray of bool
        The validated ``mask`` (returned unchanged for convenient chaining).

    Raises
    ------
    TypeError
        If ``mask`` is not a ``numpy.ndarray``.
    ValueError
        If ``mask`` is not of boolean dtype or does not match ``shape``/``size``.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray, got {type(mask).__name__}."
        )
    if mask.dtype != np.bool_:
        raise ValueError(
            f"{name} must have boolean dtype, got dtype {mask.dtype}. "
            "Construct it with e.g. np.zeros(shape, dtype=bool)."
        )
    if shape is not None and mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {mask.shape}.")
    if size is not None and mask.size != size:
        raise ValueError(f"{name} must have {size} elements, got {mask.size}.")
    return cast(BoolGrid, mask)


def validate_numeric_array(
    arr: object,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    size: int | None = None,
) -> NDArray[Any]:
    """
    Validate that ``arr`` is a numeric NumPy array of the expected extent.

    A numeric dtype is any integer, unsigned-integer, floating, or complex
    dtype; real arrays are accepted where a complex field is expected because
    they are a valid special case.

    Parameters
    ----------
    arr : object
        The value to validate.
    name : str
        Name of the argument, used in error messages.
    shape : tuple of int, optional
        If given, the exact shape ``arr`` must have.
    size : int, optional
        If given, the exact number of elements ``arr`` must have.

    Returns
    -------
    numpy.ndarray
        The validated ``arr`` (returned unchanged for convenient chaining).

    Raises
    ------
    TypeError
        If ``arr`` is not a ``numpy.ndarray``.
    ValueError
        If ``arr`` is not numeric or does not match ``shape``/``size``.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}.")
    if arr.dtype.kind not in "iufc":
        raise ValueError(
            f"{name} must have a numeric (integer, float, or complex) dtype, "
            f"got dtype {arr.dtype}."
        )
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must have {size} elements, got {arr.size}.")
    return cast(NDArray[Any], arr)
