"""Useful mathematical operations."""

import numpy as np

from dolphindes.types import ComplexArray, SparseDense


def Sym(A: SparseDense) -> SparseDense:
    """Compute the symmetric Hermitian part of a matrix A."""
    return (A + A.T.conj()) / 2


def CRdot(v1: ComplexArray, v2: ComplexArray) -> float:
    """
    Compute the inner product of two complex vectors over a real field.

    In other words, the vectors have complex values but linear combination coefficients
    have to be real. This is the vector space for the complex QCQP constraints since
    Lagrangian multipliers are real.

    Parameters
    ----------
    v1 : np.ndarray
        vector1
    v2 : np.ndarray
        vector2

    Returns
    -------
    The inner product
    """
    return float(np.real(np.vdot(v1, v2)))
