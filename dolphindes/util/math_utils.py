"""Useful mathematical operations."""

from collections.abc import Callable

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


def rotate_toward(a: ComplexArray, b: ComplexArray, theta: float) -> ComplexArray:
    """
    Rotates a complex vector 'a' towards another complex vector 'b' by a given angle.

    This function operates in an n-dimensional complex space. It rotates 'a'
    within the 2D plane spanned by 'a' and 'b'.

    Args:
        a: The complex numpy array to be rotated.
        b: The complex numpy array representing the target direction.
        theta: The maximum angle of rotation in degrees.

    Returns
    -------
    ComplexArray
        The rotated complex numpy array.

    Logic:
    1.  Calculates the angle (theta_0) between 'a' and 'b'.
    2.  If theta_0 is less than or equal to the desired rotation angle (theta),
        the function returns 'b' directly (no overshooting), normalized to the
        magnitude of 'a'.
    3.  If theta_0 > theta, it constructs an orthonormal basis {u, v} for the
        plane defined by 'a' and 'b'.
    4.  It then performs a 2D rotation of 'a' within this plane by the angle theta.
        The magnitude of the original vector 'a' is preserved.

    Note:
    1.  If either 'a' or 'b' is a zero vector, the function returns 'a' unchanged.
        Rotation is meaningless in this case.
    2.  If 'a' and 'b' are collinear (including anti-parallel), the plane is not
        unique. In this case, the function selects an arbitrary orthogonal direction to
        define the rotation plane.
    """
    theta_rad = np.deg2rad(theta)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle edge cases: if 'a' or 'b' is a zero vector, rotation is meaningless.
    if np.isclose(norm_a, 0) or np.isclose(norm_b, 0):
        return a

    # Calculate Angle Between Vectors
    inner_product = np.vdot(a, b)

    # We clip the value to handle potential floating point inaccuracies slightly > 1.0
    cos_theta_0 = np.clip(np.abs(inner_product) / (norm_a * norm_b), -1.0, 1.0)
    theta_0_rad = np.arccos(cos_theta_0)

    # If the angle between vectors is smaller than our rotation angle, return b
    if theta_0_rad <= theta_rad:
        return b * (norm_a / norm_b)

    # Get orthonormal basis {u, v}
    u = a / norm_a
    b_perp = b - np.vdot(u, b) * u
    norm_b_perp = np.linalg.norm(b_perp)

    if norm_b_perp < 1e-10:
        Warning(
            "Vectors 'a' and 'b' are collinear; rotation plane is not unique. \n \
                 This is very special! Please file an issue on GitHub."
        )
        # If a and b are collinear and anti-parallel, the rotation plane is not unique.
        if a.size <= 1:
            return a

        # Find a standard basis vector that is not collinear with u.
        # We start with e_1 = [1, 0, ...].
        temp_vec = np.zeros_like(a)
        temp_vec[0] = 1.0
        ortho_vec = temp_vec - np.vdot(u, temp_vec) * u

        # If temp_vec (e_1) was collinear with u, its orthogonal component will be zero.
        # In this case, we must try a different basis vector, like e_2 = [0, 1, ...].
        if np.linalg.norm(ortho_vec) < 1e-10:
            temp_vec = np.zeros_like(a)
            temp_vec[1] = 1.0
            ortho_vec = temp_vec - np.vdot(u, temp_vec) * u

        # Normalize the resulting orthogonal vector to get v.
        v = ortho_vec / np.linalg.norm(ortho_vec)
    else:
        v = b_perp / norm_b_perp

    # Perform the 2D rotation in the {u, v} basis.
    # The new vector has the same magnitude as the original vector 'a'.
    a_rotated: ComplexArray = norm_a * (np.cos(theta_rad) * u + np.sin(theta_rad) * v)

    return a_rotated


def bool_binary_search(
    func: Callable[[float], bool], low: float, high: float, tol: float = 0.5
) -> tuple[float, bool]:
    """
    Perform a binary search to find the largest value in [low, high] where func is True.

    The function `func` should be monotonic: if func(x) is True, then func(y) is also
    True for all y < x. Conversely, if func(x) is False, then func(y) is also False for
    all y > x.

    Parameters
    ----------
    func : callable
        A monotonic boolean function to evaluate.
    low : float
        The lower bound of the search interval.
    high : float
        The upper bound of the search interval.
    tol : float, optional
        The tolerance for stopping the search. The default is 1e-3.

    Returns
    -------
    tuple[float, bool]
        A tuple containing the largest value in [low, high] where func is True,
        and a boolean indicating whether such a value was found.
    """
    if low > high:
        raise ValueError("low must be less than or equal to high.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    f_low = func(low)
    f_high = func(high)

    if not f_low:
        return low, False
    if f_high:
        return high, True

    while high - low > tol:
        mid = (low + high) / 2
        if func(mid):
            low = mid
        else:
            high = mid

    return low, True
