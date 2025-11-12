"""Unit tests for utility functions in dolphindes.util."""

import numpy as np
import pytest
import scipy.sparse as sp

from dolphindes.util import Projectors
from dolphindes.util.math_utils import bool_binary_search


def test_Projectors():
    """Unit tests for Projectors class in dolphindes.util."""
    struct = sp.csc_array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    size = struct.size
    numP = 2
    Plist = []
    for i in range(numP):
        Plist.append(struct.copy())
        Plist[-1].data = np.random.randint(1, 8, size=size)
        print(Plist[-1].todense())

    Proj = Projectors(Plist, struct)

    test1 = sp.csc_array([[0, 0, 0], [0, 0, 1], [1, 0, 0]])

    assert Proj.validate_projector(test1), (
        "validate_projector false negative for sparser array."
    )

    test2 = sp.csc_array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])

    assert not Proj.validate_projector(test2), "validate_projector false positive."

    test2[0, 0] = 0
    assert not Proj.validate_projector(test2), (
        "validate_projector conflated explicit zeros and sparsity structure."
    )


def test_bool_binary_search_behaviour():
    """Unit tests for bool_binary_search function in dolphindes.util."""
    threshold = 3.5

    def func(x: float) -> bool:
        return x <= threshold

    value, found = bool_binary_search(func, 0.0, 5.0, tol=1e-3)
    assert found
    assert value == pytest.approx(threshold, abs=1e-3)

    value, found = bool_binary_search(lambda _: False, 0.0, 5.0)
    assert not found
    assert value == 0.0
