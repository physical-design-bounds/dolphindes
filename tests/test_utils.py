"""Unit tests for utility functions in dolphindes.util."""

import numpy as np
import scipy.sparse as sp

from dolphindes.util import Projectors


def test_Projectors():
    ## test validate_projector
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


def test_Pdata_column_stack_sparser_than_Pstruct():
    """Non-diagonal projectors that each cover only part of Pstruct round-trip."""
    struct = sp.csr_array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    P1 = sp.csr_array([[2.0, 0, 0], [0, 0, 3.0], [0, 0, 0]], dtype=complex)
    P2 = sp.csr_array([[0, 1j, 0], [0, 5.0, 0], [4.0, 0, 0]], dtype=complex)
    Proj = Projectors([P1, P2], struct)
    rows, cols = Proj.Pstruct.nonzero()
    Pdata = Proj.get_Pdata_column_stack()
    assert Pdata.shape == (struct.size, 2)
    for j, P in enumerate((P1, P2)):
        np.testing.assert_allclose(Pdata[:, j], P.toarray()[rows, cols])
    Proj.set_Pdata_column_stack(Pdata)
    np.testing.assert_allclose(Proj.get_Pdata_column_stack(), Pdata)
