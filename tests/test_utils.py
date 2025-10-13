"""Unit tests for utility functions in dolphindes.util."""

import numpy as np
import scipy.sparse as sp

from dolphindes.util import Projectors


def test_Projectors():
    ## test validate_projector
    struct = sp.csc_array([[0,1,0],
                           [0,0,1],
                           [1,0,0]])
    size = struct.size
    numP = 2
    Plist = []
    for i in range(numP):
        Plist.append(struct.copy())
        Plist[-1].data = np.random.randint(1,8, size=size)
        print(Plist[-1].todense())
    
    Proj = Projectors(Plist, struct)
    
    test1 = sp.csc_array([[0,0,0],
                          [0,0,1],
                          [1,0,0]])
    
    assert Proj.validate_projector(test1), (
        "validate_projector false negative for sparser array."
        )
    
    test2 = sp.csc_array([[1,1,0],
                          [0,0,1],
                          [1,0,0]])
    
    assert not Proj.validate_projector(test2), (
        "validate_projector false positive."
        )
    
    test2[0,0] = 0
    assert not Proj.validate_projector(test2), (
        "validate_projector conflated explicit zeros and sparsity structure."
        )