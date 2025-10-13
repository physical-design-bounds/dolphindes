"""Module to provide shared projector interface for optimizers."""

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from dolphindes.types import ComplexArray, FloatNDArray


def _detect_all_diagonal(matrix_list: list[sp.csr_array]) -> bool:
    """Return True if every matrix is square and has nonzeros only on the diagonal."""
    return all(
        mat.shape[0] == mat.shape[1]
        and all(r == c for r, c in zip(*mat.nonzero()))
        for mat in matrix_list
    )
    

class Projectors():
    """
    Class to handle sparse shared projectors.

    Parameters
    ----------
    Plist : ArrayLike
        List of sparse projector matrices.
    Pstruct : sp.csr_array
        Sparsity structure of the projectors.
    force_general : bool, optional
        If true, treat all projectors as general sparse matrices even if diagonal.
    """

    def __init__(
        self, 
        Plist: ArrayLike, 
        Pstruct: sp.csr_array, 
        force_general: bool = False
    ) -> None:
        self.Plist = []
        Pm = sp.csr_array(Pstruct)
        Pm = Pm.astype(bool, copy=True)
        Pm.data[:] = True
        self.Pstruct = Pm
        self._is_diagonal = _detect_all_diagonal([Pstruct]) and not force_general
        
        for P in Plist:
            P = sp.csr_array(P)
            if not self.validate_projector(P):
                raise ValueError("One of the provided projectors is invalid.")
            self.Plist.append(P)
            
        # Validate shapes and store metadata for fast slicing
        if not self.Plist:
            # Allow empty projector list
            self._k = 0
            self._n = None
            # Nothing else to build (no Pdiags/Pstack*)
            return

        n = self.Plist[0].shape[0]
        if any(P.shape != (n, n) for P in self.Plist):
            raise ValueError("All projectors must be square and have the same shape.")
        self._n = n
        self._k = len(self.Plist)

        if self._is_diagonal:
            self.Pdiags = np.column_stack([P.diagonal() for P in self.Plist])
        else:
            # Build vertical and horizontal stacks for P to avoid runtime transposes
            # the csr vs csc specifications chosen for convenience in converting to / from Pdata representation
            self.P_stackV = sp.vstack(self.Plist, format='csr')
            self.Pconj_stackH = sp.hstack([P.conj() for P in self.Plist], format='csc')
        del self.Plist  # We do not need the original list

    def validate_projector(self, P: sp.csr_array) -> bool:
        """Check if P is a valid projector (correct shape, subset of Pstruct)."""
        if P.shape != self.Pstruct.shape:
            return False
        Ptest = P.astype(bool, copy=True)
        Ptest.data[:] = True
        # for bool sparse arrays, + is OR and - is XOR
        outside = Ptest + self.Pstruct - self.Pstruct
        return outside.nnz == 0

    def __len__(self):
        return self._k

    def _getitem_diagonal(self, key: int) -> sp.csr_array:
        return sp.diags_array(self.Pdiags[:, key], format='csr')

    def _getitem_sparse(self, key: int) -> sp.csr_array:
        idx = key % self._k
        r0 = idx * self._n
        r1 = (idx + 1) * self._n
        # Extract block-rows corresponding to P[idx]
        return self.P_stackV[r0:r1, :]

    def __getitem__(self, key: int) -> sp.csr_array:
        """Return the key-th projector.
        
        If projectors are diagonal, return a CSC diag matrix from Pdiags.
        Else, slice the vertical stack to extract the block-rows for P[key].
        """
        if not isinstance(key, int):
            raise TypeError("Projector index must be an integer.")
        if self._k == 0:
            raise IndexError("No projectors available.")
        if self._is_diagonal:
            return self._getitem_diagonal(key)
        else:
            return self._getitem_sparse(key)

    def _setitem_diagonal(self, key: int, value: ArrayLike) -> None:
        try:
            value = sp.csr_array(value)
            self.Pdiags[:, key] = value.diagonal()
        except ValueError:
            # try again assuming that value is given as a 1D array
            self.Pdiags[:, key] = value
    
    def _setitem_sparse(self, key: int, value: ArrayLike) -> None:
        idx = key % self._k
        Pnew = sp.csr_array(value, dtype=self.P_stackV.dtype)
        if not self.validate_projector(Pnew):
            raise ValueError("New projector inconsistent with sparsity structure.")
        r0 = idx * self._n
        r1 = (idx + 1) * self._n
        # Keep both stacks consistent (store P and its adjoint)
        self.P_stackV[r0:r1, :] = Pnew
        self.Pconj_stackH[:, r0:r1] = Pnew.conj().tocsc() # check if removing is fine

    def __setitem__(self, key: int, value: ArrayLike) -> None:
        if not isinstance(key, int):
            raise TypeError("Projector index must be an integer.")
        if self._is_diagonal:
            self._setitem_diagonal(key, value)
        else:
            self._setitem_sparse(key, value)  

    def erase_leading(self, m: int) -> None:
        """
        removes first m projection matrices.
        """
        
        if self._is_diagonal:
            self.Pdiags = self.Pdiags[:, m:]
        else:
            self.P_stackV = self.P_stackV[m*self._n:, :]
            self.Pconj_stackH = self.Pconj_stackH[:, m*self._n:]
        
        self._k -= m
        return
    
    def append(self, Pnew: ArrayLike) -> None:
        """
        append a new projector
        """
        if self._is_diagonal:
            return self._append_diagonal(Pnew)
        else:
            return self._append_sparse(Pnew)
    
    def _append_diagonal(self, Pnew: ArrayLike) -> None:
        new_Pdiags = np.zeros((self._n, self._k+1), dtype=complex)
        new_Pdiags[:,:self._k] = self.Pdiags
        try:
            new_Pdiags[:,-1] = Pnew.diagonal()
        except ValueError:
            # try again assuming that value is given as a 1D array
            new_Pdiags[:,-1] = Pnew
        self.Pdiags = new_Pdiags
        self._k += 1
        return
    
    def _append_sparse(self, Pnew: ArrayLike) -> None:
        Pnew = sp.csr_array(Pnew)
        self._k += 1
        if not self.validate_projector(Pnew):
            raise ValueError("New projector inconsistent with sparsity structure.")
        self.P_stackV = sp.vstack((self.P_stackV, Pnew), format='csr')
        self.Pconj_stackH = sp.hstack((self.Pconj_stackH, Pnew.conj().tocsc()), format='csc')
        return

    def get_Pdata_column_stack(self) -> ComplexArray:
        """Extract all sparse P_j entries according to Pstruct ordering as columns of a (nnz,k) matrix.
        
        Returns a matrix whose j-th column is P_j[Pstruct]
        """
        if self._is_diagonal:
            return self.Pdiags[self.Pstruct.indices, :]
        
        P_stackV_fullsize_template = sp.vstack([self.Pstruct] * self._k, dtype=complex, format='csr')
        P_stackV_fullsize_template.data[:] = 0.0
        # template needed because individual P_j may be sparser than Pstruct
        Pdata_stack = (P_stackV_fullsize_template + self.P_stackV).data
        return Pdata_stack.reshape((self.Pstruct.size, self._k), order='F')
    
    def set_Pdata_column_stack(self, Pdata: ArrayLike):
        """Set the Projector using the columns of Pdata as the sparse entries of every P_j, 
        assuming the current Pstruct as given.
        """
        if Pdata.shape[0] != self.Pstruct.size:
            raise ValueError("Pdata size mismatch with Pstruct.")
        
        self._k = Pdata.shape[1]
        
        if self._is_diagonal:
            self.Pdiags = np.zeros((self._n,self._k), dtype=complex)
            self.Pdiags[self.Pstruct.indices, :] = Pdata
            return
        
        self.P_stackV = sp.vstack([self.Pstruct] * self._k, dtype=complex, format='csr')
        self.P_stackV.data = Pdata.flatten(order='F')
        
        self.Pconj_stackH = sp.hstack([self.Pstruct] * self._k, dtype=complex, format='csc')
        # get data permutation order to go from csr representation to csc representation
        permutation = self.Pstruct.astype(int, copy=True)
        permutation.data = np.arange(self.Pstruct.size)
        permutation = permutation.tocsc().data
        self.Pconj_stackH.data = Pdata[permutation, :].conj().flatten(order='F')
        return

    def allP_at_v(self, v: ComplexArray, dagger: bool = False) -> ComplexArray:
        """Compute all P_j @ v (or P_j^† @ v) and return an (n, k) matrix.

        Returns a matrix whose j-th column is P_j v (dagger=False) or P_j^† v (dagger=True).
        For diagonal projectors, dagger reduces to conjugation:
          allP_at_v(v, dagger=True) == (Pdiags.conj().T * v).T  (shape (n, k)).
        """
        if self._k == 0:
            # No projectors: return an (n, 0) array
            return np.zeros((v.shape[0], 0), dtype=complex)
        if self._is_diagonal:
            M = self.Pdiags.conj() if dagger else self.Pdiags
            return M * v[:, None]  # (n, k)
        # Use vertical stack or horizontal stack depending on dagger to avoid runtime transposes
        if dagger:
            stacked = v @ self.Pconj_stackH
        else:
            stacked = self.P_stackV @ v
        return stacked.reshape((self._n, self._k), order='F')          # (n, k)

    def weighted_sum_on_vector(
        self,
        v: ComplexArray,
        weights: FloatNDArray,
        dagger: bool = False,
    ) -> ComplexArray:
        """
        Compute Σ_j weights[j] * P_j^(†) @ v efficiently without forming Σ_j P_j.

        # Returns a vector of shape (n,).
        """
        if self._k == 0:
            # No projectors: sum is zero vector
            return np.zeros(v.shape[0], dtype=complex)

        w = np.asarray(weights).ravel()
        if w.shape[0] != self._k:
            raise ValueError(f"weights must have length {self._k}.")
        if not np.any(w):
            return np.zeros(self._n if self._n is not None else v.shape[0], dtype=complex)

        if self._is_diagonal:
            M = self.Pdiags.conj() if dagger else self.Pdiags   # (n, k)
            return (M * v[:, None]) @ w                          # (n,)

        if dagger:
            stacked = v @ self.Pconj_stackH
        else:
            stacked = self.P_stackV @ v                                 # (n*k,)
        mat = stacked.reshape((self._n, self._k), order='F')           # (n, k)
        return mat @ w                                                 # (n,)
