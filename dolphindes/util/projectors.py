"""Module to provide shared projector interface for optimizers."""

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from dolphindes.types import ComplexArray, FloatNDArray


def _detect_all_diagonal(matrix_list: list[sp.csc_array]) -> bool:
    """Return True if every matrix is square and has nonzeros only on the diagonal."""
    return all(
        mat.shape[0] == mat.shape[1]
        and all(r == c for r, c in zip(*mat.nonzero()))
        for mat in matrix_list
    )
    

# Add logic for diagonal and off-diagonal projectors
# Replace qcqp with this class, test with diagonal projectors to make sure it has the same logic
# Then add off-digonal projectors, make tests ensuring the off and on-diagonal projs get the same results
# Allow off-diagonal projectors in QCQP and include a simple test

class Projectors():
    """
    Class to handle sparse shared projectors.

    Parameters
    ----------
    Plist : ArrayLike
        List of sparse projector matrices.
    """

    def __init__(self, Plist: ArrayLike, force_general: bool = False) -> None:
        self.Plist = []
        for P in Plist:
            self.Plist.append(sp.csc_array(P))
        # Validate shapes and store metadata for fast slicing
        if not self.Plist:
            # Allow empty projector list
            self._k = 0
            self._n = None
            self._is_diagonal = True  # vacuously true
            # Nothing else to build (no Pdiags/Pstack*)
            return
        n = self.Plist[0].shape[0]
        if any(P.shape != (n, n) for P in self.Plist):
            raise ValueError("All projectors must be square and have the same shape.")
        self._n = n
        self._k = len(self.Plist)
        self._is_diagonal = _detect_all_diagonal(self.Plist)

        if self._is_diagonal and not force_general:
            self.Pdiags = np.column_stack([P.diagonal() for P in self.Plist])
        else:
            # Build vertical stacks for both P and P^† to avoid runtime transposes
            self.PstackV = sp.vstack(self.Plist, format='csc')                  # shape: (n*k, n)
            self.PstackV_dag = sp.vstack([P.conj().T for P in self.Plist], format='csc')  # (n*k, n)
        del self.Plist  # We do not need the original list

    def _getitem_diagonal(self, key: int) -> sp.csc_array:
        return sp.diags_array(self.Pdiags[:, key], format='csc')

    def _getitem_sparse(self, key: int) -> sp.csc_array:
        idx = key % self._k
        r0 = idx * self._n
        r1 = (idx + 1) * self._n
        # Extract block-rows corresponding to P[idx]
        return self.PstackV[r0:r1, :]

    def __getitem__(self, key: int) -> sp.csc_array:
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

    def _setitem_diagonal(self, key: int, values: ArrayLike) -> None:
        self.Pdiags[:, key] = values

    def _setitem_sparse(self, key: int, value: ArrayLike) -> None:
        idx = key % self._k
        Pnew = sp.csc_array(value, dtype=self.PstackV.dtype)
        if Pnew.shape != (self._n, self._n):
            raise ValueError(f"New projector must have shape ({self._n}, {self._n}).")
        r0 = idx * self._n
        r1 = (idx + 1) * self._n
        # Keep both stacks consistent (store P and its adjoint)
        self.PstackV[r0:r1, :] = Pnew
        self.PstackV_dag[r0:r1, :] = Pnew.conj().T

    def __setitem__(self, key: int, value: ArrayLike) -> None:
        if not isinstance(key, int):
            raise TypeError("Projector index must be an integer.")
        if self._is_diagonal:
            self._setitem_diagonal(key, value)
        else:
            self._setitem_sparse(key, value)

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
        # Use vertical stacks only; avoid runtime transposes
        stacked = (self.PstackV_dag if dagger else self.PstackV) @ v  # (n*k,)
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

        stacked = (self.PstackV_dag if dagger else self.PstackV) @ v   # (n*k,)
        mat = stacked.reshape((self._n, self._k), order='F')           # (n, k)
        return mat @ w                                                 # (n,)
