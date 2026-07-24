"""Tests for the PSD-boundary penalty's contribution to the dual derivatives.

The penalty term added to the dual is a sum over penalty vectors,

    P(lags) = sum_j v_j^dagger A(lags)^{-1} v_j,

so its gradient and Hessian must be the corresponding sums. The Hessian branch of
``get_dual`` accumulated them outside its loop over vectors, so the penalty
derivatives came from the last vector alone whenever more than one was live. The
value ``dualval_penalty`` was always summed correctly, and so was the gradient on
the ``get_grad``-only path, so only Newton steps were affected.

Two checks, both derived from the definition above rather than from the
implementation, so neither would pass if the loop were restructured incorrectly:
the contributions must be additive over penalty vectors, and the gradient and
Hessian must match finite differences of the penalty value. Both fail on the
unfixed code and neither needs reference data.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from dolphindes.cvxopt import SparseSharedProjQCQP


def _hermitian(rng, n, density=0.4):
    """Return a random sparse Hermitian matrix."""
    dense = rng.random((n, n)) + 1j * rng.random((n, n))
    dense[rng.random((n, n)) > density] = 0.0
    return sp.csc_array(dense + dense.conj().T)


@pytest.fixture(params=["diagonal", "general"])
def qcqp(request):
    """Return a small QCQP that is dual feasible at zero multipliers.

    Parameterized over diagonal and off-diagonal projectors, and carrying one
    general (B_j) constraint, so the mixed projector/general layout of
    ``precomputed_As`` is exercised too. A0 is made strongly positive definite so
    that a feasible point needs no search.
    """
    rng = np.random.default_rng(11)
    n = 9
    A0 = sp.csc_array(np.eye(n) * 40.0 + _hermitian(rng, n).toarray())
    A1 = _hermitian(rng, n)
    A2 = sp.csc_array(np.diag(rng.random(n) + 0.5).astype(complex))
    s0 = rng.random(n) + 1j * rng.random(n)
    s1 = rng.random(n) + 1j * rng.random(n)

    if request.param == "diagonal":
        Plist = [
            sp.diags_array((rng.random(n) + 1j * rng.random(n)), format="csc")
            for _ in range(3)
        ]
        Pstruct = sp.diags_array(np.ones(n, dtype=complex), format="csc")
    else:
        support = rng.random((n, n)) < 0.3
        np.fill_diagonal(support, True)
        Pstruct = sp.csc_array(support.astype(complex))
        Plist = [
            sp.csc_array(
                np.where(support, rng.random((n, n)) + 1j * rng.random((n, n)), 0.0)
            )
            for _ in range(3)
        ]

    return SparseSharedProjQCQP(
        A0,
        s0,
        0.0,
        A1,
        A2,
        s1,
        Plist,
        Pstruct,
        B_j=[_hermitian(rng, n)],
        s_2j=[rng.random(n) + 1j * rng.random(n)],
        c_2j=np.array([0.3]),
        verbose=0,
    )


def test_penalty_derivatives_are_additive_over_vectors(qcqp):
    """Two penalty vectors contribute the sum of their separate contributions.

    On the unfixed code the two-vector gradient and Hessian equal the second
    vector's contribution exactly, ignoring the first.
    """
    rng = np.random.default_rng(5)
    n = qcqp.A0.shape[0]
    lags = np.zeros(qcqp.get_number_constraints())
    assert qcqp.is_dual_feasible(lags), "fixture should be feasible at zero lags"

    v1 = rng.random(n) + 1j * rng.random(n)
    v2 = rng.random(n) + 1j * rng.random(n)

    def penalty_parts(vectors):
        aux = qcqp.get_dual(
            lags, get_grad=True, get_hess=True, penalty_vectors=vectors
        )[3]
        return aux.dualval_penalty, aux.grad_penalty, aux.hess_penalty

    val1, grad1, hess1 = penalty_parts([v1])
    val2, grad2, hess2 = penalty_parts([v2])
    both_val, both_grad, both_hess = penalty_parts([v1, v2])

    # The value was always summed correctly; check it anyway as a control.
    assert np.isclose(both_val, val1 + val2)
    assert np.allclose(both_grad, grad1 + grad2)
    assert np.allclose(both_hess, hess1 + hess2)

    # Without this the additivity checks would also pass for the buggy code in
    # the degenerate case where both vectors happen to contribute equally.
    assert not np.allclose(grad1, grad2)
    assert not np.allclose(hess1, hess2)


def test_penalized_derivatives_match_finite_differences(qcqp):
    """The penalized gradient and Hessian match differences of the dual value.

    A first-principles check that does not assume the penalty decomposes per
    vector: it differences the penalized dual itself. Uses two penalty vectors,
    since the single-vector case was never wrong.
    """
    rng = np.random.default_rng(7)
    n = qcqp.A0.shape[0]
    k = qcqp.get_number_constraints()
    lags = np.full(k, 0.01)
    assert qcqp.is_dual_feasible(lags)
    penalty = [
        rng.random(n) + 1j * rng.random(n),
        rng.random(n) + 1j * rng.random(n),
    ]

    _, grad, hess, _ = qcqp.get_dual(
        lags, get_grad=True, get_hess=True, penalty_vectors=penalty
    )
    assert grad is not None and hess is not None

    step = 1e-6

    def value(x):
        return qcqp.get_dual(x, penalty_vectors=penalty)[0]

    def gradient(x):
        return qcqp.get_dual(
            x, get_grad=True, get_hess=True, penalty_vectors=penalty
        )[1]

    numerical_grad = np.zeros(k)
    numerical_hess = np.zeros((k, k))
    for i in range(k):
        plus, minus = lags.copy(), lags.copy()
        plus[i] += step
        minus[i] -= step
        numerical_grad[i] = (value(plus) - value(minus)) / (2 * step)
        numerical_hess[:, i] = (gradient(plus) - gradient(minus)) / (2 * step)

    assert np.allclose(
        grad, numerical_grad, rtol=1e-5, atol=1e-8 * np.abs(grad).max()
    )
    assert np.allclose(
        hess, numerical_hess, rtol=1e-4, atol=1e-6 * np.abs(hess).max()
    )
