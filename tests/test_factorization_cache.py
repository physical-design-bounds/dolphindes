"""Tests for the QCQP dual factorization cache (issue #12).

The dominant cost of evaluating the QCQP dual is assembling the total matrix
``A(lags)`` and computing its Cholesky factorization. During dual optimization
the same ``A(lags)`` is factorized several times in a row -- most notably the
line-search feasibility check immediately followed by the objective evaluation
at the same point, and the accepted step, which is re-evaluated at the start of
the next iteration. ``_SharedProjQCQP`` caches recent factorizations keyed
by ``lags`` so those repeats are skipped.

These tests verify that caching:

* is transparent: reusing a factorization gives exactly the result a fresh
  factorization would, and the cache is invalidated whenever ``A(lags)``
  changes; and
* actually removes factorizations: a full dual solve performs measurably fewer
  Cholesky factorizations with the cache enabled.

Setting ``qcqp._factor_cache_maxsize = 0`` disables reuse (every lookup misses),
reproducing legacy behaviour of refactorizing on every dual evaluation; it
is used here as the baseline.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from dolphindes.cvxopt import DenseSharedProjQCQP, SparseSharedProjQCQP


def _reference_dir(kind: str) -> Path:
    return (
        Path(os.path.dirname(__file__))
        / "reference_arrays"
        / "qcqp_example"
        / kind
        / "global"
    )


def _interleaved_projectors(data_path: Path):
    """Rebuild the projector list exactly as the main QCQP fixtures do."""
    projections_diags = np.asarray(
        np.load(data_path / "ldos_some_projections.npy", allow_pickle=True)
    )
    interleaved = np.empty(
        (2 * projections_diags.shape[0], projections_diags.shape[1]), dtype=complex
    )
    interleaved[0::2] = projections_diags
    interleaved[1::2] = projections_diags * -1j
    Pdiags = interleaved.T
    return [sp.diags_array(Pdiags[:, j], format="csc") for j in range(Pdiags.shape[1])]


@pytest.fixture
def sparse_qcqp_builder():
    """Return a zero-argument factory that builds a fresh SparseSharedProjQCQP."""
    data_path = _reference_dir("sparse")
    A0 = sp.csc_array(sp.load_npz(data_path / "ldos_sparse_A0.npz"))
    A1 = sp.csc_array(sp.load_npz(data_path / "ldos_sparse_A1.npz"))
    A2 = sp.csc_array(sp.load_npz(data_path / "ldos_sparse_A2.npz"))
    s0 = np.load(data_path / "ldos_sparse_s0.npy", allow_pickle=True)
    s1 = np.load(data_path / "ldos_sparse_s1.npy", allow_pickle=True)
    c = np.load(data_path / "ldos_dualconst.npy", allow_pickle=True)
    Projlist = _interleaved_projectors(data_path)

    def build():
        return SparseSharedProjQCQP(
            A0, s0, c, A1, A2, s1, Projlist, Projlist[0], verbose=0
        )

    return build


@pytest.fixture
def dense_qcqp_builder():
    """Return a zero-argument factory that builds a fresh DenseSharedProjQCQP."""
    data_path = _reference_dir("dense")
    A0 = np.load(data_path / "ldos_dense_A0.npy")
    A1 = np.load(data_path / "ldos_dense_A1.npy")
    s0 = np.load(data_path / "ldos_dense_s0.npy", allow_pickle=True)
    s1 = np.load(data_path / "ldos_dense_s1.npy", allow_pickle=True)
    c = np.load(data_path / "ldos_dualconst.npy", allow_pickle=True)
    Projlist = _interleaved_projectors(data_path)

    def build():
        return DenseSharedProjQCQP(
            A0, s0, c, A1, s1, Projlist, Projlist[0], None, verbose=0
        )

    return build


def _count_factorizations(qcqp):
    """Patch qcqp._factorize to count invocations; return a mutable counter."""
    counter = {"n": 0}
    original = qcqp._factorize

    def counting(A):
        counter["n"] += 1
        return original(A)

    qcqp._factorize = counting
    return counter


def _solve_counting(build, method, maxsize, init_lags):
    """Solve the dual and return (n_factorizations, solve_result)."""
    np.random.seed(0)
    qcqp = build()
    qcqp._factor_cache_maxsize = maxsize
    qcqp._invalidate_factor_cache()
    counter = _count_factorizations(qcqp)
    # Seed again so the ARPACK starting vector in any PSD-penalty eigsh call is
    # identical between the baseline and cached runs.
    np.random.seed(0)
    result = qcqp.solve_current_dual_problem(method, init_lags=init_lags.copy())
    return counter["n"], result


@pytest.mark.parametrize("builder", ["sparse_qcqp_builder", "dense_qcqp_builder"])
def test_feasibility_check_factorization_is_reused(builder, request):
    """is_dual_feasible(lags) then get_dual(lags) -> a single factorization.

    This is redundancy #1: the line-search feasibility probe factorizes A(lags),
    then the objective evaluation at the same point would refactorize it.
    """
    qcqp = request.getfixturevalue(builder)()
    lags = qcqp.find_feasible_lags()

    qcqp._invalidate_factor_cache()
    counter = _count_factorizations(qcqp)

    assert qcqp.is_dual_feasible(lags)  # factorizes once
    assert counter["n"] == 1
    qcqp.get_dual(lags, get_grad=True)
    assert counter["n"] == 1, "get_dual refactorized after a feasibility check"


@pytest.mark.parametrize("builder", ["sparse_qcqp_builder", "dense_qcqp_builder"])
def test_repeated_get_dual_reuses_factorization(builder, request):
    """get_dual(lags) twice at the same lags factorizes only once."""
    qcqp = request.getfixturevalue(builder)()
    lags = qcqp.find_feasible_lags()

    qcqp._invalidate_factor_cache()
    counter = _count_factorizations(qcqp)

    d0 = qcqp.get_dual(lags, get_grad=True, get_hess=True)
    assert counter["n"] == 1
    d1 = qcqp.get_dual(lags, get_grad=True, get_hess=True)
    assert counter["n"] == 1, "second get_dual at the same lags refactorized"

    # Transparency: the reused factorization yields identical dual/grad/hess.
    assert np.isclose(d0[0], d1[0], rtol=0, atol=0)
    assert np.array_equal(d0[1], d1[1])
    assert np.array_equal(d0[2], d1[2])


@pytest.mark.parametrize("builder", ["sparse_qcqp_builder", "dense_qcqp_builder"])
def test_cache_invalidated_when_constraints_change(builder, request):
    """Rebuilding the constraint data must drop stale cached factorizations."""
    qcqp = request.getfixturevalue(builder)()
    lags = qcqp.find_feasible_lags()

    qcqp._invalidate_factor_cache()
    counter = _count_factorizations(qcqp)

    qcqp.get_dual(lags)
    assert counter["n"] == 1
    qcqp.compute_precomputed_values()
    qcqp.get_dual(lags)
    assert counter["n"] == 2, "cache was not invalidated by compute_precomputed_values"


@pytest.mark.parametrize("builder", ["sparse_qcqp_builder", "dense_qcqp_builder"])
def test_disabling_cache_refactorizes_every_call(builder, request):
    """With maxsize=0 (cache disabled), every evaluation refactorizes."""
    qcqp = request.getfixturevalue(builder)()
    lags = qcqp.find_feasible_lags()

    qcqp._factor_cache_maxsize = 0
    qcqp._invalidate_factor_cache()
    counter = _count_factorizations(qcqp)

    assert qcqp.is_dual_feasible(lags)
    qcqp.get_dual(lags)
    qcqp.get_dual(lags)
    assert counter["n"] == 3, "cache disabled but factorizations were still reused"


@pytest.mark.parametrize("method", ["bfgs", "newton"])
@pytest.mark.parametrize("builder", ["sparse_qcqp_builder", "dense_qcqp_builder"])
def test_cache_reduces_factorizations_in_full_solve(method, builder, request, capsys):
    """A full dual solve does strictly fewer factorizations with the cache on.

    Compares a baseline solve (cache disabled) against a cached solve from the
    same feasible starting point. The result must be unchanged and the number of
    Cholesky factorizations must drop.
    """
    build = request.getfixturevalue(builder)

    np.random.seed(0)
    init_lags = build().find_feasible_lags()

    base_n, base_res = _solve_counting(build, method, 0, init_lags)
    cache_n, cache_res = _solve_counting(build, method, 2, init_lags)

    # 1) Caching must not change the optimum. Reusing a factorization is
    #    bit-identical per evaluation (see the unit tests above); across a full
    #    solve the two trajectories may differ only by floating-point noise, far
    #    below the optimizer's own tolerance (opttol=1e-2), so a tight-but-not
    #    bit-exact tolerance is the meaningful transparency check.
    assert np.isclose(base_res[0], cache_res[0], rtol=1e-6), (
        f"dual value changed: {base_res[0]} vs {cache_res[0]}"
    )
    assert np.allclose(base_res[1], cache_res[1], rtol=1e-5, atol=1e-8), (
        "optimal lags changed with caching enabled"
    )

    # 2) Caching must remove factorizations.
    reduction = (base_n - cache_n) / base_n
    kind = "sparse" if "sparse" in builder else "dense"
    with capsys.disabled():
        print(
            f"\n[{kind:6s} {method:6s}] Cholesky factorizations: "
            f"{base_n} (baseline) -> {cache_n} (cached)  "
            f"= {100 * reduction:.1f}% fewer"
        )
    assert cache_n < base_n, "caching did not reduce the number of factorizations"
    # Conservative floor.
    assert reduction >= 0.05
