"""Tests for sparse and dense QCQP formulations and general constraints.

The tests validate consistency between sparse and dense implementations, check
dual optimization (BFGS and Newton), verify gradients/Hessians against stored
reference arrays, and exercise added general constraints.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from dolphindes.cvxopt import (
    DenseSharedProjQCQP,
    GCDHyperparameters,
    OptimizationHyperparameters,
    SparseSharedProjQCQP,
    gcd,
)
from dolphindes.util import Sym


@pytest.fixture
def data_dir():
    """Return the path to the reference data directory."""
    return (
        Path(os.path.dirname(__file__)) / "reference_arrays" / "qcqp_example" / "sparse"
    )


@pytest.fixture(params=["global", pytest.param("local", marks=pytest.mark.slow)])
def sparse_qcqp_data(data_dir, request):
    """Fixture to load data and initialize SparseSharedProjQCQP."""
    added_str = request.param
    print(f"\nLoading data for: {added_str} constraints")
    data_path = data_dir / added_str
    lags = np.load(data_path / "ldos_sparse_lags.npy", allow_pickle=True)
    lags_optimal = np.load(
        data_path / "ldos_sparse_lags_optimal.npy", allow_pickle=True
    )
    A0 = sp.csc_array(sp.load_npz(data_path / "ldos_sparse_A0.npz"))
    A1 = sp.csc_array(sp.load_npz(data_path / "ldos_sparse_A1.npz"))
    A2 = sp.csc_array(sp.load_npz(data_path / "ldos_sparse_A2.npz"))
    s0 = np.load(data_path / "ldos_sparse_s0.npy", allow_pickle=True)
    s1 = np.load(data_path / "ldos_sparse_s1.npy", allow_pickle=True)
    projections_diags = np.load(
        data_path / "ldos_some_projections.npy", allow_pickle=True
    )
    # Interleave projections_diags and projections_diags * -1j
    projections_diags = np.asarray(projections_diags)
    interleaved = np.empty(
        (2 * projections_diags.shape[0], projections_diags.shape[1]), dtype=complex
    )
    interleaved[0::2] = projections_diags
    interleaved[1::2] = projections_diags * -1j
    projections_diags = interleaved
    Pdiags = projections_diags.T

    c = np.load(data_path / "ldos_dualconst.npy", allow_pickle=True)

    # Build list of diagonal projectors from Pdiags
    Projlist = [
        sp.diags_array(Pdiags[:, j], format="csc") for j in range(Pdiags.shape[1])
    ]

    sparse_ldos_qcqp_instance = SparseSharedProjQCQP(
        A0, s0, c, A1, A2, s1, Projlist, Projlist[0], verbose=0
    )

    return {
        "qcqp": sparse_ldos_qcqp_instance,
        "lags": lags,
        "lags_optimal": lags_optimal,
        "A0": A0,
        "A1": A1,
        "A2": A2,
        "s0": s0,
        "s1": s1,
        "Pdiags": Pdiags,
        "c": c,
        "data_path": data_path,
        "added_str": added_str,
    }


@pytest.fixture
def data_dir_dense():
    """Return the path to the reference data directory."""
    return (
        Path(os.path.dirname(__file__)) / "reference_arrays" / "qcqp_example" / "dense"
    )


@pytest.fixture(params=["global", pytest.param("local", marks=pytest.mark.slow)])
def dense_qcqp_data(data_dir_dense, request):
    """Fixture to load data and initialize DenseSharedProjQCQP."""
    added_str = request.param
    print(f"\nLoading data for: {added_str} constraints")
    data_path = data_dir_dense / added_str
    lags = np.load(data_path / "ldos_dense_lags.npy", allow_pickle=True)
    lags_optimal = np.load(data_path / "ldos_dense_lags_optimal.npy", allow_pickle=True)
    A0 = np.load(data_path / "ldos_dense_A0.npy")
    A1 = np.load(data_path / "ldos_dense_A1.npy")
    s0 = np.load(data_path / "ldos_dense_s0.npy", allow_pickle=True)
    s1 = np.load(data_path / "ldos_dense_s1.npy", allow_pickle=True)
    projections_diags = np.load(
        data_path / "ldos_some_projections.npy", allow_pickle=True
    )
    # Interleave projections_diags and projections_diags * -1j
    projections_diags = np.asarray(projections_diags)
    interleaved = np.empty(
        (2 * projections_diags.shape[0], projections_diags.shape[1]), dtype=complex
    )
    interleaved[0::2] = projections_diags
    interleaved[1::2] = projections_diags * -1j
    projections_diags = interleaved
    Pdiags = projections_diags.T

    c = np.load(data_path / "ldos_dualconst.npy", allow_pickle=True)

    # Build list of diagonal projectors from Pdiags
    Projlist = [
        sp.diags_array(Pdiags[:, j], format="csc") for j in range(Pdiags.shape[1])
    ]

    sparse_ldos_qcqp_instance = DenseSharedProjQCQP(
        A0, s0, c, A1, s1, Projlist, Projlist[0], None, verbose=0
    )

    return {
        "qcqp": sparse_ldos_qcqp_instance,
        "lags": lags,
        "lags_optimal": lags_optimal,
        "A0": A0,
        "A1": A1,
        "A2": None,
        "s0": s0,
        "s1": s1,
        "Pdiags": Pdiags,
        "c": c,
        "data_path": data_path,
        "added_str": added_str,
    }


@pytest.fixture(scope="module")
def dual_results():
    """Fixture to store dual optimization results for comparison."""
    return {}


class TestQCQP:
    """Group of QCQP tests comparing sparse and dense formulations."""

    @pytest.mark.dependency(name="sparse_test")
    def test_sparse_qcqp(self, sparse_qcqp_data, dual_results):
        """Test QCQP optimization vs. stored optimal values.

        This doesn't combine lags into projectors, so gradients and penalties
        should match known reference cases.
        """
        sparse_ldos_qcqp = sparse_qcqp_data["qcqp"]
        lags = sparse_qcqp_data["lags"]
        lags_optimal = sparse_qcqp_data["lags_optimal"]
        data = sparse_qcqp_data["data_path"]
        added_str = sparse_qcqp_data["added_str"]

        print("Testing totalA = known total A")
        ref_totalA = sp.load_npz(data / "ldos_sparse_totalA.npz")
        calc_totalA = sparse_ldos_qcqp._get_total_A(lags)
        assert sp.issparse(calc_totalA), "Total A should be a sparse matrix."
        assert sp.csr_array(calc_totalA).shape == ref_totalA.shape, (
            "Shape of calculated total A does not match reference."
        )
        assert np.allclose(calc_totalA.toarray(), ref_totalA.toarray()), (
            "Calculated total A does not match reference total A."
        )

        print("Testing totalS = known total S")
        ref_totals = np.load(data / "ldos_sparse_total_s.npy", allow_pickle=True)
        # Pass projector multipliers directly
        calc_totals = sparse_ldos_qcqp._get_total_S(lags)
        assert calc_totals.shape == ref_totals.shape, (
            "Shape of calculated total S does not match reference."
        )
        assert np.allclose(calc_totals, ref_totals), (
            "Calculated total S does not match reference total S."
        )

        print(
            "Testing dualval = known dualval and grad = known grad, as well as "
            "penalty vector dualval and grad"
        )
        init_lags = sparse_ldos_qcqp.find_feasible_lags()
        dual, grad, hess, aux = sparse_ldos_qcqp.get_dual(init_lags, get_grad=True)
        assert dual is not None
        print(f"some feasible dual : {dual}")
        feasible = sparse_ldos_qcqp.is_dual_feasible(init_lags)
        assert feasible, "Dual is not feasible."

        dual_opt, grad_opt, hess_opt, aux_opt = sparse_ldos_qcqp.get_dual(
            lags_optimal, get_grad=True, get_hess=False
        )
        assert dual is not None
        print(f"optimal dual : {dual_opt}")
        print(np.load(data / "ldos_dualval.npy"))
        assert np.allclose(dual_opt, np.load(data / "ldos_dualval.npy"), atol=1e-6), (
            "Dual values do not match."
        )
        assert np.allclose(grad_opt, np.load(data / "ldos_optgrad.npy"), atol=1e-6), (
            "Gradients do not match."
        )

        print("Testing dualvalue of penalty vector")
        penalty, value = sparse_ldos_qcqp._get_PSD_penalty(lags_optimal)
        dual_opt_penalty, grad_opt_penalty, hess_opt_penalty, aux_opt_penalty = (
            sparse_ldos_qcqp.get_dual(
                lags_optimal, get_grad=True, penalty_vectors=[penalty]
            )
        )
        assert np.allclose(
            aux_opt_penalty.dualval_penalty,
            np.load(data / "ldos_dualval_penalty.npy"),
            atol=1e-6,
        ), "Penalty dual values do not match."
        assert np.allclose(
            aux_opt_penalty.grad_penalty,
            np.load(data / "ldos_grad_penalty.npy"),
            atol=1e-6,
        ), "Penalty gradients do not match."

        print("Testing solving the dual problem with BFGS and Newton")
        current_dual, dual_lambda, current_grad, current_hess, xstar = (
            sparse_ldos_qcqp.solve_current_dual_problem("bfgs", init_lags=init_lags)
        )

        print(f"dual lambda: {dual_lambda}")
        print(f"bound BFGS: {current_dual}")

        if added_str == "global":
            dual_results["sparse_bfgs_global"] = current_dual
        else:
            dual_results["sparse_bfgs_local"] = current_dual

        assert np.allclose(dual_lambda, lags_optimal, atol=1e-4), (
            "Dual lambda does not match optimal lags."
        )
        assert np.allclose(current_dual, dual_opt, atol=1e-2), (
            "Dual values does not match optimal value."
        )

        current_dual, dual_lambda, current_grad, current_hess, xstar = (
            sparse_ldos_qcqp.solve_current_dual_problem("newton", init_lags=init_lags)
        )
        print(f"dual lambda newton: {dual_lambda}")
        print(f"bound Newton: {current_dual}")

        if added_str == "global":
            dual_results["sparse_newton_global"] = current_dual
        else:
            dual_results["sparse_newton_local"] = current_dual

        assert np.allclose(dual_lambda, lags_optimal, atol=1e-4), (
            "Newton dual lambda does not match optimal lags."
        )
        assert np.allclose(current_dual, dual_opt, atol=1e-2), (
            "Newton dual values does not match optimal value."
        )
        dual_opt, grad_opt, hess_opt, aux_opt = sparse_ldos_qcqp.get_dual(
            dual_lambda, get_grad=True, get_hess=True
        )
        # Hessian values can be large, allow relative tolerance.
        assert np.allclose(hess_opt, np.load(data / "ldos_opthess.npy"), rtol=1e-1), (
            "Hessian does not match optimal Hessian."
        )

    @pytest.mark.slow
    def test_sparse_qcqp_iterative_splitting(self, sparse_qcqp_data):
        """Test iterative splitting step and merging (sparse)."""
        sparse_ldos_qcqp = sparse_qcqp_data["qcqp"]
        added_str = sparse_qcqp_data["added_str"]
        init_lags = sparse_ldos_qcqp.find_feasible_lags()

        # Must solve first to set up state for splitting
        sparse_ldos_qcqp.solve_current_dual_problem("bfgs", init_lags=init_lags)

        print("Testing iterative splitting step (sparse)")
        results = []
        result_counter = 0
        initial_num = sparse_ldos_qcqp.n_proj_constr
        for result in sparse_ldos_qcqp.iterative_splitting_step():
            results.append((sparse_ldos_qcqp.n_proj_constr, result[0]))
            if result_counter > 0:
                # ensure dual is non-increasing as constraints refine
                assert results[result_counter - 1][1] >= result[0], (
                    "Iterative splitting step must decrease dual value."
                )
            result_counter += 1
            # Limit iterations for test runtime
            if result_counter >= 4 or sparse_ldos_qcqp.n_proj_constr > initial_num + 4:
                break
        print(results)

        print("Testing the merging of constraints")
        from dolphindes.cvxopt import gcd

        sparse_ldos_qcqp.compute_precomputed_values()
        dual_opt, dual_grad, dual_hess, _ = sparse_ldos_qcqp.get_dual(
            sparse_ldos_qcqp.current_lags, get_grad=True, get_hess=True
        )

        if added_str == "global":
            cstrt_merge_num = 2
        else:
            cstrt_merge_num = 5
        gcd.merge_lead_constraints(sparse_ldos_qcqp, cstrt_merge_num)
        merged_dual, merged_grad, merged_hess, _ = sparse_ldos_qcqp.get_dual(
            sparse_ldos_qcqp.current_lags, get_grad=True, get_hess=True
        )
        assert np.allclose(dual_opt, merged_dual, atol=1e-2), (
            "dual value changed after constraint merge."
        )
        assert np.allclose(dual_grad[-5:], merged_grad[-5:], atol=1e-2), (
            "dual grad changed after constraint merge."
        )
        assert np.allclose(dual_hess[-5:, -5:], merged_hess[-5:, -5:], rtol=1e-2), (
            "dual Hess changed after constraint merge."
        )

    @pytest.mark.dependency(name="dense_test")
    def test_dense_qcqp(self, dense_qcqp_data, dual_results):
        """Test dense QCQP optimization vs. stored optimal values.

        Does not combine lags into projectors so gradients and penalties
        match known reference cases.
        """
        dense_ldos_qcqp = dense_qcqp_data["qcqp"]
        lags = dense_qcqp_data["lags"]
        lags_optimal = dense_qcqp_data["lags_optimal"]
        data = dense_qcqp_data["data_path"]
        # (No additional scalar constants needed.)

        print("Testing totalA = known total A")
        ref_totalA = np.load(data / "ldos_dense_totalA.npy")
        calc_totalA = dense_ldos_qcqp._get_total_A(lags)
        assert calc_totalA.shape == ref_totalA.shape, (
            "Shape of calculated total A does not match reference."
        )
        assert np.allclose(calc_totalA, ref_totalA, atol=1e-12, rtol=1e-12), (
            "Calculated total A does not match reference total A."
        )

        print("Testing totalS = known total S")
        ref_totals = np.load(data / "ldos_dense_total_s.npy", allow_pickle=True)
        # Pass projector multipliers directly
        calc_totals = dense_ldos_qcqp._get_total_S(lags)
        assert calc_totals.shape == ref_totals.shape, (
            "Shape of calculated total S does not match reference."
        )
        assert np.allclose(calc_totals, ref_totals, atol=1e-12, rtol=1e-12), (
            "Calculated total S does not match reference total S."
        )

        init_dual, init_grad, hess, aux = dense_ldos_qcqp.get_dual(lags, get_grad=True)
        print(init_grad, np.load(data / "ldos_init_grad.npy"))
        assert np.allclose(init_grad, np.load(data / "ldos_init_grad.npy")), (
            "Initial grad does not match reference."
        )

        print(
            "Testing dualval = known dualval and grad = known grad, as well as "
            "penalty vector dualval and grad"
        )
        init_lags = dense_ldos_qcqp.find_feasible_lags()
        dual, grad, hess, aux = dense_ldos_qcqp.get_dual(init_lags, get_grad=True)
        assert dual is not None
        print(f"some feasible dual : {dual}")
        feasible = dense_ldos_qcqp.is_dual_feasible(init_lags)
        assert feasible, "Dual is not feasible."

        dual_opt, grad_opt, hess_opt, aux_opt = dense_ldos_qcqp.get_dual(
            lags_optimal, get_grad=True, get_hess=False
        )
        assert dual is not None
        print(f"optimal dual : {dual_opt}")
        print(np.load(data / "ldos_dualval.npy"))
        assert np.allclose(dual_opt, np.load(data / "ldos_dualval.npy"), atol=1e-6), (
            "Dual values do not match."
        )
        # TODO(alessio): investigate why this is not matching well.
        assert np.allclose(grad_opt, np.load(data / "ldos_grad.npy")), (
            "Gradients do not match."
        )

        print("Testing solving the dual problem with BFGS and Newton")
        current_dual_bfgs, dual_lambda, current_grad, current_hess, xstar = (
            dense_ldos_qcqp.solve_current_dual_problem("bfgs", init_lags=init_lags)
        )

        print(f"dual lambda: {dual_lambda}")
        print(f"bound BFGS: {current_dual_bfgs}")

        assert np.allclose(dual_lambda, lags_optimal, atol=1e-4), (
            "Dual lambda does not match optimal lags."
        )
        assert np.allclose(current_dual_bfgs, dual_opt, atol=1e-2), (
            "Dual values does not match optimal value."
        )

        current_dual_newton, dual_lambda, current_grad, current_hess, xstar = (
            dense_ldos_qcqp.solve_current_dual_problem("newton", init_lags=init_lags)
        )
        print(f"dual lambda newton: {dual_lambda}")
        print(f"bound Newton: {current_dual_newton}")
        if dense_qcqp_data["added_str"] == "global":
            dual_results["dense_bfgs_global"] = current_dual_bfgs
            dual_results["dense_newton_global"] = current_dual_newton
        else:
            dual_results["dense_bfgs_local"] = current_dual_bfgs
            dual_results["dense_newton_local"] = current_dual_newton

        assert np.allclose(dual_lambda, lags_optimal, atol=1e-4), (
            "Newton dual lambda does not match optimal lags."
        )
        assert np.allclose(current_dual_newton, dual_opt, atol=1e-2), (
            "Newton dual values does not match optimal value."
        )
        dual_opt, grad_opt, hess_opt, aux_opt = dense_ldos_qcqp.get_dual(
            dual_lambda, get_grad=True, get_hess=True
        )

    @pytest.mark.slow
    def test_dense_qcqp_iterative_splitting(self, dense_qcqp_data):
        """Test iterative splitting step (dense)."""
        dense_ldos_qcqp = dense_qcqp_data["qcqp"]
        init_lags = dense_ldos_qcqp.find_feasible_lags()

        # Must solve first to set up state for splitting
        dense_ldos_qcqp.solve_current_dual_problem("bfgs", init_lags=init_lags)

        print("Testing iterative splitting step (dense)")
        results = []
        result_counter = 0
        initial_num = dense_ldos_qcqp.n_proj_constr
        for result in dense_ldos_qcqp.iterative_splitting_step():
            results.append((dense_ldos_qcqp.n_proj_constr, result[0]))
            if result_counter > 0:
                assert results[result_counter - 1][1] >= result[0], (
                    "Iterative splitting step must decrease dual value."
                )
            result_counter += 1
            if result_counter >= 4 or dense_ldos_qcqp.n_proj_constr > initial_num + 4:
                break
        print(results)

    @pytest.mark.dependency(depends=["sparse_test", "dense_test"])
    def test_compare_dual_results(self, dual_results):
        """Compare dual results from BFGS and Newton (sparse vs. dense)."""
        print("Comparing dual results from BFGS and Newton methods:")
        for key, value in dual_results.items():
            print(f"{key}: {value}")

        # Global results (should always run unless explicitly deselected)
        if "sparse_bfgs_global" in dual_results and "dense_bfgs_global" in dual_results:
            assert np.allclose(
                dual_results["sparse_bfgs_global"],
                dual_results["dense_bfgs_global"],
                atol=1e-2,
            ), "BFGS global results for sparse and dense do not match."

        if (
            "sparse_newton_global" in dual_results
            and "dense_newton_global" in dual_results
        ):
            assert np.allclose(
                dual_results["sparse_newton_global"],
                dual_results["dense_newton_global"],
                atol=1e-2,
            ), "Newton global results for sparse and dense do not match."

        # Local results (skipped if marked slow and running fast tests)
        if "sparse_bfgs_local" in dual_results and "dense_bfgs_local" in dual_results:
            assert np.allclose(
                dual_results["sparse_bfgs_local"],
                dual_results["dense_bfgs_local"],
                atol=1e-2,
            ), "BFGS local results for sparse and dense do not match."

        if (
            "sparse_newton_local" in dual_results
            and "dense_newton_local" in dual_results
        ):
            assert np.allclose(
                dual_results["sparse_newton_local"],
                dual_results["dense_newton_local"],
                atol=1e-2,
            ), "Newton local results for sparse and dense do not match."


def test_sparse_qcqp_with_general_constraint(sparse_qcqp_data):
    """Add one general constraint (||A2 x|| term) to sparse formulation."""
    base = sparse_qcqp_data
    A0, A1, A2 = base["A0"], base["A1"], base["A2"]
    s0, s1, Pdiags, c = base["s0"], base["s1"], base["Pdiags"], base["c"]
    n_rows_A2 = A2.shape[0]
    B_j = [sp.eye_array(n_rows_A2, format="csc")]
    s_2j = [np.zeros(n_rows_A2, dtype=complex)]
    c_2j = np.array([1.0])
    # Build list of diagonal projectors from Pdiags
    Projlist = [
        sp.diags_array(Pdiags[:, j], format="csc") for j in range(Pdiags.shape[1])
    ]
    qcqp_gc = SparseSharedProjQCQP(
        A0, s0, c, A1, A2, s1, Projlist, None, B_j=B_j, s_2j=s_2j, c_2j=c_2j, verbose=0
    )
    init_lags = qcqp_gc.find_feasible_lags()
    dual, lags, grad, hess, xstar = qcqp_gc.solve_current_dual_problem(
        "bfgs", init_lags=init_lags
    )
    print(f"[Sparse + general constraint] dual value: {dual}")


def test_dense_qcqp_with_general_constraint(dense_qcqp_data):
    """Add one general constraint (||A2 x|| term) to dense formulation."""
    base = dense_qcqp_data
    A0, A1 = base["A0"], base["A1"]
    s0, s1, Pdiags, c = base["s0"], base["s1"], base["Pdiags"], base["c"]
    n_dim = s0.shape[0]
    B_j = [np.eye(n_dim, dtype=complex)]
    s_2j = [np.zeros(n_dim, dtype=complex)]
    c_2j = np.array([1.0])  # per user request
    # Build list of diagonal projectors from Pdiags
    Projlist = [
        sp.diags_array(Pdiags[:, j], format="csc") for j in range(Pdiags.shape[1])
    ]
    qcqp_gc = DenseSharedProjQCQP(
        A0,
        s0,
        c,
        A1,
        s1,
        Projlist,
        None,
        A2=None,
        B_j=B_j,
        s_2j=s_2j,
        c_2j=c_2j,
        verbose=0,
    )
    init_lags = qcqp_gc.find_feasible_lags()
    dual, lags, grad, hess, xstar = qcqp_gc.solve_current_dual_problem(
        "bfgs", init_lags=init_lags
    )
    print(f"[Dense + general constraint] dual value: {dual}")


def test_dense_qcqp_only_general_constraint():
    """Solve tiny dense QCQP: maximize x1 + x2 s.t. ||x||^2 = 1."""
    # Problem data
    A0 = np.zeros((2, 2), dtype=complex)
    A1 = np.zeros((2, 2), dtype=complex)  # Unused (no projectors)
    s0 = 0.5 * np.ones(2, dtype=complex)  # Gives x1 + x2
    c0 = 0.0
    s1 = np.zeros(2, dtype=complex)  # No projector linear term
    # Pdiags = np.zeros((2, 0), dtype=complex)    # No shared projector constraints
    Projlist = []  # No shared projector constraints

    # General constraint: ||x||^2 = 1
    B_j = [np.eye(2, dtype=complex)]
    s_2j = [np.zeros(2, dtype=complex)]
    c_2j = np.array([1.0])  # -x^T x + 1 = 0

    n_dim = s0.shape[0]
    # dummy structure for empty projector set
    Pstruct = sp.eye_array(n_dim, format="csc")

    qcqp = DenseSharedProjQCQP(
        A0,
        s0,
        c0,
        A1,
        s1,
        Projlist,
        Pstruct,
        A2=None,
        B_j=B_j,
        s_2j=s_2j,
        c_2j=c_2j,
        verbose=0,
    )

    # Provide manual initial lag (one general constraint only)
    init_lags = np.array([1.0])
    dual_opt, lags_opt, grad_opt, hess_opt, xstar = qcqp.solve_current_dual_problem(
        "bfgs", init_lags=init_lags
    )

    expected_val = np.sqrt(2)
    expected_mu = 1 / np.sqrt(2)

    # Primal value from x*
    primal_val = np.real(np.sum(xstar))

    assert np.isclose(lags_opt[0], expected_mu, rtol=1e-4, atol=1e-6), (
        "Multiplier not optimal."
    )
    assert np.isclose(dual_opt, expected_val, rtol=1e-4, atol=1e-6), (
        "Dual optimum incorrect."
    )
    assert np.isclose(primal_val, expected_val, rtol=1e-4, atol=1e-6), (
        "Primal optimum incorrect."
    )
    assert np.linalg.norm(grad_opt) < 1e-3, "Gradient not (near) zero at optimum."


def test_sparse_qcqp_with_nondiagonal_projectors():
    """Sparse QCQP with non-diagonal (Hermitian) projectors runs end-to-end."""
    n = 4
    # Non-diagonal Hermitian projectors: P = u u^H
    u1 = (np.eye(n)[:, 0] + np.eye(n)[:, 1]) / np.sqrt(2)  # (e0 + e1)/sqrt(2)
    u2 = (np.eye(n)[:, 2] + 1j * np.eye(n)[:, 3]) / np.sqrt(2)  # (e2 + i e3)/sqrt(2)
    P1 = np.outer(u1, u1.conj())
    P2 = np.outer(u2, u2.conj())
    Projlist = [sp.csc_array(P1), sp.csc_array(P2)]

    # Build Pstruct as union of projector sparsity patterns
    Pstruct = Projlist[0] != 0
    for P in Projlist[1:]:
        Pstruct = Pstruct.maximum(P != 0)
    Pstruct = sp.csc_array(Pstruct)

    # Simple PSD setup
    A0 = sp.eye_array(n, format="csc")
    A1 = sp.eye_array(n, format="csc")
    A2 = sp.eye_array(n, format="csc")
    rng = np.random.default_rng(0)
    s0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    s1 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    c0 = 0.0

    qcqp = SparseSharedProjQCQP(A0, s0, c0, A1, A2, s1, Projlist, Pstruct, verbose=0)

    lags0 = qcqp.find_feasible_lags()
    dual, grad, _, _ = qcqp.get_dual(lags0, get_grad=True)
    assert np.isfinite(dual)
    assert grad.shape[0] == len(Projlist)
    assert np.all(np.isfinite(grad))

    dual_opt, lags_opt, grad_opt, _, xstar = qcqp.solve_current_dual_problem(
        "bfgs", init_lags=lags0
    )
    assert np.isfinite(dual_opt)
    assert lags_opt.shape[0] == len(Projlist)
    assert grad_opt is not None and grad_opt.shape[0] == len(Projlist)
    assert xstar.shape[0] == n


def test_dense_qcqp_with_nondiagonal_projectors():
    """Dense QCQP with non-diagonal (Hermitian) projectors runs end-to-end."""
    n = 4
    # Non-diagonal Hermitian projectors: P = u u^H
    u1 = (np.eye(n)[:, 0] + np.eye(n)[:, 1]) / np.sqrt(2)
    u2 = (np.eye(n)[:, 2] + 1j * np.eye(n)[:, 3]) / np.sqrt(2)
    P1 = np.outer(u1, u1.conj())
    P2 = np.outer(u2, u2.conj())
    Projlist = [sp.csc_array(P1), sp.csc_array(P2)]

    # Build Pstruct as union of projector sparsity patterns
    Pstruct = Projlist[0] != 0
    for P in Projlist[1:]:
        Pstruct = Pstruct.maximum(P != 0)
    Pstruct = sp.csc_array(Pstruct)

    A0 = np.eye(n, dtype=complex)
    A1 = np.eye(n, dtype=complex)
    rng = np.random.default_rng(1)
    s0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    s1 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    c0 = 0.0

    qcqp = DenseSharedProjQCQP(
        A0, s0, c0, A1, s1, Projlist, Pstruct, A2=None, verbose=0
    )

    lags0 = qcqp.find_feasible_lags()
    dual, grad, _, _ = qcqp.get_dual(lags0, get_grad=True)
    assert np.isfinite(dual)
    assert grad.shape[0] == len(Projlist)
    assert np.all(np.isfinite(grad))

    dual_opt, lags_opt, grad_opt, _, xstar = qcqp.solve_current_dual_problem(
        "bfgs", init_lags=lags0
    )
    assert np.isfinite(dual_opt)
    assert lags_opt.shape[0] == len(Projlist)
    assert grad_opt is not None and grad_opt.shape[0] == len(Projlist)
    assert xstar.shape[0] == n


def _interior_feasible_lags(qcqp):
    """A dual-feasible multiplier vector pushed slightly interior so that
    A(lags) is comfortably positive definite for the Cholesky solves."""
    lags = np.array(qcqp.find_feasible_lags(), dtype=float)
    lags[1] *= 2.0
    return lags


def _assert_hs_orthonormal(qcqp, atol=1e-7):
    """The projector constraints should be orthonormal in the augmented HS metric."""
    gram = gcd._hs_gram_matrix(qcqp)
    assert np.allclose(gram, np.eye(qcqp.n_proj_constr), atol=atol), (
        "constraint basis is not HS-orthonormal"
    )


def _check_hs_invariants(qcqp):
    """Every HS-metric basis operation preserves the dual value and keeps the
    constraint basis HS-orthonormal.

    Exercises the full sequence a GCD run relies on: initial re-orthonormalization,
    then add_constraints, then merge_lead_constraints. Each step's assertion
    message identifies where a regression occurred.
    """
    qcqp.current_lags = _interior_feasible_lags(qcqp)

    # (1) Re-orthonormalization is a pure basis change: dual invariant.
    dual = qcqp.get_dual(qcqp.current_lags)[0]
    gcd._orthonormalize_hs(qcqp)
    assert np.isclose(dual, qcqp.get_dual(qcqp.current_lags)[0], rtol=1e-6, atol=1e-7), (
        "HS orthonormalization changed the dual value"
    )
    _assert_hs_orthonormal(qcqp)

    # (2) Adding constraints enters them with multiplier 0, so the dual is unchanged.
    rng = np.random.default_rng(0)
    nnz = qcqp.Proj.Pstruct.size
    new = [rng.standard_normal(nnz) + 1j * rng.standard_normal(nnz) for _ in range(2)]
    dual = qcqp.get_dual(qcqp.current_lags)[0]
    qcqp.add_constraints(new, orthonormalize=True, metric="hilbert_schmidt")
    assert np.isclose(dual, qcqp.get_dual(qcqp.current_lags)[0], rtol=1e-6, atol=1e-7), (
        "HS add_constraints changed the dual value"
    )
    _assert_hs_orthonormal(qcqp)

    # (3) Merging leading constraints preserves the dual (multiplier is rescaled).
    dual = qcqp.get_dual(qcqp.current_lags)[0]
    qcqp.merge_lead_constraints(merged_num=2, metric="hilbert_schmidt")
    assert np.isclose(dual, qcqp.get_dual(qcqp.current_lags)[0], rtol=1e-6, atol=1e-6), (
        "HS merge_lead_constraints changed the dual value"
    )
    _assert_hs_orthonormal(qcqp)


def test_hs_invariants_sparse(sparse_qcqp_data):
    """HS ortho/add/merge preserve the dual + orthonormality (sparse)."""
    _check_hs_invariants(sparse_qcqp_data["qcqp"])


def test_hs_invariants_dense(dense_qcqp_data):
    """HS ortho/add/merge preserve the dual + orthonormality (dense)."""
    _check_hs_invariants(dense_qcqp_data["qcqp"])


def _general_projector_qcqp(with_duplicate=False, n=5, seed=3):
    """A sparse QCQP with full-support rank-1 (non-diagonal) projectors, on which
    ``_hs_analytic_available`` is False so the operator-based HS paths run.
    Optionally appends a duplicate projector to make the set HS-dependent."""
    rng = np.random.default_rng(seed)
    vs = [rng.standard_normal(n) + 1j * rng.standard_normal(n) for _ in range(4)]
    Projlist = [sp.csc_array(np.outer(v, v.conj())) for v in vs]
    if with_duplicate:
        Projlist.append(sp.csc_array(np.outer(vs[0], vs[0].conj())))
    Pstruct = sp.csc_array(np.ones((n, n), dtype=complex))

    A0 = sp.csc_array(sp.eye_array(n, format="csc") * 2.0)
    A1 = sp.eye_array(n, format="csc")
    A2 = sp.eye_array(n, format="csc")
    s0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    s1 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return SparseSharedProjQCQP(A0, s0, 0.0, A1, A2, s1, Projlist, Pstruct, verbose=0)


def test_hs_invariants_general_projectors():
    """HS ortho/add/merge preserve the dual + orthonormality for general
    (non-diagonal) projectors, i.e. through the operator-based paths."""
    qcqp = _general_projector_qcqp()
    assert not gcd._hs_analytic_available(qcqp)
    _check_hs_invariants(qcqp)


def test_hs_orthonormalization_ridge_fallback():
    """An HS-dependent projector set (duplicate constraint) exercises the ridge
    fallback in _orthonormalize_hs, and the dual is still preserved."""
    qcqp = _general_projector_qcqp(with_duplicate=True)

    lags = _interior_feasible_lags(qcqp)
    qcqp.current_lags = lags.copy()
    dual_before = qcqp.get_dual(lags)[0]
    gcd._orthonormalize_hs(qcqp)
    dual_after = qcqp.get_dual(qcqp.current_lags)[0]
    assert np.isclose(dual_before, dual_after, rtol=1e-6, atol=1e-7)


def test_run_gcd_hilbert_schmidt_runs(sparse_qcqp_data):
    """run_gcd with the HS metric runs end-to-end and yields a finite bound."""
    params = GCDHyperparameters(
        ortho_metric="hilbert_schmidt",
        max_gcd_iter_num=6,
        max_proj_cstrt_num=6,
        gcd_iter_period=5,
        gcd_tol=1e-3,
    )
    qcqp = sparse_qcqp_data["qcqp"]
    qcqp.run_gcd(params)
    assert qcqp.current_dual is not None and np.isfinite(qcqp.current_dual)


def _record_gcd_inner_methods(qcqp, params, monkeypatch):
    """Run GCD and return the ``method`` argument of every inner dual solve."""
    seen = []
    original = type(qcqp).solve_current_dual_problem

    def spy(self, method, *args, **kwargs):
        seen.append(method)
        return original(self, method, *args, **kwargs)

    monkeypatch.setattr(type(qcqp), "solve_current_dual_problem", spy)
    qcqp.run_gcd(params)
    return seen


@pytest.mark.parametrize("method, expected", [(None, "newton"), ("bfgs", "bfgs")])
def test_run_gcd_inner_solve_method(sparse_qcqp_data, monkeypatch, method, expected):
    """GCDHyperparameters.method selects the inner dual optimizer; the default
    (an unset field) still routes to Newton."""
    kwargs = {} if method is None else {"method": method}
    params = GCDHyperparameters(
        max_gcd_iter_num=2,
        max_proj_cstrt_num=6,
        gcd_iter_period=5,
        gcd_tol=1e-3,
        **kwargs,
    )
    qcqp = sparse_qcqp_data["qcqp"]
    seen = _record_gcd_inner_methods(qcqp, params, monkeypatch)

    assert seen and set(seen) == {expected}
    assert qcqp.current_dual is not None and np.isfinite(qcqp.current_dual)


@pytest.mark.parametrize("method", ["bfgs", "newton"])
def test_solve_dual_reports_smallest_dual_without_penalties(
    sparse_qcqp_data, monkeypatch, method
):
    """current_dual is the smallest penalty-free dual over every evaluation, and the
    lags, gradient, Hessian and xstar returned with it are those where it was found.

    Uses run_gcd's inner-solve settings (one penalty level, max_restart=1), under
    which the optimizer stops with its PSD-boundary penalties still active.
    """
    qcqp = sparse_qcqp_data["qcqp"]
    seen = []
    original = type(qcqp).get_dual

    def spy(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        seen.append(out[3].dualval_real)
        return out

    opt_params = OptimizationHyperparameters(
        opttol=1e-2,
        min_inner_iter=5,
        max_restart=1,
        penalty_ratio=1e-2,
        penalty_reduction=0.1,
        break_iter_period=20,
    )
    init_lags = qcqp.find_feasible_lags()
    monkeypatch.setattr(type(qcqp), "get_dual", spy)
    _, _, _, hess, _ = qcqp.solve_current_dual_problem(
        method, init_lags=init_lags, opt_params=opt_params
    )
    monkeypatch.undo()

    assert qcqp.current_dual == min(seen)
    dual, grad, hess_at_lags, _ = qcqp.get_dual(
        qcqp.current_lags, get_grad=True, get_hess=method == "newton"
    )
    assert qcqp.current_dual == dual
    assert np.array_equal(qcqp.current_grad, grad)
    if method == "newton":
        assert np.array_equal(hess, hess_at_lags)
    else:
        assert hess is None
    assert np.allclose(qcqp.current_xstar, qcqp._get_xstar(qcqp.current_lags)[0])


@pytest.mark.parametrize("method", ["bfgs", "newton"])
def test_run_gcd_reported_bound_never_increases(sparse_qcqp_data, capsys, method):
    """The bound run_gcd reports after each iteration is never above the previous
    one, and it leaves current_dual as the dual at current_lags."""
    params = GCDHyperparameters(
        max_gcd_iter_num=8,
        max_proj_cstrt_num=6,
        gcd_iter_period=5,
        gcd_tol=1e-3,
        method=method,
    )
    qcqp = sparse_qcqp_data["qcqp"]

    # restore the global random state so changing test order doesn't change tests
    state = np.random.get_state()
    np.random.seed(3)
    try:
        qcqp.run_gcd(params)
    finally:
        np.random.set_state(state)
    reported = [
        float(m)
        for m in re.findall(
            r"best dual bound found is\s+(\S+)\.\n", capsys.readouterr().out
        )
    ]

    assert len(reported) >= 2
    assert all(b <= a for a, b in zip(reported, reported[1:]))
    assert qcqp.current_dual == qcqp.get_dual(qcqp.current_lags)[0]


def test_minAeig_constraint_raises_smallest_eigenvalue(sparse_qcqp_data):
    """The smallest-eigenvector constraint must move away from the PSD boundary.

    Reproduces run_gcd's construction of that constraint and checks the property it
    exists for: turning up its multiplier from 0 increases the smallest eigenvalue
    of A(lags).
    """
    qcqp = sparse_qcqp_data["qcqp"]
    lags = _interior_feasible_lags(qcqp)

    v, _w = qcqp._get_PSD_penalty(lags)
    v = v / np.linalg.norm(v)
    rows, cols = qcqp.Proj.Pstruct.nonzero()
    pdata = (qcqp.A1.conj().T @ v)[rows] * (qcqp.A2 @ v).conj()[cols]
    pdata = pdata / np.linalg.norm(pdata)

    P = qcqp.Proj.Pstruct.astype(complex, copy=True)
    P.data = pdata
    A_new = Sym(qcqp.A1 @ P @ qcqp.A2)

    A_base = qcqp._get_total_A(lags)
    A_base = A_base.toarray() if sp.issparse(A_base) else np.asarray(A_base)
    A_new = A_new.toarray() if sp.issparse(A_new) else np.asarray(A_new)

    def smallest_eig(t):
        return float(np.linalg.eigvalsh(A_base + t * A_new)[0])

    # Hellmann-Feynman: d(eta)/dt at t=0 is v^H A_new v, and it must be positive.
    analytic = float(np.real(v.conj() @ (A_new @ v)))
    assert analytic > 0, "min-eig constraint steers toward the PSD boundary, not away"

    h = 1e-6 * max(1.0, abs(smallest_eig(0.0)))
    numeric = (smallest_eig(h) - smallest_eig(-h)) / (2 * h)
    assert np.isclose(analytic, numeric, rtol=1e-4, atol=1e-8), (
        f"eigenvalue derivative mismatch: analytic {analytic}, numeric {numeric}"
    )


def test_invalid_gcd_method_raises():
    """An unknown inner-solve method is rejected at construction time."""
    with pytest.raises(ValueError, match="method"):
        GCDHyperparameters(method="quasi-newton")


def _dense_diag_qcqp(n=24, k=5, seed=7):
    """A dense QCQP with diagonal projectors (partial support) for the closed-form
    HS path. Returns a DenseSharedProjQCQP on which ``_hs_analytic_available`` is
    True."""
    rng = np.random.default_rng(seed)

    def rc(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    M = rc((n, n))
    A0 = M.conj().T @ M + n * np.eye(n)  # Hermitian PD
    A1 = rc((n, n))
    s0 = rc(n)
    s1 = rc(n)
    Pdiags = rc((n, k))
    Pdiags[rng.random((n, k)) < 0.3] = 0.0  # sparse, unequal projector supports
    Projlist = [sp.diags_array(Pdiags[:, j], format="csc") for j in range(k)]
    Pstruct = Projlist[0] != 0
    for P in Projlist[1:]:
        Pstruct = Pstruct.maximum(P != 0)
    return DenseSharedProjQCQP(
        A0, s0, 0.0, A1, s1, Projlist, sp.csc_array(Pstruct), A2=None, verbose=0
    )


def _assert_hs_analytic_matches_loop(qcqp, monkeypatch):
    """The closed-form HS Gram matrix and new-constraint orthonormalization match
    the operator-based double-loop fallback to machine precision."""
    assert gcd._hs_analytic_available(qcqp)

    # (1) Gram matrix: analytic vs forced loop.
    gram_analytic = gcd._hs_gram_matrix(qcqp)
    monkeypatch.setattr(gcd, "_hs_analytic_available", lambda _q: False)
    gram_loop = gcd._hs_gram_matrix(qcqp)
    monkeypatch.undo()
    assert np.allclose(gram_analytic, gram_loop, atol=1e-8)

    # (2) Orthonormalizing new constraints against the HS-orthonormal basis.
    qcqp.current_lags = qcqp.find_feasible_lags()
    gcd._orthonormalize_hs(qcqp)
    rng = np.random.default_rng(1)
    nnz = qcqp.Proj.Pstruct.size
    new_analytic = [rng.standard_normal(nnz) + 1j * rng.standard_normal(nnz)
                    for _ in range(2)]
    new_loop = [a.copy() for a in new_analytic]

    gcd._orthonormalize_new_hs(qcqp, new_analytic)  # analytic path
    monkeypatch.setattr(gcd, "_hs_analytic_available", lambda _q: False)
    gcd._orthonormalize_new_hs(qcqp, new_loop)  # op-based fallback
    monkeypatch.undo()
    for a, b in zip(new_analytic, new_loop):
        assert np.allclose(a, b, atol=1e-8)


def test_hs_analytic_matches_loop_dense(monkeypatch):
    """Dense formulation: closed-form HS matches the operator loop."""
    _assert_hs_analytic_matches_loop(_dense_diag_qcqp(), monkeypatch)


def test_hs_analytic_matches_loop_sparse(sparse_qcqp_data, monkeypatch):
    """Sparse formulation: closed-form HS (sparse kernels) matches the loop."""
    _assert_hs_analytic_matches_loop(sparse_qcqp_data["qcqp"], monkeypatch)


def test_hs_kernel_cache_reused():
    """Kernels are built once and cached (they depend only on A1, A2, s1)."""
    qcqp = _dense_diag_qcqp()
    k1 = gcd._hs_kernels(qcqp)
    k2 = gcd._hs_kernels(qcqp)
    for a, b in zip(k1, k2):
        assert a is b


def _elementwise_total_A(qcqp, lags):
    """A(lags) summed one constraint at a time, i.e. without the assembly map."""
    return qcqp.A0 + sum(lags[i] * qcqp.precomputed_As[i] for i in range(len(lags)))


def _assert_total_A_matches_elementwise(qcqp, lags, atol=1e-12):
    got = sp.csc_array(qcqp._get_total_A(lags))
    want = sp.csc_array(_elementwise_total_A(qcqp, lags))
    assert abs(got - want).max() <= atol * abs(want).max()


def _sparse_qcqp_for_assembly(diagonal, n_gen, n=12, k=4, seed=5):
    """A sparse QCQP with diagonal or non-diagonal projectors and optional general
    constraints, for exercising the fixed-pattern assembly of A(lags)."""
    rng = np.random.default_rng(seed)

    def rc(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    if diagonal:
        # unequal, partly overlapping supports, as refine_projectors produces
        diags = rc((n, k))
        diags[rng.random((n, k)) < 0.4] = 0.0
        Projlist = [sp.diags_array(diags[:, j], format="csc") for j in range(k)]
    else:
        Projlist = [
            sp.csc_array(np.outer(v, v.conj())) for v in (rc(n) for _ in range(k))
        ]
    Pstruct = Projlist[0] != 0
    for P in Projlist[1:]:
        Pstruct = Pstruct.maximum(P != 0)

    A0 = sp.csc_array(sp.eye_array(n, format="csc") * 3.0)
    return SparseSharedProjQCQP(
        A0,
        rc(n),
        0.0,
        sp.csc_array(rc((n, n))),
        sp.csc_array(rc((n, n))),
        rc(n),
        Projlist,
        sp.csc_array(Pstruct),
        B_j=[sp.csc_array(rc((n, n))) for _ in range(n_gen)] or None,
        s_2j=[rc(n) for _ in range(n_gen)] or None,
        c_2j=rng.standard_normal(n_gen) if n_gen else None,
        verbose=0,
    )


def _real_sparse_qcqp_for_assembly(n=10, k=3, seed=8):
    """A sparse QCQP whose quadratic parts are real. A0 keeps the dtype it is given on
    the sparse path, so the assembly map must not promote such a problem to complex."""
    rng = np.random.default_rng(seed)
    Projlist = [sp.diags_array(rng.standard_normal(n), format="csc") for _ in range(k)]
    Pstruct = sp.csc_array(sp.eye_array(n, format="csc"))
    return SparseSharedProjQCQP(
        sp.csc_array(sp.eye_array(n, format="csc") * 3.0),
        rng.standard_normal(n),
        0.0,
        sp.csc_array(rng.standard_normal((n, n))),
        sp.csc_array(rng.standard_normal((n, n))),
        rng.standard_normal(n),
        Projlist,
        Pstruct,
        verbose=0,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wall_step_finds_psd_boundary(seed):
    """_wall_step gives the step along d at which A(lags + t d) stops being positive
    definite: the exact one, from the dense generalized eigenproblem, on a small QCQP."""
    qcqp = _sparse_qcqp_for_assembly(diagonal=True, n_gen=0)
    rng = np.random.default_rng(seed)
    k = qcqp.get_number_constraints()
    lags = 1e-3 * rng.standard_normal(k)  # A0 = 3I dominates: interior
    d = rng.standard_normal(k)
    A = qcqp._get_total_A(lags).toarray()
    B = qcqp._get_total_A(lags + d).toarray() - A
    mu = la.eigh(B, A, eigvals_only=True)[0]
    assert mu < 0, "the test needs a direction that leaves the cone"

    t = qcqp._wall_step(lags, d)

    assert t is not None and np.isclose(t, -1.0 / mu, rtol=1e-2)
    assert qcqp.is_dual_feasible(lags + 0.9 * t * d)
    assert not qcqp.is_dual_feasible(lags + 1.1 * t * d)


def test_is_dual_feasible_refuses_small_indefinite():
    """A small indefinite A(lags) is infeasible, and a small PD one feasible.

    CHOLMOD's "auto" analysis factors a matrix this small as a simplicial LDL^H,
    which succeeds on an indefinite matrix; the supernodal analysis
    SparseSharedProjQCQP uses raises instead.
    """
    qcqp = _sparse_qcqp_for_assembly(diagonal=True, n_gen=0)
    d = np.random.default_rng(4).standard_normal(qcqp.get_number_constraints())
    assert qcqp.is_dual_feasible(1e-3 * d)  # A0 = 3I dominates
    t = 1.0
    while np.linalg.eigvalsh(qcqp._get_total_A(t * d).toarray())[0] > -1.0:
        t *= 2.0
    assert not qcqp.is_dual_feasible(t * d)


@pytest.mark.parametrize("diagonal", [True, False])
@pytest.mark.parametrize("n_gen", [0, 2])
def test_assembly_map_matches_elementwise_sum(diagonal, n_gen):
    """A(lags) assembled on the fixed pattern equals the elementwise sum, at more than
    one lags (the second reuses the cached map)."""
    qcqp = _sparse_qcqp_for_assembly(diagonal, n_gen)
    n_constr = qcqp.get_number_constraints()
    rng = np.random.default_rng(2)
    _assert_total_A_matches_elementwise(qcqp, rng.standard_normal(n_constr))
    assert qcqp._assembly_map is not None, "expected a fixed-pattern map here"
    _assert_total_A_matches_elementwise(qcqp, rng.standard_normal(n_constr))


@pytest.mark.parametrize("diagonal", [True, False])
@pytest.mark.parametrize("n_gen", [0, 2])
def test_assembly_map_matches_elementwise_derivatives(diagonal, n_gen):
    """The two assembly paths agree on what callers actually consume: the dual value,
    its gradient and its Hessian, penalty terms included."""
    qcqp = _sparse_qcqp_for_assembly(diagonal, n_gen)
    rng = np.random.default_rng(3)
    n = qcqp.A0.shape[0]
    # A0 = 3I dominates at this scale, so a small point is interior
    lags = 1e-3 * rng.standard_normal(qcqp.get_number_constraints())
    assert qcqp.is_dual_feasible(lags)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    kwargs = dict(get_grad=True, get_hess=True, penalty_vectors=[1e-3 * v])

    dual_map, grad_map, hess_map, aux_map = qcqp.get_dual(lags, **kwargs)
    assert qcqp._assembly_map is not None, "expected a fixed-pattern map here"

    qcqp._assembly_map_max_bytes = 1  # force the elementwise sum
    qcqp._invalidate_factor_cache()
    dual_el, grad_el, hess_el, aux_el = qcqp.get_dual(lags, **kwargs)
    assert qcqp._assembly_map is None

    assert np.isclose(dual_map, dual_el, rtol=1e-11)
    assert np.allclose(grad_map, grad_el, rtol=1e-9, atol=1e-11)
    assert np.allclose(hess_map, hess_el, rtol=1e-9, atol=1e-11)
    assert np.allclose(aux_map.grad_penalty, aux_el.grad_penalty, rtol=1e-9, atol=1e-11)
    assert np.allclose(aux_map.hess_penalty, aux_el.hess_penalty, rtol=1e-9, atol=1e-11)


def test_assembly_map_preserves_real_dtype():
    """A real-valued problem must stay real: promoting A(lags) to complex would double
    CHOLMOD's memory and factorization cost."""
    qcqp = _real_sparse_qcqp_for_assembly()
    lags = np.abs(qcqp.find_feasible_lags())
    assert qcqp.A0.dtype == np.float64  # else the fixture is not exercising this
    assert all(Ak.dtype == np.float64 for Ak in qcqp.precomputed_As)

    got = qcqp._get_total_A(lags)
    assert qcqp._assembly_map is not None, "expected a fixed-pattern map here"
    want = _elementwise_total_A(qcqp, lags)
    assert got.dtype == want.dtype == np.float64
    _assert_total_A_matches_elementwise(qcqp, lags)


def test_assembly_map_dropped_on_constraint_edits(sparse_qcqp_data):
    """Every edit to the constraint set leaves A(lags) consistent: a surviving stale
    map (it holds a copy of each A_k's data) would show up immediately."""
    qcqp = sparse_qcqp_data["qcqp"]
    qcqp.current_lags = qcqp.find_feasible_lags()
    qcqp._get_total_A(np.abs(qcqp.current_lags))
    assert qcqp._assembly_map is not None

    qcqp._invalidate_factor_cache()
    assert qcqp._assembly_map is None and not qcqp._assembly_map_built

    rng = np.random.default_rng(0)
    edits = [
        lambda: qcqp.refine_projectors(),
        lambda: qcqp.add_constraints(
            [rng.standard_normal(qcqp.Proj.Pstruct.size) + 0j]
        ),
        lambda: qcqp.merge_lead_constraints(merged_num=2),
    ]
    for edit in edits:
        qcqp._get_total_A(np.abs(qcqp.current_lags))  # build a map, then invalidate it
        edit()
        _assert_total_A_matches_elementwise(qcqp, np.abs(qcqp.current_lags))


def test_assembly_map_falls_back_when_too_large(sparse_qcqp_data):
    """Above the memory budget the map is declined and the elementwise sum is used."""
    qcqp = sparse_qcqp_data["qcqp"]
    lags = np.abs(qcqp.find_feasible_lags())
    qcqp._assembly_map_max_bytes = 1
    qcqp._invalidate_factor_cache()
    _assert_total_A_matches_elementwise(qcqp, lags)
    assert qcqp._assembly_map is None


def test_assembly_map_declined_for_dense(dense_qcqp_data):
    """The dense formulation has no shared pattern to exploit, so no map is built."""
    qcqp = dense_qcqp_data["qcqp"]
    lags = np.abs(qcqp.find_feasible_lags())
    got = qcqp._get_total_A(lags)
    want = _elementwise_total_A(qcqp, lags)
    assert qcqp._assembly_map is None
    assert np.allclose(got, want, atol=1e-12)


def test_invalid_ortho_metric_raises():
    """An unknown ortho_metric is rejected at construction time."""
    with pytest.raises(ValueError, match="ortho_metric"):
        GCDHyperparameters(ortho_metric="bogus")


def test_invalid_metric_raises(sparse_qcqp_data):
    """A typo'd metric in add/merge raises instead of silently running the
    euclidean path, and leaves the QCQP unmodified."""
    qcqp = sparse_qcqp_data["qcqp"]
    qcqp.current_lags = _interior_feasible_lags(qcqp)
    n_before = qcqp.n_proj_constr
    lags_before = qcqp.current_lags.copy()

    nnz = qcqp.Proj.Pstruct.size
    with pytest.raises(ValueError, match="metric"):
        qcqp.add_constraints([np.ones(nnz, dtype=complex)], metric="hs")
    with pytest.raises(ValueError, match="metric"):
        qcqp.merge_lead_constraints(merged_num=2, metric="hilbert-schmidt")

    assert qcqp.n_proj_constr == n_before
    assert np.array_equal(qcqp.current_lags, lags_before)


# --- general inequality constraints (gen_ineq) ---

# (method, newton_alternate_above): BFGS, Newton taking full steps, and Newton
# alternating with gradient steps, as it does above 100 multipliers
INEQ_SOLVERS = [("bfgs", 100), ("newton", 100), ("newton", 0)]
INEQ_SOLVER_IDS = ["bfgs", "newton", "newton-alternating"]

# A multiplier at its bound is exactly 0 where the optimizer stops, but the reported
# multipliers are those of the smallest dual evaluated, which can be the iterate just
# before it landed on 0 (seen at ~1e-14).
AT_BOUND = 1e-12


def _solve_tight(qcqp, solver, init_lags):
    """Solve the dual to tight tolerance; return (dual, lags)."""
    method, alternate_above = solver
    params = OptimizationHyperparameters(
        opttol=1e-8, break_iter_period=20, newton_alternate_above=alternate_above
    )
    dual, lags, _, _, _ = qcqp.solve_current_dual_problem(
        method, opt_params=params, init_lags=init_lags
    )
    return dual, lags


def _norm_bound_qcqp(s0, c0, A0, bounds, gen_ineq):
    """QCQP whose only constraints are |x|^2 <= b (or = b) for each b in bounds."""
    n = len(s0)
    eye = np.eye(n, dtype=complex)
    return DenseSharedProjQCQP(
        A0,
        s0,
        c0,
        0 * eye,
        np.zeros(n, dtype=complex),
        [],
        sp.eye_array(n, format="csc"),
        B_j=[eye] * len(bounds),
        s_2j=[np.zeros(n, dtype=complex)] * len(bounds),
        c_2j=bounds,
        gen_ineq=gen_ineq,
    )


@pytest.mark.parametrize("solver", INEQ_SOLVERS, ids=INEQ_SOLVER_IDS)
def test_general_inequality_sign_convention(solver):
    """Toy problems with known multipliers: an inequality's multiplier stays >= 0."""
    eye2 = np.eye(2, dtype=complex)

    # max -|x - a|^2 s.t. |x|^2 <= 1 with |a| = 0.5: the constraint is inactive. As an
    # equality (|x|^2 = 1) the bound is -0.25 at multiplier -0.5; as an inequality the
    # multiplier stays at exactly 0 and the bound is 0.
    a = np.array([0.3, 0.4], dtype=complex)
    qcqp = _norm_bound_qcqp(a, -0.25, eye2, [1.0], [False])
    dual, lags = _solve_tight(qcqp, solver, [1.0])
    assert np.isclose(dual, -0.25, atol=1e-7) and np.isclose(lags[0], -0.5, atol=1e-5)
    qcqp = _norm_bound_qcqp(a, -0.25, eye2, [1.0], [True])
    dual, lags = _solve_tight(qcqp, solver, [1.0])
    assert np.isclose(dual, 0.0, atol=1e-10) and lags[0] == 0.0

    # max Re(x_1 + x_2) s.t. |x|^2 <= 1 and |x|^2 <= 4: only the tighter one is active
    s0 = 0.5 * np.ones(2, dtype=complex)
    qcqp = _norm_bound_qcqp(s0, 0.0, 0 * eye2, [1.0, 4.0], [True, True])
    dual, lags = _solve_tight(qcqp, solver, [1.0, 1.0])
    assert np.isclose(dual, np.sqrt(2), rtol=1e-7)
    assert np.isclose(lags[0], 1 / np.sqrt(2), rtol=1e-6) and lags[1] == 0.0


def _wall_qcqp(gen_ineq=None, keep=None, n=40, k=8, m=4, seed=3, a0_shift=0.05):
    """Dense QCQP with k diagonal projectors, m general constraints and near-singular A0.

    A0 is scaled so that the PSD wall binds at the dual optimum, unless a0_shift is
    made large (30 moves the optimum well inside). The general constraints have
    c_2j = 0, so x = 0 satisfies every constraint and each variant
    (equality, inequality, dropped) has a finite dual. keep lists the general
    constraints to include (default all); gen_ineq is a mask over all m of them.
    """
    rng = np.random.default_rng(seed)

    def rc(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    M = rc(n, n)
    A0 = M.conj().T @ M / n + a0_shift * np.eye(n)
    A1, s0, s1 = rc(n, n), rc(n), rc(n)
    Pdiags = rc(n, k)
    Pdiags[rng.random((n, k)) < 0.3] = 0.0
    Projlist = [sp.diags_array(Pdiags[:, j], format="csc") for j in range(k)]
    B_j = [(H + H.conj().T) / (2 * np.sqrt(n)) for H in (rc(n, n) for _ in range(m))]
    s_2j = [rc(n) for _ in range(m)]
    keep = list(range(m)) if keep is None else list(keep)
    return DenseSharedProjQCQP(
        A0,
        s0,
        0.0,
        A1,
        s1,
        Projlist,
        sp.eye_array(n, format="csc"),
        B_j=[B_j[j] for j in keep],
        s_2j=[s_2j[j] for j in keep],
        c_2j=np.zeros(len(keep)),
        gen_ineq=None if gen_ineq is None else [gen_ineq[j] for j in keep],
    )


@pytest.mark.parametrize("solver", INEQ_SOLVERS, ids=INEQ_SOLVER_IDS)
def test_general_inequality_against_equality_and_dropped(solver):
    """With the PSD wall binding, constraint j as an inequality gives the equality dual
    if its multiplier there is >= 0, and otherwise the dual with j dropped, with its
    multiplier at 0: the dual is convex, so when the unbounded optimum has
    mu_j < 0 the optimum over mu_j >= 0 lies on mu_j = 0."""
    k, m = 8, 4
    x0 = np.concatenate([np.full(k, 1e-6), np.zeros(m)])
    qcqp = _wall_qcqp()
    dual_eq, lags_eq = _solve_tight(qcqp, solver, x0)
    mu_eq = lags_eq[k:]
    assert np.any(mu_eq < 0) and np.any(mu_eq > 0), "fixture should cover both cases"
    eigs = np.linalg.eigvalsh(qcqp._get_total_A(lags_eq))
    assert eigs[0] < 1e-6 * eigs[-1], "fixture should have the PSD wall binding"

    for j in range(m):
        dual, lags = _solve_tight(_wall_qcqp(gen_ineq=np.arange(m) == j), solver, x0)
        assert lags[k + j] >= 0.0
        if mu_eq[j] >= 0:
            assert np.isclose(dual, dual_eq, rtol=1e-7)
        else:
            others = [i for i in range(m) if i != j]
            dropped = _wall_qcqp(keep=others)
            dual_drop, _ = _solve_tight(dropped, solver, np.delete(x0, k + j))
            assert np.isclose(dual, dual_drop, rtol=1e-7)
            assert lags[k + j] <= AT_BOUND

    # all inequalities at once: a relaxation of the equalities, a restriction of
    # dropping them all, and some bounds active
    dual, lags = _solve_tight(_wall_qcqp(gen_ineq=[True] * m), solver, x0)
    dual_none, _ = _solve_tight(_wall_qcqp(keep=[]), solver, x0[:k])
    assert dual_eq * (1 - 1e-7) <= dual <= dual_none * (1 + 1e-7)
    assert np.all(lags[k:] >= 0.0) and np.any(lags[k:] <= AT_BOUND)


@pytest.mark.parametrize("solver", INEQ_SOLVERS, ids=INEQ_SOLVER_IDS)
def test_general_inequality_complementary_slackness_off_the_wall(solver):
    """Off the PSD wall the dual's gradient is the constraint values f_j(x*), so at the
    optimum f_j(x*) = 0 for the equalities and for inequalities with mu_j > 0, and
    f_j(x*) >= 0 where mu_j = 0: x* is primal feasible and attains the bound. On the
    wall the gradient also carries the wall's normal, and none of this need hold."""
    k, m = 8, 4
    qcqp = _wall_qcqp(gen_ineq=[True] * m, a0_shift=30.0)
    x0 = np.concatenate([np.full(k, 1e-6), np.zeros(m)])
    dual, lags = _solve_tight(qcqp, solver, x0)
    eigs = np.linalg.eigvalsh(qcqp._get_total_A(lags))
    assert eigs[0] > 0.1 * eigs[-1], "fixture should have the optimum off the wall"

    mu, grad = lags[k:], qcqp.current_grad
    at_bound = mu <= AT_BOUND
    assert np.all(mu >= 0.0) and np.any(at_bound) and not np.all(at_bound)
    tol = 1e-6 * max(1.0, abs(dual))
    free = np.concatenate([np.ones(k, dtype=bool), ~at_bound])
    assert np.all(np.abs(grad[free]) < tol)  # equalities and active inequalities
    assert np.all(grad[k:][at_bound] > 0.0)  # inactive inequalities hold strictly
    assert np.all(np.abs(mu * grad[k:]) < tol)

    x = qcqp.current_xstar
    primal = np.real(-x.conj() @ qcqp.A0 @ x + 2 * x.conj() @ qcqp.s0) + qcqp.c0
    assert np.isclose(primal, dual, rtol=1e-6)


def test_run_gcd_keeps_general_inequality(sparse_qcqp_data):
    """GCD only changes projector constraints, so a power bound |A2 x|^2 <= P marked
    as an inequality keeps its place and its sign through GCD: its multiplier ends
    > 0 when the bound cuts off the unconstrained optimum, and 0 when not."""
    d = sparse_qcqp_data
    Pdiags = d["Pdiags"]
    Projlist = [
        sp.diags_array(Pdiags[:, j], format="csc") for j in range(Pdiags.shape[1])
    ]
    params = GCDHyperparameters(
        max_gcd_iter_num=6, max_proj_cstrt_num=6, gcd_iter_period=5, gcd_tol=1e-3
    )

    def ldos_qcqp(**general):
        return SparseSharedProjQCQP(
            d["A0"], d["s0"], d["c"], d["A1"], d["A2"], d["s1"], Projlist,
            Projlist[0], verbose=0, **general,
        )  # fmt: skip

    free = ldos_qcqp()
    free.run_gcd(params)
    power = np.linalg.norm(d["A2"] @ free.current_xstar) ** 2
    m = d["A2"].shape[0]

    for factor, active in [(0.5, True), (2.0, False)]:
        qcqp = ldos_qcqp(
            B_j=[sp.eye_array(m, format="csc", dtype=complex)],
            s_2j=[np.zeros(m, dtype=complex)],
            c_2j=[factor * power],
            gen_ineq=[True],
        )
        qcqp.run_gcd(params)
        assert qcqp.n_gen_constr == 1 and list(qcqp.gen_ineq) == [True]
        assert len(qcqp.current_lags) == qcqp.n_proj_constr + 1
        mu = qcqp.current_lags[-1]
        assert mu > AT_BOUND if active else mu <= AT_BOUND


def test_gen_ineq_validation_and_mask():
    """gen_ineq must match the general constraints; the bounded mask is in lags order."""
    eye2 = np.eye(2, dtype=complex)
    s0 = np.ones(2, dtype=complex)
    with pytest.raises(ValueError, match="gen_ineq"):
        _norm_bound_qcqp(s0, 0.0, eye2, [1.0], [True, False])

    # without inequalities the optimizer runs unbounded, exactly as before
    assert _norm_bound_qcqp(s0, 0.0, eye2, [1.0], None)._nonneg_mask() is None
    assert _norm_bound_qcqp(s0, 0.0, eye2, [1.0], [False])._nonneg_mask() is None

    # projector multipliers first (never bounded), then the general ones
    mask = _wall_qcqp(gen_ineq=[False, True, False, True])._nonneg_mask()
    assert np.array_equal(mask, [False] * 8 + [False, True, False, True])

    # an inequality's multiplier must start >= 0
    qcqp = _norm_bound_qcqp(s0, 0.0, eye2, [1.0], [True])
    with pytest.raises(ValueError, match="negative"):
        qcqp.solve_current_dual_problem("newton", init_lags=[-0.5])

    # reported multipliers: roundoff below 0 on an inequality is clipped, so warm
    # starts work; anything more negative is an error, not roundoff
    qcqp = _wall_qcqp(gen_ineq=[False, True, False, True])
    lags = np.r_[np.full(8, -0.3), -0.2, -1e-17, 0.5, 0.0]
    clipped = qcqp._clip_to_bounds(lags)
    assert clipped[9] == 0.0
    assert np.array_equal(np.delete(clipped, 9), np.delete(lags, 9))
    with pytest.raises(RuntimeError, match="below 0"):
        qcqp._clip_to_bounds(np.r_[np.zeros(8), 0.0, -1e-3, 0.5, 0.0])


def test_gen_ineq_keeps_positional_order():
    """gen_ineq comes after verbose, so positional calls written before it still work."""
    n = 3
    eye, zeros = np.eye(n, dtype=complex), np.zeros(n, dtype=complex)
    P = sp.diags_array(np.ones(n), format="csc")
    eye_sp = sp.csc_array(eye)
    dense = DenseSharedProjQCQP(
        eye, zeros, 0.0, eye, zeros, [P], P, None, [eye], [zeros], [1.0], 0
    )
    sparse = SparseSharedProjQCQP(
        eye_sp, zeros, 0.0, eye_sp, eye_sp, zeros, [P], P, [eye_sp], [zeros], [1.0], 0
    )
    for qcqp in (dense, sparse):
        assert qcqp.verbose == 0 and not np.any(qcqp.gen_ineq)
