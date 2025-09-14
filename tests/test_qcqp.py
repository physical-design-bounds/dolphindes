"""Tests for sparse and dense QCQP formulations and general constraints.

The tests validate consistency between sparse and dense implementations, check
dual optimization (BFGS and Newton), verify gradients/Hessians against stored
reference arrays, and exercise added general constraints.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from dolphindes.cvxopt import DenseSharedProjQCQP, SparseSharedProjQCQP


@pytest.fixture
def data_dir():
    """Return the path to the reference data directory."""
    return Path(os.path.dirname(__file__)) / "reference_arrays" / "qcqp_example" / "sparse"  # noqa: E501


@pytest.fixture(params=['global', 'local'])
def sparse_qcqp_data(data_dir, request):
    """Fixture to load data and initialize SparseSharedProjQCQP."""
    added_str = request.param
    print(f"\nLoading data for: {added_str} constraints")
    data_path = data_dir / added_str
    lags = np.load(data_path / 'ldos_sparse_lags.npy', allow_pickle=True)
    lags_optimal = np.load(
        data_path / 'ldos_sparse_lags_optimal.npy', allow_pickle=True
    )
    A0 = sp.csc_array(sp.load_npz(data_path / 'ldos_sparse_A0.npz'))
    A1 = sp.csc_array(sp.load_npz(data_path / 'ldos_sparse_A1.npz'))
    A2 = sp.csc_array(sp.load_npz(data_path / 'ldos_sparse_A2.npz'))
    s0 = np.load(data_path / 'ldos_sparse_s0.npy', allow_pickle=True)
    s1 = np.load(data_path / 'ldos_sparse_s1.npy', allow_pickle=True)
    projections_diags = np.load(
        data_path / 'ldos_some_projections.npy', 
        allow_pickle=True
    )
    # Interleave projections_diags and projections_diags * -1j
    projections_diags = np.asarray(projections_diags)
    interleaved = np.empty(
        (2 * projections_diags.shape[0], projections_diags.shape[1]),
        dtype=complex
    )
    interleaved[0::2] = projections_diags
    interleaved[1::2] = projections_diags * -1j
    projections_diags = interleaved
    Pdiags = projections_diags.T

    c = np.load(data_path / 'ldos_dualconst.npy', allow_pickle=True)
    
    sparse_ldos_qcqp_instance = SparseSharedProjQCQP(
        A0, s0, c, A1, A2, s1, Pdiags, verbose=0
    )
    
    return {
        "qcqp": sparse_ldos_qcqp_instance,
        "lags": lags,
        "lags_optimal": lags_optimal,
        "A0": A0, "A1": A1, "A2": A2,
        "s0": s0, "s1": s1,
        "Pdiags": Pdiags,
        "c": c,
        "data_path": data_path,
        "added_str": added_str
    }


@pytest.fixture
def data_dir_dense():
    """Return the path to the reference data directory."""
    return (
        Path(os.path.dirname(__file__))
        / "reference_arrays"
        / "qcqp_example"
        / "dense"
    )


@pytest.fixture(params=['global', 'local'])
def dense_qcqp_data(data_dir_dense, request):
    """Fixture to load data and initialize DenseSharedProjQCQP."""
    added_str = request.param
    print(f"\nLoading data for: {added_str} constraints")
    data_path = data_dir_dense / added_str
    lags = np.load(data_path / 'ldos_dense_lags.npy', allow_pickle=True)
    lags_optimal = np.load(data_path / 'ldos_dense_lags_optimal.npy', allow_pickle=True)
    A0 = np.load(data_path / 'ldos_dense_A0.npy')
    A1 = np.load(data_path / 'ldos_dense_A1.npy')
    s0 = np.load(data_path / 'ldos_dense_s0.npy', allow_pickle=True)
    s1 = np.load(data_path / 'ldos_dense_s1.npy', allow_pickle=True)
    projections_diags = np.load(
        data_path / 'ldos_some_projections.npy', allow_pickle=True
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

    c = np.load(data_path / 'ldos_dualconst.npy', allow_pickle=True)
    sparse_ldos_qcqp_instance = DenseSharedProjQCQP(
        A0, s0, c, A1, s1, Pdiags, None, verbose=0
    )
    
    return {
        "qcqp": sparse_ldos_qcqp_instance,
        "lags": lags,
        "lags_optimal": lags_optimal,
        "A0": A0, "A1": A1, "A2": None,
        "s0": s0, "s1": s1,
        "Pdiags": Pdiags,
        "c": c,
        "data_path": data_path,
        "added_str": added_str
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

        combined_projector = sparse_ldos_qcqp._add_projectors(lags)

        print("Testing totalA = known total A")
        ref_totalA = sp.load_npz(data / 'ldos_sparse_totalA.npz')
        calc_totalA = sparse_ldos_qcqp._get_total_A(lags)
        assert sp.issparse(calc_totalA), "Total A should be a sparse matrix."
        assert sp.csr_array(calc_totalA).shape == ref_totalA.shape, (
            "Shape of calculated total A does not match reference."
        )
        assert np.allclose(calc_totalA.toarray(), ref_totalA.toarray()), (
            "Calculated total A does not match reference total A."
        )

        print("Testing totalS = known total S")
        ref_totals = np.load(data / 'ldos_sparse_total_s.npy', allow_pickle=True)
        calc_totals = sparse_ldos_qcqp._get_total_S(combined_projector, np.array([]))
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
        print(f'some feasible dual : {dual}')
        feasible = sparse_ldos_qcqp.is_dual_feasible(init_lags)
        assert feasible, "Dual is not feasible."

        dual_opt, grad_opt, hess_opt, aux_opt = sparse_ldos_qcqp.get_dual(
            lags_optimal, get_grad=True, get_hess=False
        )
        assert dual is not None
        print(f'optimal dual : {dual_opt}')
        print(np.load(data / 'ldos_dualval.npy'))
        assert np.allclose(
            dual_opt, np.load(data / 'ldos_dualval.npy'), atol=1e-6
        ), "Dual values do not match."\
        
        assert np.allclose(
            grad_opt, np.load(data / 'ldos_optgrad.npy'), atol=1e-6
        ), "Gradients do not match."

        print("Testing dualvalue of penalty vector")
        penalty, value = sparse_ldos_qcqp._get_PSD_penalty(lags_optimal)
        dual_opt_penalty, grad_opt_penalty, hess_opt_penalty, aux_opt_penalty = (
            sparse_ldos_qcqp.get_dual(
                lags_optimal, get_grad=True, penalty_vectors=[penalty]
            )
        )
        assert np.allclose(
            aux_opt_penalty.dualval_penalty,
            np.load(data / 'ldos_dualval_penalty.npy'),
            atol=1e-6,
        ), "Penalty dual values do not match."
        assert np.allclose(
            aux_opt_penalty.grad_penalty,
            np.load(data / 'ldos_grad_penalty.npy'),
            atol=1e-6,
        ), "Penalty gradients do not match."

        print("Testing solving the dual problem with BFGS and Newton")
        current_dual, dual_lambda, current_grad, current_hess, xstar = (
            sparse_ldos_qcqp.solve_current_dual_problem('bfgs', init_lags=init_lags)
        )

        print(f'dual lambda: {dual_lambda}')
        print(f'bound BFGS: {current_dual}')
        
        if added_str == 'global':
            dual_results['sparse_bfgs_global'] = current_dual 
        else:
            dual_results['sparse_bfgs_local'] = current_dual

        assert np.allclose(dual_lambda, lags_optimal, atol=1e-4), (
            "Dual lambda does not match optimal lags."
        )
        assert np.allclose(current_dual, dual_opt, atol=1e-2), (
            "Dual values does not match optimal value."
        )

        current_dual, dual_lambda, current_grad, current_hess, xstar = (
            sparse_ldos_qcqp.solve_current_dual_problem('newton', init_lags=init_lags)
        )
        print(f'dual lambda newton: {dual_lambda}')
        print(f'bound Newton: {current_dual}')
        
        if added_str == 'global':
            dual_results['sparse_newton_global'] = current_dual
        else:
            dual_results['sparse_newton_local'] = current_dual

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
        assert np.allclose(
            hess_opt, np.load(data / 'ldos_opthess.npy'), rtol=1e-1
        ), "Hessian does not match optimal Hessian."
        
        print("Testing iterative splitting step")
        results = [] 
        result_counter = 0
        for result in sparse_ldos_qcqp.iterative_splitting_step():
            results.append((sparse_ldos_qcqp.Pdiags.shape[1], result[0]))
            result_counter += 1
            if result_counter > 0:
                assert results[result_counter - 1][1] >= result[0], (
                    "Iterative splitting step must decrease dualval."
                )
            # Limit to avoid excessive iterations (run full test manually for rigor).
            if sparse_ldos_qcqp.Pdiags.shape[1] > 4:
                break
        print(results)

        # print("Testing the merging of constraints")
        # from dolphindes.cvxopt import gcd
        # sparse_ldos_qcqp.compute_precomputed_values()
        # dual_opt, dual_grad, dual_hess, _ = sparse_ldos_qcqp.get_dual(
        #     sparse_ldos_qcqp.current_lags, get_grad=True, get_hess=True
        # )
            
            # gcd.merge_lead_constraints(sparse_ldos_qcqp, 5)
        # merged_dual, merged_grad, merged_hess, _ = sparse_ldos_qcqp.get_dual(
        #     sparse_ldos_qcqp.current_lags, get_grad=True, get_hess=True
        # )
        # assert np.allclose(
        #     dual_opt, merged_dual, atol=1e-2
        # ), "dual value changed after constraint merge."
        # assert np.allclose(
        #     dual_grad[-5:], merged_grad[-5:], atol=1e-2
        # ), "dual grad changed after constraint merge."
        # assert np.allclose(
        #     dual_hess[-5:, -5:], merged_hess[-5:, -5:], rtol=1e-2
        # ), "dual Hess changed after constraint merge."
    
    
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
        combined_projector = dense_ldos_qcqp._add_projectors(lags)

        print("Testing totalA = known total A")
        ref_totalA = np.load(data / 'ldos_dense_totalA.npy')
        calc_totalA = dense_ldos_qcqp._get_total_A(lags)
        assert calc_totalA.shape == ref_totalA.shape, (
            "Shape of calculated total A does not match reference."
        )
        assert np.allclose(
            calc_totalA, ref_totalA, atol=1e-12, rtol=1e-12
        ), "Calculated total A does not match reference total A."

        print("Testing totalS = known total S")
        ref_totals = np.load(data / 'ldos_dense_total_s.npy', allow_pickle=True)
        calc_totals = dense_ldos_qcqp._get_total_S(combined_projector, np.array([]))
        assert calc_totals.shape == ref_totals.shape, (
            "Shape of calculated total S does not match reference."
        )
        assert np.allclose(
            calc_totals, ref_totals, atol=1e-12, rtol=1e-12
        ), "Calculated total S does not match reference total S."

        init_dual, init_grad, hess, aux = dense_ldos_qcqp.get_dual(lags, get_grad=True)
        print(init_grad, np.load(data / 'ldos_init_grad.npy'))
        assert np.allclose(
            init_grad, np.load(data / 'ldos_init_grad.npy')
        ), "Initial grad does not match reference."

        print(
            "Testing dualval = known dualval and grad = known grad, as well as "
            "penalty vector dualval and grad"
        )
        init_lags = dense_ldos_qcqp.find_feasible_lags()
        dual, grad, hess, aux = dense_ldos_qcqp.get_dual(init_lags, get_grad=True)
        assert dual is not None
        print(f'some feasible dual : {dual}')
        feasible = dense_ldos_qcqp.is_dual_feasible(init_lags)
        assert feasible, "Dual is not feasible."

        dual_opt, grad_opt, hess_opt, aux_opt = dense_ldos_qcqp.get_dual(
            lags_optimal, get_grad=True, get_hess=False
        )
        assert dual is not None
        print(f'optimal dual : {dual_opt}')
        print(np.load(data / 'ldos_dualval.npy'))
        assert np.allclose(
            dual_opt, np.load(data / 'ldos_dualval.npy'), atol=1e-6
        ), "Dual values do not match."
        # TODO(alessio): investigate why this is not matching well.
        assert np.allclose(
            grad_opt, np.load(data / 'ldos_grad.npy')
        ), "Gradients do not match."

        print("Testing solving the dual problem with BFGS and Newton")
        current_dual_bfgs, dual_lambda, current_grad, current_hess, xstar = (
            dense_ldos_qcqp.solve_current_dual_problem('bfgs', init_lags=init_lags)
        )

        print(f'dual lambda: {dual_lambda}')
        print(f'bound BFGS: {current_dual_bfgs}')

        assert np.allclose(dual_lambda, lags_optimal, atol=1e-4), (
            "Dual lambda does not match optimal lags."
        )
        assert np.allclose(current_dual_bfgs, dual_opt, atol=1e-2), (
            "Dual values does not match optimal value."
        )

        current_dual_newton, dual_lambda, current_grad, current_hess, xstar = (
            dense_ldos_qcqp.solve_current_dual_problem('newton', init_lags=init_lags)
        )
        print(f'dual lambda newton: {dual_lambda}')
        print(f'bound Newton: {current_dual_newton}')
        if dense_qcqp_data["added_str"] == 'global':
            dual_results['dense_bfgs_global'] = current_dual_bfgs
            dual_results['dense_newton_global'] = current_dual_newton
        else:
            dual_results['dense_bfgs_local'] = current_dual_bfgs
            dual_results['dense_newton_local'] = current_dual_newton

        assert np.allclose(dual_lambda, lags_optimal, atol=1e-4), (
            "Newton dual lambda does not match optimal lags."
        )
        assert np.allclose(current_dual_newton, dual_opt, atol=1e-2), (
            "Newton dual values does not match optimal value."
        )
        dual_opt, grad_opt, hess_opt, aux_opt = dense_ldos_qcqp.get_dual(
            dual_lambda, get_grad=True, get_hess=True
        )

    @pytest.mark.dependency(depends=["sparse_test", "dense_test"])
    def test_compare_dual_results(self, dual_results):
        """Compare dual results from BFGS and Newton (sparse vs. dense)."""
        print("Comparing dual results from BFGS and Newton methods:")
        for key, value in dual_results.items():
            print(f"{key}: {value}")
        
        assert np.allclose(
            dual_results['sparse_bfgs_global'],
            dual_results['dense_bfgs_global'],
            atol=1e-2,
        ), "BFGS global results for sparse and dense do not match."
        assert np.allclose(
            dual_results['sparse_newton_global'],
            dual_results['dense_newton_global'],
            atol=1e-2,
        ), "Newton global results for sparse and dense do not match."
        assert np.allclose(
            dual_results['sparse_bfgs_local'],
            dual_results['dense_bfgs_local'],
            atol=1e-2,
        ), "BFGS local results for sparse and dense do not match."
        assert np.allclose(
            dual_results['sparse_newton_local'],
            dual_results['dense_newton_local'],
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
    qcqp_gc = SparseSharedProjQCQP(
        A0, s0, c, A1, A2, s1, Pdiags,
        B_j=B_j, s_2j=s_2j, c_2j=c_2j, verbose=0
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
    qcqp_gc = DenseSharedProjQCQP(
        A0, s0, c, A1, s1, Pdiags,
        A2=None, B_j=B_j, s_2j=s_2j, c_2j=c_2j, verbose=0
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
    A1 = np.zeros((2, 2), dtype=complex)          # Unused (no projectors)
    s0 = 0.5 * np.ones(2, dtype=complex)          # Gives x1 + x2
    c0 = 0.0
    s1 = np.zeros(2, dtype=complex)               # No projector linear term
    Pdiags = np.zeros((2, 0), dtype=complex)      # No shared projector constraints

    # General constraint: ||x||^2 = 1
    B_j = [np.eye(2, dtype=complex)]
    s_2j = [np.zeros(2, dtype=complex)]
    c_2j = np.array([1.0])  # -x^T x + 1 = 0

    qcqp = DenseSharedProjQCQP(
        A0, s0, c0, A1, s1, Pdiags,
        A2=None, B_j=B_j, s_2j=s_2j, c_2j=c_2j, verbose=0
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