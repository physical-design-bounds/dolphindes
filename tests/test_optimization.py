"""Tests for optimization code."""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest
import scipy.optimize

from dolphindes.cvxopt import BFGS, Alt_Newton_GD, OptimizationHyperparameters
from dolphindes.types import ComplexArray, FloatNDArray

np.random.seed(0)


def _generate_psd_matrix(n: int, cond_number: int = 1000) -> FloatNDArray:
    """Generate a random positive semi-definite matrix."""
    Q, _ = np.linalg.qr(np.random.randn(n, n))  # Random orthogonal matrix
    eigenvalues = np.logspace(0, np.log10(cond_number), n)  # Spread eigenvalues
    A = Q @ np.diag(eigenvalues) @ Q.T  # Construct A with desired conditioning
    return np.asarray(A, dtype=np.float64)


@pytest.fixture
def _optimization_setup() -> Dict[str, Any]:
    """Set up the optimization problem."""
    num_dof = 20
    A = _generate_psd_matrix(num_dof)
    b = np.random.rand(num_dof)
    opttol = 1e-4

    def optfunc(
        x: Union[ComplexArray, FloatNDArray],
        get_grad: bool = False,
        get_hess: bool = False,
        penalty_vectors: List[FloatNDArray] = [],
    ) -> Tuple[
        float,
        Union[ComplexArray, FloatNDArray, List[Any]],
        Union[ComplexArray, FloatNDArray, List[Any]],
        int,
    ]:
        objval = float((x.conj() @ A @ x - b.conj() @ x).real)
        grad: Union[ComplexArray, FloatNDArray, List[Any]] = (
            2 * A @ x - b if get_grad else []
        )
        hess: Union[ComplexArray, FloatNDArray, List[Any]] = 2 * A if get_hess else []
        return objval, grad, hess, 0

    def feasible_func(x: Union[ComplexArray, FloatNDArray, List[Any]]) -> bool:
        return True

    def penalty_func(x: FloatNDArray) -> Tuple[FloatNDArray, Any]:
        # This test never hits the feasibility wall, but the optimizer expects
        # (penalty_vector, aux) if it ever needs it.
        return np.zeros_like(x, dtype=np.float64), None

    opt_config = OptimizationHyperparameters(
        opttol=opttol,
        verbose=4,
        break_iter_period=10,
        gradConverge=True,
    )

    Ainv = np.linalg.inv(A)
    analytical_solution = 1 / 2 * Ainv @ b

    return {
        "num_dof": num_dof,
        "optfunc": optfunc,
        "feasible_func": feasible_func,
        "penalty_func": penalty_func,
        "opt_config": opt_config,
        "opttol": opttol,
        "analytical_solution": analytical_solution,
    }


def test_bfgs_optimum(_optimization_setup: Dict[str, Any]) -> None:
    """Test BFGS routine."""
    setup = _optimization_setup
    opt = BFGS(
        setup["optfunc"],
        setup["feasible_func"],
        setup["penalty_func"],
        True,
        setup["opt_config"],
    )
    opt.run(10 * np.array(np.random.random(setup["num_dof"])))
    x, fx = opt.get_last_opt()

    assert x is not None
    assert fx is not None
    assert np.allclose(x, setup["analytical_solution"], atol=setup["opttol"])
    assert np.allclose(
        fx, setup["optfunc"](setup["analytical_solution"])[0], atol=setup["opttol"]
    )


def test_newton_optimum(_optimization_setup: Dict[str, Any]) -> None:
    """Test Newton routine."""
    setup = _optimization_setup
    opt = Alt_Newton_GD(
        setup["optfunc"],
        setup["feasible_func"],
        setup["penalty_func"],
        True,
        setup["opt_config"],
    )
    opt.run(10 * np.array(np.random.random(setup["num_dof"])))
    x, fx = opt.get_last_opt()

    assert x is not None
    assert fx is not None
    assert np.allclose(x, setup["analytical_solution"], atol=setup["opttol"])
    assert np.allclose(
        fx, setup["optfunc"](setup["analytical_solution"])[0], atol=setup["opttol"]
    )
    assert np.allclose(
        fx, setup["optfunc"](setup["analytical_solution"])[0], atol=setup["opttol"]
    )


def _bfgs_for_update() -> BFGS:
    """A BFGS instance for calling _update_Hinv directly (its callables are unused)."""
    return BFGS(
        lambda x, **kwargs: (0.0, np.zeros_like(x), None, None),
        lambda x: True,
        lambda x: (np.zeros_like(x), None),
        True,
        OptimizationHyperparameters(),
    )


def test_bfgs_update_keeps_small_scale_curvature() -> None:
    """A genuine curvature pair at a small scale still updates Hinv.

    With a small objective and short steps, gamma @ delta is far below any absolute
    threshold (here ~1e-18) while the curvature it measures is real: the update must
    still satisfy the equation Hinv_new @ gamma = delta.
    """
    rng = np.random.default_rng(1)
    n = 6
    H = 1e-8 * _generate_psd_matrix(n, cond_number=10)
    delta = 1e-5 * rng.standard_normal(n)
    gamma = H @ delta
    assert 0 < gamma @ delta < 1e-15

    Hinv = np.eye(n)
    new_Hinv = _bfgs_for_update()._update_Hinv(Hinv, gamma, np.zeros(n), delta)

    assert not np.allclose(new_Hinv, Hinv)
    assert np.allclose(new_Hinv @ gamma, delta, rtol=1e-8, atol=0)


@pytest.mark.parametrize(
    "gamma, delta",
    [
        (np.array([1.0, -2.0, 0.5]), np.zeros(3)),  # the step did not move x
        (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),  # negative curvature
        (np.array([np.nan, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),  # NaN gradient
    ],
)
def test_bfgs_update_skips_degenerate_pairs(gamma, delta) -> None:
    """Pairs without positive curvature leave Hinv as it was, never NaN."""
    Hinv = np.diag([1.0, 2.0, 3.0])
    new_Hinv = _bfgs_for_update()._update_Hinv(Hinv, gamma, np.zeros(3), delta)
    assert np.array_equal(new_Hinv, Hinv)


def test_newton_reaches_far_minimizer_of_ill_conditioned_quadratic() -> None:
    """Newton steps keep their length: a far, badly conditioned minimizer is cheap.

    The minimizer of this quadratic lies 1e6 away along its flattest direction (the
    Hessian's condition number is 1e8). The full Newton step reaches it at once
    (~30 objective evaluations in all). A normalized direction whose step size is
    grown by doubling would take a lot more.
    """
    n = 6
    A = _generate_psd_matrix(n, cond_number=10**8)
    flattest = np.linalg.eigh(A)[1][:, 0]
    x_star = 1e6 * flattest
    b = 2 * A @ x_star
    calls = {"n": 0}

    def optfunc(x, get_grad=False, get_hess=False, penalty_vectors=[]):
        calls["n"] += 1
        return float(x @ A @ x - b @ x), 2 * A @ x - b, 2 * A, 0

    opt = Alt_Newton_GD(
        optfunc,
        lambda x: True,
        lambda x: (np.zeros_like(x), None),
        True,
        OptimizationHyperparameters(opttol=1e-6, max_restart=1),
    )
    x, _, _, _ = opt.run(np.zeros(n))

    assert np.allclose(x, x_star, rtol=1e-6, atol=1e-6 * np.linalg.norm(x_star))
    assert calls["n"] <= 100


@pytest.mark.parametrize("wall, expected_checks", [(1.0, [100.0, 0.9]), (None, None)])
def test_line_search_jumps_to_the_wall(wall, expected_checks) -> None:
    """After a first infeasible trial step, the line search tries 0.9x the wall
    distance wall_step_func reports, instead of shrinking the step 0.7x at a time;
    without an estimate (None) it shrinks as before."""
    checks = []

    def feasible(x):
        checks.append(float(x[0]))
        return bool(x[0] < 1.0)  # the wall is 1 away along +x

    opt = Alt_Newton_GD(
        lambda x, **kwargs: (float(-x[0]), None, None, None),
        feasible,
        lambda x: (np.zeros_like(x), None),
        True,
        OptimizationHyperparameters(),
    )
    opt.wall_step_func = lambda x, d: wall
    x0, d = np.zeros(1), np.ones(1)
    alpha_opt, alpha_feas = opt._line_search(d, x0, 0.0, -d, 100.0)

    if expected_checks is not None:
        assert checks == expected_checks
        assert alpha_feas == alpha_opt == 0.9
    else:
        assert len(checks) > 10
        assert np.isclose(alpha_feas, 100.0 * 0.7 ** (len(checks) - 1))


@pytest.mark.parametrize("alternate_above", [np.inf, 3])
def test_newton_alternates_with_gradient_steps_above_threshold(alternate_above) -> None:
    """With more multipliers than newton_alternate_above, Newton asks for a Hessian
    only every other step, alternating with gradient steps; with fewer, every step is
    a Newton step. Both reach the minimizer."""
    # a local generator: other test modules depend on the global random state
    rng = np.random.default_rng(0)
    n = 6
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(np.logspace(0, 3, n)) @ Q.T
    b = rng.random(n)
    hess_requested = []

    def optfunc(x, get_grad=False, get_hess=False, penalty_vectors=[]):
        if get_grad:  # one call per step; the line search asks for values only
            hess_requested.append(get_hess)
        hess = 2 * A if get_hess else None
        return float(x @ A @ x - b @ x), 2 * A @ x - b, hess, 0

    opt = Alt_Newton_GD(
        optfunc,
        lambda x: True,
        lambda x: (np.zeros_like(x), None),
        True,
        OptimizationHyperparameters(
            opttol=1e-6, max_restart=0, newton_alternate_above=alternate_above
        ),
    )
    x, _, _, _ = opt.run(np.ones(n))

    assert np.allclose(x, 0.5 * np.linalg.solve(A, b), atol=1e-4)
    if alternate_above < n:
        assert hess_requested == [i % 2 == 0 for i in range(len(hess_requested))]
    else:
        assert all(hess_requested)


# (optimizer, newton_alternate_above): GCD runs Newton alternating with gradient steps
SOLVERS = [(BFGS, 100), (Alt_Newton_GD, 100), (Alt_Newton_GD, 0)]
SOLVER_IDS = ["bfgs", "newton", "newton-alternating"]


def _bounded_quadratic(n=20, seed=0):
    """Ill-conditioned x^T A x - b^T x with every other x_j >= 0; returns A, b, mask, x0."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(np.logspace(0, 3, n)) @ Q.T
    b = rng.standard_normal(n)  # mixed signs, so several bounds end up active
    nonneg = np.arange(n) % 2 == 0
    return A, b, nonneg, 10 * rng.random(n)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("start", ["positive", "zeros"])
def test_bounded_quadratic_matches_lbfgsb(solver, start):
    """Minimize x^T A x - b^T x with every other x_j >= 0, compared with L-BFGS-B.

    The "zeros" start puts every bounded variable at 0, including ones whose optimum
    is positive, so those must be released from the fixed set.
    """
    A, b, nonneg, x0 = _bounded_quadratic()
    n = b.size
    x_ref = scipy.optimize.minimize(
        lambda x: x @ A @ x - b @ x,
        np.zeros(n),
        jac=lambda x: 2 * A @ x - b,
        method="L-BFGS-B",
        bounds=[(0, None) if m else (None, None) for m in nonneg],
        options=dict(ftol=1e-15, gtol=1e-12, maxiter=10000),
    ).x
    active = nonneg & (x_ref < 1e-10)
    assert active.sum() >= 3 and np.any(nonneg & (x_ref > 1e-6))

    def optfunc(x, get_grad=False, get_hess=False, penalty_vectors=[]):
        return float(x @ A @ x - b @ x), 2 * A @ x - b, 2 * A, 0

    if start == "zeros":
        x0[nonneg] = 0.0
    cls, alternate_above = solver
    params = OptimizationHyperparameters(
        opttol=1e-8, break_iter_period=20, newton_alternate_above=alternate_above
    )
    no_penalty = lambda x: (np.zeros_like(x), None)  # noqa: E731
    opt = cls(optfunc, lambda x: True, no_penalty, True, params, nonneg)
    x = opt.run(x0)[0]

    assert np.allclose(x, x_ref, atol=1e-6)
    assert np.all(x[nonneg] >= 0.0) and np.all(x[active] == 0.0)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_bounded_tied_variables_land_on_zero(solver):
    """Variables that reach 0 at the same step all land on exactly 0 in that step.

    Minimizes sum (x_j + 1)^2 with every x_j >= 0 from x = 1, where every step
    direction is uniform, so all 30 bounds are hit at once.
    """
    n_grad = 0

    def optfunc(x, get_grad=False, get_hess=False, penalty_vectors=[]):
        nonlocal n_grad
        n_grad += get_grad
        return float(np.sum((x + 1) ** 2)), 2 * (x + 1), 2 * np.eye(x.size), 0

    cls, alternate_above = solver
    params = OptimizationHyperparameters(
        opttol=1e-8, break_iter_period=20, newton_alternate_above=alternate_above
    )
    no_penalty = lambda x: (np.zeros_like(x), None)  # noqa: E731
    opt = cls(
        optfunc, lambda x: True, no_penalty, True, params, nonneg=np.ones(30, bool)
    )
    x = opt.run(np.ones(30))[0]

    assert np.all(x == 0.0)
    assert n_grad < 10


def test_nonneg_mask_validation():
    """The mask must be boolean and match x0; x0 must be >= 0 where it is set."""
    args = (
        lambda x, **kwargs: (0.0, np.zeros_like(x), np.eye(x.size), None),
        lambda x: True,
        lambda x: (np.zeros_like(x), None),
        True,
        OptimizationHyperparameters(),
    )
    with pytest.raises(TypeError, match="boolean"):
        BFGS(*args, nonneg=[0])  # an index list, not a mask
    opt = Alt_Newton_GD(*args, nonneg=np.array([True, False]))
    with pytest.raises(ValueError, match="shape"):
        opt.run(np.ones(3))
    with pytest.raises(ValueError, match="negative"):
        opt.run(np.array([-1.0, 0.0]))


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_bounded_stopping_sees_variable_to_release(solver):
    """A variable at 0 whose gradient says it should grow keeps the run going.

    Minimizes |x - 1|^2 with x_2 >= 0 from x = [1, 0]. The stationarity test weighs
    each gradient entry by |x_j|, so on its own it would stop at once, at [1, 0].
    """

    def optfunc(x, get_grad=False, get_hess=False, penalty_vectors=[]):
        return float(np.sum((x - 1) ** 2)), 2 * (x - 1), 2 * np.eye(2), 0

    cls, alternate_above = solver
    params = OptimizationHyperparameters(
        opttol=1e-8, min_inner_iter=0, newton_alternate_above=alternate_above
    )
    no_penalty = lambda x: (np.zeros_like(x), None)  # noqa: E731
    nonneg = np.array([False, True])
    opt = cls(optfunc, lambda x: True, no_penalty, True, params, nonneg=nonneg)
    x = opt.run(np.array([1.0, 0.0]))[0]

    assert np.allclose(x, [1.0, 1.0])


def test_bounded_grad_converge_ignores_gradient_held_at_bound():
    """With gradConverge, the gradient-norm test leaves out entries held at 0.

    Those entries stay > 0 at the solution, so if they counted the test could never
    pass, and Newton would run on until the periodic objective test.
    """
    A, b, nonneg, x0 = _bounded_quadratic()
    results = {}
    for grad_converge in (False, True):
        count = 0

        def optfunc(x, get_grad=False, get_hess=False, penalty_vectors=[]):
            nonlocal count
            count += 1
            return float(x @ A @ x - b @ x), 2 * A @ x - b, 2 * A, 0

        params = OptimizationHyperparameters(
            opttol=1e-8, break_iter_period=20, gradConverge=grad_converge
        )
        no_penalty = lambda x: (np.zeros_like(x), None)  # noqa: E731
        opt = Alt_Newton_GD(optfunc, lambda x: True, no_penalty, True, params, nonneg)
        results[grad_converge] = opt.run(x0.copy())[0], count

    (x_plain, n_plain), (x_grad, n_grad) = results[False], results[True]
    assert np.allclose(x_grad, x_plain, atol=1e-8)
    assert n_grad <= n_plain + 2
