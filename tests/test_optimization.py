"""Tests for optimization code."""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest

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
