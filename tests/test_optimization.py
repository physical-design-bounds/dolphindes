"""Tests for optimization code."""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest

from dolphindes.cvxopt import BFGS, Alt_Newton_GD
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

    def penalty_func(x: Any) -> None:
        pass

    opt_config = {
        "opttol": opttol,
        "verbose": 4,
        "break_iter_period": 10,
        "gradConverge": True,
    }

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
