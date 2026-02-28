"""Test compilation and parametrizations of models."""  # noqa: INP001

import logging

import numpy as np
import pytest

from velocefw.model import (
    CompiledModel,
    FixedConstant,
    Parametrization,
    PolynomialBasis,
    compile_model,
)

logger = logging.getLogger(__name__)

rng = np.random.default_rng(seed=42)  # for reproducibility


def test_compile_model() -> None:
    """Test that the compiled model evaluates correctly."""
    # Create a simple polynomial model
    degree = 3
    model = PolynomialBasis(degree=degree)
    # Compile the model
    compiled_model = compile_model(model)
    if isinstance(compiled_model, CompiledModel):
        logger.info("Model compiled successfully.")
    else:
        msg = (
            f"Compiled model is of type {type(compiled_model)}, expected CompiledModel."
        )
        logger.warning(msg)
        raise TypeError(msg)
    # Check that the number of free parameters matches the model's n_params
    if model.n_params == compiled_model.n_params_full:
        logger.info("Compiled model n_params_full matches model n_params.")
    else:
        msg = (
            f"Compiled model n_params_full {compiled_model.n_params_full} does not "
            f"match model n_params {model.n_params}."
        )
        logger.warning(msg)
        raise ValueError(msg)


@pytest.mark.parametrize("degree", [1, 2, 10])
@pytest.mark.parametrize("x", [3.0, [0.0, 1.0, 2.0], np.array([0.0, 1.0, 2.0])])
def test_compile_model_evaluation(degree: int, x: float | list | np.ndarray) -> None:
    """Test that the compiled model evaluates correctly."""
    # Define parameters and input
    theta = rng.uniform(low=-1.0, high=1.0, size=degree)  # coefficients for x, x^2, x^3
    x = np.array([0.0, 1.0, 2.0])  # input values
    # Evaluate the compiled model
    compiled_model = compile_model(PolynomialBasis(degree=degree))
    output = compiled_model(theta_free=theta, x=x)
    expected_output = np.zeros_like(x)
    for i in range(degree):
        expected_output += theta[i] * x ** (i + 1)
    if np.allclose(output, expected_output):
        logger.info("Compiled model evaluation matches expected output.")
    else:
        msg = f"Compiled model evaluation {output} does not "
        msg += f"match expected {expected_output}."
        logger.warning(msg)
        raise RuntimeError(msg)

    # Check that it raises an error when x is not provided and the model requires it
    with pytest.raises(ValueError, match="x"):
        compiled_model(theta_free=theta)

    # Check if output has the correct shape
    if output.shape == x.shape:
        logger.info("Compiled model output has correct shape.")
    else:
        msg = f"Compiled model output shape {output.shape} does not match "
        msg += f"input shape {x.shape}."
        logger.warning(msg)
        raise ValueError(msg)


def test_compile_model_wrong_theta_length() -> None:
    """Test that the compiled model raises an error when given wrong theta length."""
    degree = 3
    model = PolynomialBasis(degree=degree)
    compiled_model = compile_model(model)
    theta_wrong_length = rng.uniform(
        low=-1.0,
        high=1.0,
        size=degree + 1,
    )  # one extra parameter
    with pytest.raises(ValueError, match="free"):
        compiled_model(theta_free=theta_wrong_length, x=np.array([1.0]))


def test_compile_model_wrong_theta_length_too_few() -> None:
    """Test that the compiled model raises an error when given too few parameters."""
    degree = 3
    model = PolynomialBasis(degree=degree)
    compiled_model = compile_model(model)
    theta_wrong_length = rng.uniform(
        low=-1.0,
        high=1.0,
        size=degree - 1,
    )  # one fewer parameter
    with pytest.raises(ValueError, match="free"):
        compiled_model(theta_free=theta_wrong_length, x=np.array([1.0]))


def test_fixed_constant_polynomial_combination() -> None:
    """Test that a fixed constant combined with a polynomial basis works correctly."""
    constant_value = 2.0
    constant_model = FixedConstant(value=constant_value)
    degree = 2
    polynomial_model = PolynomialBasis(degree=degree)
    combined_model = constant_model + polynomial_model
    theta_polynomial = np.array([1.0, 0.5])  # coefficients for x, x^2
    x = np.array([0.0, 1.0, 2.0])  # input values
    expected_polynomial = theta_polynomial[0] * x + theta_polynomial[1] * x**2
    expected_output = constant_value + expected_polynomial
    compiled_combined_model = compile_model(combined_model)
    output = compiled_combined_model(theta_free=theta_polynomial, x=x)
    if np.allclose(output, expected_output):
        logger.info("Combined model evaluation matches expected output.")
    else:
        msg = f"Combined compiled model evaluation {output} does not "
        msg += f"match expected {expected_output}."
        logger.warning(msg)
        raise RuntimeError(msg)


def test_compile_of_two_polynomials() -> None:
    """Test that the combination of two polynomials compiles correctly."""
    degree1 = 2
    degree2 = 3
    model1 = PolynomialBasis(degree=degree1)
    model2 = PolynomialBasis(degree=degree2)
    combined_model = model1 + model2
    theta_combined = np.concatenate(
        [
            rng.uniform(low=-1.0, high=1.0, size=degree1),
            rng.uniform(low=-1.0, high=1.0, size=degree2),
        ],
    )
    x = np.array([0.0, 1.0, 2.0])  # input values
    expected_output = np.zeros_like(x)
    for i in range(degree1):
        expected_output += theta_combined[i] * x ** (i + 1)
    for j in range(degree2):
        expected_output += theta_combined[degree1 + j] * x ** (j + 1)
    compiled_combined_model = compile_model(combined_model)
    output = compiled_combined_model(theta_free=theta_combined, x=x)
    if np.allclose(output, expected_output):
        logger.info("Combined polynomial model evaluation matches expected output.")
    else:
        msg = f"Combined polynomial model evaluation {output} does not "
        msg += f"match expected {expected_output}."
        logger.warning(msg)
        raise RuntimeError(msg)

    # Check that it raises an error when given wrong theta length
    theta_wrong_length = theta_combined[:-1]  # one fewer parameter
    with pytest.raises(ValueError, match="free"):
        compiled_combined_model(theta_free=theta_wrong_length, x=x)
    theta_wrong_length = [*theta_combined, 0.01]  # one extra parameter
    with pytest.raises(ValueError, match="free"):
        compiled_combined_model(theta_free=theta_wrong_length, x=x)

    # Check that wrong Parametrization raises an error
    wrong_parametrization = Parametrization.identity(full_ndim=len(theta_combined) + 3)
    with pytest.raises(ValueError, match="Expected"):
        compile_model(combined_model, parametrization=wrong_parametrization)
    wrong_parametrization = Parametrization.identity(full_ndim=len(theta_combined) - 2)
    with pytest.raises(ValueError, match="Expected"):
        compile_model(combined_model, parametrization=wrong_parametrization)


@pytest.mark.parametrize("shared_degree", ["first", "second", "both", "all"])
def test_compile_model_shared_parametrization(shared_degree: int | str) -> None:
    """Test that a model with shared parameters compiles correctly."""
    degree = 2
    model1 = PolynomialBasis(degree=degree, name="poly1")
    model2 = PolynomialBasis(degree=degree, name="poly2")
    combined_model = model1 + model2
    full_ndim = combined_model.n_params
    if shared_degree == "first":
        shared_groups = [[0, 2]]
    if shared_degree == "second":
        shared_groups = [[1, 3]]
    if shared_degree == "both":
        shared_groups = [[0, 2], [1, 3]]
    if shared_degree == "all":
        shared_groups = [[0, 1, 2, 3]]

    parametrization = Parametrization.from_shared(
        full_ndim=full_ndim,
        shared_groups=shared_groups,
    )

    compiled_model = compile_model(
        model=combined_model,
        parametrization=parametrization,
    )
    theta_free = rng.uniform(low=-1.0, high=1.0, size=compiled_model.n_params_free)
    x = np.array([0.0, 1.0, 2.0])  # input values
    output = compiled_model(theta_free=theta_free, x=x)
    # We won't check the exact output here, but we can check that it runs and
    # returns an array of the correct shape.
    if isinstance(output, np.ndarray) and output.shape == x.shape:
        msg = "Compiled model with shared parametrization evaluated "
        msg += f"successfully with output shape {output.shape}."
        logger.info(msg)
    else:
        msg = f"Compiled model output is of type {type(output)}, "
        msg += f"expected numpy.ndarray with shape {x.shape}."
        logger.warning(msg)
        raise TypeError(msg)

    # Check returns error when given wrong theta length
    theta_wrong_length = [*theta_free, 0.01]  # one extra parameter
    with pytest.raises(ValueError, match="free"):
        compiled_model(theta_free=theta_wrong_length, x=x)
    theta_wrong_length = theta_free[:-1]  # one fewer parameter
    with pytest.raises(ValueError, match="free"):
        compiled_model(theta_free=theta_wrong_length, x=x)


def test_compile_unary_operator() -> None:
    """Test that a model with a unary operator compiles correctly."""
    degree = 2
    model = -PolynomialBasis(degree=degree)
    compiled_model = compile_model(model)
    theta = rng.uniform(low=-1.0, high=1.0, size=degree)  # coefficients for x, x^2
    x = np.array([0.0, 1.0, 2.0])  # input values
    expected_output = np.zeros_like(x)
    for i in range(degree):
        expected_output -= theta[i] * x ** (i + 1)
    output = compiled_model(theta_free=theta, x=x)
    if np.allclose(output, expected_output):
        logger.info("Compiled model with unary operator evaluated successfully.")
    else:
        msg = f"Compiled model with unary operator evaluation {output} does not "
        msg += f"match expected {expected_output}."
        logger.warning(msg)
        raise RuntimeError(msg)
