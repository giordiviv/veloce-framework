"""Test models and their implementation."""  # noqa: INP001

import logging

import numpy as np
import pytest

from velocefw.model import FixedConstant, PolynomialBasis

logger = logging.getLogger(__name__)


def test_evaluate_shape_polynomial() -> None:
    """Test that the shape of the output of the polynomial model is correct."""
    degree = 3
    model = PolynomialBasis(degree=degree)
    theta = np.array([1.0, 0.5, 0.25])  # coefficients for x, x^2, x^3
    x = np.array([0.0, 1.0, 2.0])  # input values
    output = model.evaluate(theta=theta, x=x)
    if output.shape == x.shape:
        logger.info("Output shape matches input shape.")
    else:
        msg = f"Output shape {output.shape} does not match input shape {x.shape}."
        logger.warning(msg)
        raise TypeError(msg)


def test_evaluate_type_polynomial() -> None:
    """Test that the type of the output of the polynomial model is correct."""
    degree = 3
    model = PolynomialBasis(degree=degree)
    theta = np.array([1.0, 0.5, 0.25])  # coefficients for x, x^2, x^3
    x = 3.0  # input values
    output = model.evaluate(theta=theta, x=x)
    if isinstance(output, np.ndarray):
        logger.info("Output is of type numpy.ndarray.")
    else:
        msg = f"Output is of type {type(output)}, expected numpy.ndarray."
        logger.warning(msg)
        raise TypeError(msg)


def test_evaluate_polynomial() -> None:
    """Test that the polynomial model evaluates correctly."""
    degree = 3
    model = PolynomialBasis(degree=degree)
    theta = np.array([1.0, 0.5, 0.25])  # coefficients for x, x^2, x^3
    x = np.array([0.0, 1.0, 2.0])  # input values
    expected_output = theta[0] * x + theta[1] * x**2 + theta[2] * x**3
    output = model.evaluate(theta=theta, x=x)
    if np.allclose(output, expected_output):
        logger.info("Polynomial evaluation matches expected output.")
    else:
        msg = (
            f"Polynomial evaluation {output} does not match expected {expected_output}."
        )
        logger.warning(msg)
        raise TypeError(msg)


def test_combination_fixedconstant_polynomial() -> None:
    """Test that the combination fixed constant + polynomial model."""
    # Define the models
    constant_value = 2.0
    constant_model = FixedConstant(value=constant_value)
    degree = 2
    polynomial_model = PolynomialBasis(degree=degree)
    # Combine the models
    combined_model = constant_model + polynomial_model
    # Define parameters and input
    theta_polynomial = np.array([1.0, 0.5])  # coefficients for x, x^2
    x = np.array([0.0, 1.0, 2.0])  # input values

    # Evaluate the combined model
    output = combined_model.evaluate(theta=theta_polynomial, x=x)
    # Expected output is the constant value plus the polynomial evaluation
    expected_polynomial = theta_polynomial[0] * x + theta_polynomial[1] * x**2
    expected_output = constant_value + expected_polynomial
    if np.allclose(output, expected_output):
        logger.info("Combined model evaluation matches expected output.")
    else:
        msg = f"Model evaluation {output} does not match expected {expected_output}."
        logger.warning(msg)
        raise TypeError(msg)


@pytest.mark.parametrize("invalid_degree", [0, -1, -5])
def test_polynomial_degree_must_be_at_least_one(invalid_degree: int) -> None:
    """Initializing PolynomialBasis with degree < 1 raises ValueError."""
    with pytest.raises(ValueError, match="degree"):
        PolynomialBasis(degree=invalid_degree)


@pytest.mark.parametrize("invalid_degree", [1.5, None, "five"])
def test_polynomial_degree_must_be_integer(invalid_degree: int) -> None:
    """Initializing PolynomialBasis with a non-integer degree raises TypeError."""
    with pytest.raises(TypeError, match="degree"):
        PolynomialBasis(degree=invalid_degree)


def test_exponential_polynomial_combination() -> None:
    """Test that a exponential of a model works correctly."""
    # Define the models
    degree = 2
    polynomial_model = PolynomialBasis(degree=degree)
    exponential_model = np.exp(1) ** polynomial_model
    # Define parameters and input
    theta_polynomial = np.array([1.0, 0.5])  # coefficients for x, x^2
    x = np.array([0.0, 1.0, 2.0])  # input values

    # Evaluate the combined model
    output = exponential_model.evaluate(theta=theta_polynomial, x=x)
    expected_polynomial = theta_polynomial[0] * x + theta_polynomial[1] * x**2
    expected_output = np.exp(expected_polynomial)
    if np.allclose(output, expected_output):
        logger.info("Exponential of polynomial evaluation matches expected output.")
    else:
        msg = f"Model evaluation {output} does not match expected {expected_output}."
        logger.warning(msg)
        raise TypeError(msg)


def test_layout_two_polynomials() -> None:
    """Test that a layout with two polynomial models works correctly."""
    degree1 = 2
    degree2 = 3
    model1 = PolynomialBasis(degree=degree1, name="poly1")
    model2 = PolynomialBasis(degree=degree2, name="poly2")
    combined_model = model1 + model2
    full_ndim = combined_model.n_params
    layout = combined_model.layout()
    if layout.ndim == full_ndim:
        msg = "Layout ndim matches full_ndim."
        logger.info(msg)
    else:
        msg = f"Layout ndim {layout.ndim} does not match full_ndim {full_ndim}."
        logger.warning(msg)
        raise TypeError(msg)
