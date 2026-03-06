"""Test models and their implementation."""  # noqa: INP001

import logging

import numpy as np
import pytest

from velocefw.model import (
    BaseModel,
    Constant,
    FixedConstant,
    FourierSeries,
    PolynomialBasis,
    calculate_phase,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("x", [3.0, [0.0, 1.0, 2.0], np.array([0.0, 1.0, 2.0])])
def test_evaluate_shape(x: float | list | np.ndarray) -> None:
    """Test that the shape of the output of the polynomial model is correct."""
    degree = 3
    model = PolynomialBasis(degree=degree)
    theta = np.array([1.0, 0.5, 0.25])  # coefficients for x, x^2, x^3
    output = model.evaluate(theta=theta, x=x)
    inputs = np.asarray(x, float)
    if output.shape == inputs.shape:
        logger.info("Output shape matches input shape.")
    else:
        msg = f"Output shape {output.shape} does not match input shape {inputs.shape}."
        logger.warning(msg)
        raise TypeError(msg)

    # Test that the shape of the output of the combined model is correct
    model = Constant() + model
    output = model.evaluate(theta=[1.0, *theta.tolist()], x=x)
    inputs = np.atleast_1d(x)
    if output.shape == inputs.shape:
        logger.info("Output shape matches input shape for combined model.")
    else:
        msg = f"Output shape {output.shape} does not match input "
        msg += f"shape {inputs.shape} for combined model."
        logger.warning(msg)
        raise TypeError(msg)


@pytest.mark.parametrize("x", [3.0, [0.0, 1.0, 2.0], np.array([0.0, 1.0, 2.0])])
def test_evaluate_type(x: float | list | np.ndarray) -> None:
    """Test that the type of the output of the polynomial model is correct."""
    degree = 3
    model = PolynomialBasis(degree=degree)
    theta = np.array([1.0, 0.5, 0.25])  # coefficients for x, x^2, x^3
    output = model.evaluate(theta=theta, x=x)
    if isinstance(output, np.ndarray):
        logger.info("Output is of type numpy.ndarray.")
    else:
        msg = f"Output is of type {type(output)}, expected numpy.ndarray."
        logger.warning(msg)
        raise TypeError(msg)

    # Test that the type of the output of the combined model is correct
    model = Constant() + model
    output = model.evaluate(theta=[1.0, *theta.tolist()], x=x)
    if isinstance(output, np.ndarray):
        logger.info("Output is of type numpy.ndarray for combined model.")
    else:
        msg = f"Output is of type {type(output)}, "
        msg += "expected numpy.ndarray for combined model."
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

    # Test that the combined model evaluates correctly
    model = Constant() + model
    expected_output += 1.0  # add the constant term
    output = model.evaluate(theta=[1.0, *theta.tolist()], x=x)
    if np.allclose(output, expected_output):
        logger.info("Combined model evaluation matches expected output.")
    else:
        msg = f"Combined model evaluation {output} does "
        msg += f"not match expected {expected_output}."
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


def test_wrong_model_initialization() -> None:
    """Test that initializing a model with wrong parameters raises an error."""
    with pytest.raises(NotImplementedError, match="evaluate"):

        class WrongModelMissingEvaluate(BaseModel):
            pass

    with pytest.raises(TypeError, match="evaluate"):

        class WrongModelMissingKwargs(BaseModel):
            def evaluate(
                self,
                theta: list | np.ndarray,
                x: float | list | np.ndarray,
            ) -> np.ndarray:
                return np.array(x) * theta

    with pytest.raises(TypeError, match="evaluate"):

        class WrongModelMissingX(BaseModel):
            def evaluate(
                self,
                theta: list | np.ndarray,
                **kwargs: object,  # noqa: ARG002
            ) -> np.ndarray:
                return np.array([0.0]) * theta

    with pytest.raises(NotImplementedError, match="n_params"):

        class WrongModelMissingNParams(BaseModel):
            def evaluate(
                self,
                theta: list | np.ndarray,
                x: float | list | np.ndarray,
                **kwargs: object,  # noqa: ARG002
            ) -> np.ndarray:
                return np.array(x) * theta


def test_model_names_mapping() -> None:
    """Test that the model names mapping in the layout is correct."""
    # Define a simple layout with multiple models
    constant_model = Constant(name="constant")
    polynomial_model = PolynomialBasis(degree=2, name="polynomial")
    combined_model = constant_model + polynomial_model
    # Get the model names mapping
    available_names, map_model_to_index = (
        combined_model.layout._available_names_mapped()  # noqa: SLF001
    )
    # Check that the available names are correct
    expected_names = {"constant", "polynomial"}
    if available_names != expected_names:
        msg = (
            f"Available names {available_names} do not match expected {expected_names}."
        )
        logger.warning(msg)
        raise TypeError(msg)

    # Check that the mapping from model names to parameter indices is correct
    expected_mapping = {
        "constant": [0],  # the constant model has 1 parameter at index 0
        "polynomial": [
            1,
            2,
        ],  # the polynomial model has 2 parameters at indices 1 and 2
    }
    if map_model_to_index != expected_mapping:
        msg = f"Model to index mapping {map_model_to_index} does not match "
        msg += f"expected {expected_mapping}."
        logger.warning(msg)
        raise TypeError(msg)


def test_model_names_mapping_for_models_sharing_name() -> None:
    """Test that the model names mapping is correct when models share the same name."""
    # Define a simple layout with multiple models sharing the same name
    poly_model_1 = PolynomialBasis(degree=2, name="poly")
    poly_model_2 = PolynomialBasis(degree=3, name="poly")
    combined_model = poly_model_1 + poly_model_2
    # Get the model names mapping
    available_names, map_model_to_index = (
        combined_model.layout._available_names_mapped()  # noqa: SLF001
    )
    # Check that the available names are correct
    expected_names = {"poly0", "poly1"}
    if available_names != expected_names:
        msg = (
            f"Available names {available_names} do not match expected {expected_names}."
        )
        logger.warning(msg)
        raise TypeError(msg)
    # Check that the mapping from model names to parameter indices is correct
    expected_mapping = {
        "poly0": [0, 1],  # the first polynomial model has
        "poly1": [2, 3, 4],  # the second polynomial model has
    }
    if map_model_to_index != expected_mapping:
        msg = f"Model to index mapping {map_model_to_index} does not match "
        msg += f"expected {expected_mapping}."
        logger.warning(msg)
        raise TypeError(msg)

    # Test mask
    mask_poly1 = combined_model.layout.mask(model_name="poly0")
    expected_mask_poly1 = np.array([True, True, False, False, False])
    if np.array_equal(mask_poly1, expected_mask_poly1):
        logger.info("Mask for poly0 matches expected mask.")
    else:
        msg = f"Mask for poly0 {mask_poly1} does not match "
        msg += f"expected {expected_mask_poly1}."
        logger.warning(msg)
        raise TypeError(msg)
    mask_poly2 = combined_model.layout.mask(model_name="poly1")
    expected_mask_poly2 = np.array([False, False, True, True, True])
    if np.array_equal(mask_poly2, expected_mask_poly2):
        logger.info("Mask for poly1 matches expected mask.")
    else:
        msg = f"Mask for poly1 {mask_poly2} does not match "
        msg += f"expected {expected_mask_poly2}."
        logger.warning(msg)
        raise TypeError(msg)


# Fourier series tests -----------------------------------------------------------------
def test_calculate_phase() -> None:
    """Test that the calculate_phase function works correctly."""
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    period = 1.0
    epoch = 0.0
    expected_phase = np.array([0.0, 0.25, 0.5, 0.75, 0.0])
    phase = calculate_phase(x=x, period=period, epoch=epoch)
    if np.allclose(phase, expected_phase):
        logger.info("Calculated phase matches expected phase.")
    else:
        msg = f"Calculated phase {phase} does not match expected {expected_phase}."
        logger.warning(msg)
        raise TypeError(msg)

    period = 0.5
    epoch = 0.25
    expected_phase = np.array([0.5, 0.0, 0.5, 0.0, 0.5])
    phase = calculate_phase(x=x, period=period, epoch=epoch)
    if np.allclose(phase, expected_phase):
        logger.info("Phases matches the expected phase for given period and epoch.")
    else:
        msg = f"Calculated phase {phase} does not match expected {expected_phase} "
        msg += "for different period and epoch."
        logger.warning(msg)
        raise TypeError(msg)


def test_fourier_series_parameter_names() -> None:
    """Test that the parameter names of the Fourier series model are correct."""
    nharm = 3
    model = FourierSeries(nharm=nharm)
    expected_param_names = ["a1", "b1", "a2", "b2", "a3", "b3"]
    if model.param_names == expected_param_names:
        logger.info("Fourier series parameter names match expected names.")
    else:
        msg = f"Fourier series parameter names {model.param_names} do not match "
        msg += f"expected {expected_param_names}."
        logger.warning(msg)
        raise TypeError(msg)


def test_fourier_series_evaluation() -> None:
    """Test that the Fourier series model evaluates correctly."""
    nharm = 2
    model = FourierSeries(nharm=nharm)
    theta = np.array([1.0, 0.5, 0.25, 0.125])  # a1, b1, a2, b2
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # input values
    period = 1.0
    epoch = 0.0
    expected_output = (
        theta[0] * np.cos(2 * np.pi * 1 * (x - epoch) / period)
        + theta[1] * np.sin(2 * np.pi * 1 * (x - epoch) / period)
        + theta[2] * np.cos(2 * np.pi * 2 * (x - epoch) / period)
        + theta[3] * np.sin(2 * np.pi * 2 * (x - epoch) / period)
    )
    output = model.evaluate(theta, x, period=period, epoch=epoch)
    if np.allclose(output, expected_output):
        logger.info("Fourier series evaluation matches expected output.")
    else:
        msg = f"Fourier series evaluation {output} does not "
        msg += f"match expected {expected_output}."
        logger.warning(msg)
        raise TypeError(msg)


def test_fourier_series_combination() -> None:
    """Test that a Fourier series combined with a polynomial evaluates correctly."""
    # Define the models
    nharm = 2
    fourier_model = FourierSeries(nharm=nharm)
    degree = 2
    polynomial_model = PolynomialBasis(degree=degree)
    combined_model = fourier_model + polynomial_model
    # Define parameters and input
    theta_fourier = np.array([1.0, 0.5, 0.25, 0.125])  # a1, b1, a2, b2
    theta_polynomial = np.array([0.5, 0.25])  # coefficients for x, x^2
    theta_combined = np.concatenate((theta_fourier, theta_polynomial))
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # input values
    period = 1.0
    epoch = 0.0

    # Evaluate the combined model
    output = combined_model.evaluate(
        theta=theta_combined,
        x=x,
        period=period,
        epoch=epoch,
    )
    expected_fourier = (
        theta_fourier[0] * np.cos(2 * np.pi * 1 * (x - epoch) / period)
        + theta_fourier[1] * np.sin(2 * np.pi * 1 * (x - epoch) / period)
        + theta_fourier[2] * np.cos(2 * np.pi * 2 * (x - epoch) / period)
        + theta_fourier[3] * np.sin(2 * np.pi * 2 * (x - epoch) / period)
    )
    expected_polynomial = theta_polynomial[0] * x + theta_polynomial[1] * x**2
    expected_output = expected_fourier + expected_polynomial
    if np.allclose(output, expected_output):
        msg = "Combined Fourier series and polynomial evaluation "
        msg += "matches expected output."
        logger.info(msg)
    else:
        msg = f"Combined model evaluation {output} does not "
        msg += f"match expected {expected_output}."
        logger.warning(msg)
        raise TypeError(msg)


def test_fourier_series_invalid_initialization() -> None:
    """Test that initializing FourierSeries with invalid nharm raises an error."""
    with pytest.raises(ValueError, match="nharm"):
        FourierSeries(nharm=0)

    with pytest.raises(ValueError, match="nharm"):
        FourierSeries(nharm=-1)

    with pytest.raises(TypeError, match="nharm"):
        FourierSeries(nharm=1.5)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="nharm"):
        FourierSeries(nharm=None)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="nharm"):
        FourierSeries(nharm="two")  # pyright: ignore[reportArgumentType]


def test_fourier_series_invalid_period() -> None:
    """Test that evaluating FourierSeries with invalid period raises an error."""
    model = FourierSeries(nharm=1)
    theta = np.array([1.0, 0.5])  # a1, b1
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # input values

    with pytest.raises(ValueError, match="Period"):
        model.evaluate(theta=theta, x=x, P=0.0, E=0.0)

    with pytest.raises(ValueError, match="Period"):
        model.evaluate(theta=theta, x=x, P=-1.0, E=0.0)

    with pytest.raises(TypeError, match="Period"):
        model.evaluate(theta=theta, x=x, P="one", E=0.0)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="Period"):
        model.evaluate(theta=theta, x=x, P=None, E=0.0)  # pyright: ignore[reportArgumentType]


def test_fourier_series_invalid_epoch() -> None:
    """Test that evaluating FourierSeries with invalid epoch raises an error."""
    model = FourierSeries(nharm=1)
    theta = np.array([1.0, 0.5])  # a1, b1
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # input values
    period = 1.0

    with pytest.raises(TypeError, match="Epoch"):
        model.evaluate(theta=theta, x=x, P=period, E="zero")  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="Epoch"):
        model.evaluate(theta=theta, x=x, P=period, E=None)  # pyright: ignore[reportArgumentType]
