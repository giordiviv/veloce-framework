"""Test models and their implementation."""  # noqa: INP001

import logging

from velocefw.model import Constant, FixedConstant, PolynomialBasis

logger = logging.getLogger(__name__)


def test_layout_polynomial() -> None:
    """Test that a layout with a polynomial model works correctly."""
    degree = 3
    model = PolynomialBasis(degree=degree)
    full_ndim = model.n_params
    layout = model.layout()
    if layout.ndim == full_ndim:
        msg = "Layout ndim matches full_ndim."
        logger.info(msg)
    else:
        msg = f"Layout ndim {layout.ndim} does not match full_ndim {full_ndim}."
        logger.warning(msg)
        raise TypeError(msg)


def test_layout_names() -> None:
    """Test that the layout names are correct."""
    degree = 3
    model = PolynomialBasis(degree=degree)
    layout = model.layout()
    expected_names = [f"a{i}" for i in range(1, degree + 1)]
    if layout.names == expected_names:
        msg = "Layout names match expected names."
        logger.info(msg)
    else:
        msg = (
            f"Layout names {layout.names} do not match expected names {expected_names}."
        )
        logger.warning(msg)
        raise ValueError(msg)


def test_layout_constant() -> None:
    """Test that a layout with a constant model works correctly."""
    model = Constant()
    full_ndim = 1
    layout = model.layout()
    if layout.ndim == full_ndim:
        msg = "Layout ndim matches full_ndim for constant model."
        logger.info(msg)
    else:
        msg = f"Layout ndim {layout.ndim} is not 1 for constant model."
        logger.warning(msg)
        raise TypeError(msg)


def test_layout_fixed_constant() -> None:
    """Test that a layout with a fixed constant model works correctly."""
    model = FixedConstant(value=2.0)
    full_ndim = 0
    layout = model.layout()
    if layout.ndim == full_ndim:
        msg = "Layout ndim matches full_ndim for fixed constant model."
        logger.info(msg)
    else:
        msg = f"Layout ndim {layout.ndim} is not 0 for fixed constant model."
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
