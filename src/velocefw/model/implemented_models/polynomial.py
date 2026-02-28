"""Implementation of a polynomial model."""

import logging

import numpy as np

from velocefw.model.base import BaseModel

logger = logging.getLogger(__name__)


class PolynomialBasis(BaseModel):
    """Zero-mean polynomial model: a1*x + a2*x^2 + ... + an*x^n.

    The model is defined as a polynomial of degree `degree` with zero mean
    (i.e., no constant term). The parameters of the model are the coefficients
    of the polynomial terms, which are passed as arguments to the `evaluate`
    method.

    NOTE: The constant term was removed to avoid degeneracies when combining
    this model with others (e.g., a constant offset), and to allow for better
    interpretability of the parameters as the amplitude of the polynomial terms.

    Parameters (theta_local):
      - a1, ..., an: coefficients of the polynomial terms.
    """

    def __init__(self, degree: int, name: str = "poly") -> None:
        """Initialize the polynomial basis model."""
        if not isinstance(degree, int):
            msg = f"degree must be an integer, got {type(degree).__name__}."
            logger.error(msg)
            raise TypeError(msg)

        if degree < 1:
            msg = f"degree must be at least 1, got {degree}."
            logger.error(msg)
            raise ValueError(msg)

        super().__init__(name=name)
        self.degree = degree

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self.degree

    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
        return [f"a{i}" for i in range(1, self.degree + 1)]

    def evaluate(
        self,
        theta: list | np.ndarray,
        x: float | list | np.ndarray,
        **kwargs: object,  # noqa: ARG002
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        theta : list | np.ndarray
            The parameters of the model, which are the coefficients of the
            polynomial terms.
        x : float | list | np.ndarray
            The input values at which to evaluate the model.
        **kwargs : object
            Additional keyword arguments that may be needed for evaluation (not
            used in this model).

        Returns
        -------
        np.ndarray
            The model predictions for the corresponding inputs.

        """
        x = np.asarray(x, float)
        theta = np.asarray(theta, float).ravel()
        result = np.zeros_like(x)
        for i in range(1, self.degree + 1):
            result += float(theta[i - 1]) * x**i
        return result
