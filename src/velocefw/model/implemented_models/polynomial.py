"""Implementation of a polynomial model."""

import numpy as np

from velocefw.model.base import BaseModel


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

    def __init__(self, degree: int, name: str = "polynomial") -> None:
        """Init."""
        super().__init__(name=name)
        self.degree = degree

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self.degree

    def param_names(self) -> list[str]:
        """Parameter names."""
        return [f"a{i}" for i in range(1, self.degree + 1)]

    def evaluate(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        **kwargs: object,  # noqa: ARG002
    ) -> np.ndarray:
        """Evaluate the model."""
        result = np.zeros_like(x)
        for i in range(1, self.degree + 1):
            result += float(theta[i - 1]) * x**i
        return result
