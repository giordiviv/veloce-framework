"""Fourier series model implementation.

The models are zero-mean Fourier series, that currently only support a single
variable and are of the form:

f(x) = ... + an * cos(2 * pi * n * x / P) + bn * sin(2 * pi * n * x / P)
"""

import logging
from itertools import batched

import numpy as np

from velocefw.model.base import BaseModel

logger = logging.getLogger(__name__)


def calculate_phase(
    x: float | list | np.ndarray,
    period: float,
    epoch: float,
) -> np.ndarray:
    """Calculate the phase of the input variable x given period and epoch."""
    x = np.asarray(x, float)
    return np.remainder(np.subtract(x, epoch) / period, 1.0)  # Phase in [0, 1) range


class FourierSeries(BaseModel):
    """Fourier series model.

    The model is defined as a Fourier series of harmonics `nharm` with zero mean
    (i.e., no constant term). The parameters of the model are the coefficients
    of the Fourier series harmonics, which are passed as arguments to the
    `evaluate` method.

    Parametrization: cos-sin
    f(x) = ... + an * cos(2 * pi * n * x / P) + bn * sin(2 * pi * n * x / P)

    NOTE: The constant term was removed to avoid degeneracies when combining
    this model with others (e.g., a constant offset), and to allow for better
    interpretability of the parameters as the amplitude of the FS harmonics.

    Parameters (theta_local):
      - a1, b1, ..., an, bn: coefficients of the FS harmonics.
    """

    def __init__(self, nharm: int, name: str = "FS") -> None:
        """Initialize the Fourier series model."""
        if not isinstance(nharm, int):
            msg = f"nharm must be an integer, got {type(nharm).__name__}."
            logger.error(msg)
            raise TypeError(msg)

        if nharm < 1:
            msg = f"nharm must be at least 1, got {nharm}."
            logger.error(msg)
            raise ValueError(msg)

        super().__init__(name=name)
        self.nharm = nharm

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return 2 * self.nharm

    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
        # Alternating a1, b1, a2, b2, ..., an, bn
        return [f"{coef}{i}" for i in range(1, self.nharm + 1) for coef in ("a", "b")]

    def evaluate(
        self,
        theta: list | np.ndarray,
        x: float | list | np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Evaluate the model."""
        theta = np.asarray(theta, float).ravel()
        x = np.asarray(x, float)
        period = kwargs.get("P", 1.0)  # Default period is 1.0 if not provided
        epoch = kwargs.get("E", 0.0)  # Default epoch is 0.0 if not provided

        # Validate that the required keyword arguments are provided
        if not isinstance(period, (int, float)):
            msg = f"Period P must be a number, got {type(period).__name__}."
            logger.error(msg)
            raise TypeError(msg)
        if period <= 0:
            msg = f"Period P must be positive, got {period}."
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(epoch, (int, float)):
            msg = f"Epoch E must be a number, got {type(epoch).__name__}."
            logger.error(msg)
            raise TypeError(msg)

        # Calculate the phase of x given period and epoch
        x_phase = calculate_phase(x, period, epoch)

        result = np.zeros_like(x)

        # Batch theta into pairs of (a_n, b_n)
        iter_coeff = enumerate(
            batched(theta, 2),
            1,
        )
        for n, (a_n, b_n) in iter_coeff:
            factor = 2 * np.pi * n
            result += a_n * np.cos(factor * x_phase) + b_n * np.sin(factor * x_phase)

        return result
