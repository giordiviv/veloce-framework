"""Constant model.

It can be used to represent a constant term in a model expression, which can be
useful for modeling offsets or baseline levels in data. The model has a single
parameter, which is the constant value. The evaluate method simply returns this
constant value for any input. This model can be combined with other models to
create more complex expressions, and the constant term can be easily fitted to
data to account for any baseline offset.
"""

import logging

import numpy as np

from velocefw.model.base import BaseModel

logger = logging.getLogger(__name__)


class Constant(BaseModel):
    """Constant model.

    This model represents a constant term in a model expression, which can be
    useful for modeling offsets or baseline levels in data. The model has a
    single parameter, which is the constant value. The evaluate method simply
    returns this constant value for any input. This model can be combined with
    other models to create more complex expressions, and the constant term can
    be easily fitted to data to account for any baseline offset.

    Parameters (theta_local):
      - c: the constant value.
    """

    def __init__(self, name: str = "constant") -> None:
        """Initialize the constant model."""
        super().__init__(name=name)

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return 1

    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
        return ["c"]

    def evaluate(
        self,
        theta: list | np.ndarray,
        x: float | list | np.ndarray,
        **kwargs: object,  # noqa: ARG002
    ) -> np.ndarray:
        """Evaluate the model."""
        theta = np.asarray(theta, float).ravel()
        return np.full_like(np.asarray(x, float), float(theta[0]))
