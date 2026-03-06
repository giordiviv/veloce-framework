"""Package containing implemented models.

Currently, the following models are implemented:

- Constant
    a simple constant model, defined as f(x) = c, where c is the constant value.
    The model has a single parameter, which is the constant value.

- PolynomialBasis
    a zero-mean polynomial model, defined as a1*x + a2*x^2 + ...  + an*x^n.
    The parameters of the model are the coefficients of the polynomial terms.

- FourierSeries
    a zero-mean Fourier series model, defined as a sum of harmonics
    an*cos(2*pi*n*x/P) + bn*sin(2*pi*n*x/P).
"""

from velocefw.model.implemented_models.constant import Constant
from velocefw.model.implemented_models.fourier_series import (
    FourierSeries,
    calculate_phase,
)
from velocefw.model.implemented_models.polynomial import PolynomialBasis

__all__ = [
    "Constant",
    "FourierSeries",
    "PolynomialBasis",
    "calculate_phase",
]
