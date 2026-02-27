"""Package containing implemented models.

Currently, the following models are implemented:

- PolynomialBasis
    a zero-mean polynomial model, defined as a1*x + a2*x^2 + ...  + an*x^n.
    The parameters of the model are the coefficients of the polynomial terms.



"""

from .polynomial import PolynomialBasis

__all__ = [
    "PolynomialBasis",
]
