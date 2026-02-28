"""Model Framework.

This directory contains the core classes and functions used to define
mathematical models. In this framework, models are used to simply evaluate
mathematical expressions for given parameters and inputs (e.g., time, etc.).
They do not contain the values of the parameters, which are instead passed as
arguments to the evaluation method. This allows to better separate the concept
of a possible fitting solution from the model itself.

In this implementation, the class `BaseModel` defines the basic interface for a
model, which includes the `evaluate` method that evaluates the model for given
parameters and input values.  This class also allows for the construction of
complex models by combining simpler ones via classical mathematical operations
(addition, multiplication, exponential etc.). The user can easily build new
models by combining existing ones, or by inheriting from `BaseModel` and
implementing the `evaluate` method and the `n_params` property (also the
`param_names` method can be implemented for easier access of the parameter).

When complex models are built by combining simpler ones, the order and structure
of the parameters is automatically handled by the framework, and can be accessed
via the `layout` attribute of the model.

A `Parametrization` cna be defined to specify if certain relations between
parameters should be accounted for. So far, the supported relations are:
- identity: all parameters are free and independent.
- from_shared: parameters are shared according to specified groups of indices
that should share the same free parameter. This allows to easily implement tied
parameters, which are common in astronomical modeling (e.g., same amplitude for
different nodes, etc.).

The `compile_model` function can be used to compile a model for faster evaluation.
It returns a `CompiledModel` that avoids solving the possible model tree
everytime the model is evaluated. A `Parametrization` can be provided to the
compiled model (defualt is identity).

NOTE: With `theta` we indicate the parameters of the model, which are passed as
arguments to the `evaluate` method.
"""

from velocefw.model.base import BaseModel, FixedConstant, ParamMeta
from velocefw.model.compile import CompiledModel, compile_model
from velocefw.model.implemented_models import Constant, PolynomialBasis
from velocefw.model.parametrization import Parametrization

__all__ = [
    "BaseModel",
    "CompiledModel",
    "Constant",
    "FixedConstant",
    "ParamMeta",
    "Parametrization",
    "PolynomialBasis",
    "compile_model",
]
