"""Base classes and utilities for model definition and composition.

BaseModel is the core class that all models inherit from. It defines the basic
interface for models, including parameter handling, evaluation, and algebraic
composition.

The Layout class defines the structure of the global parameter vector for a
model, including metadata for each parameter.

The _LayoutBuilder class is a helper for constructing the Layout from a model
expression tree.

The FixedConstant class is a simple model that represents a constant value,
which can be used in model expressions to allow for seamless integration of
scalar constants.

The UnaryOp and BinaryOp classes represent unary and binary operations on
models, allowing for algebraic composition of models using standard operators
like +, -, *, /, and **.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .parametrization import ParamMeta

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass
class Layout:
    """Layout of the global parameter vector."""

    ndim: int
    params_meta: list[ParamMeta]
    name_to_index: Mapping[str, int] | None = None

    @property
    def names(self) -> list[str]:
        """List of parameter names in the global theta vector."""
        out = []
        for param in self.params_meta:
            if param.global_name:
                out.append(f"{param.global_name}")
            elif param.local_name:
                out.append(f"{param.local_name}")
            else:
                out.append(f"p{param.local_index}")
        self.name_to_index = {name: i for i, name in enumerate(out)}
        return out

    def find(self, name: str) -> int:
        """Find the global index of a parameter by full name.

        Parameters
        ----------
        name : str
            The full name of the parameter to find.

        Returns
        -------
        int
            The global index of the parameter in the full theta vector.

        Raises
        ------
        KeyError
            If the parameter name is not found in the layout.

        """
        if self.name_to_index is not None:
            try:
                return self.name_to_index[name]
            except KeyError:
                logger.debug(
                    "Parameter name '%s' not found in layout name_to_index mapping."
                    " Falling back to linear search in names list.",
                    name,
                )
        names = self.names
        try:
            return names.index(name)
        except ValueError:
            msg = f"Parameter name '{name}' not found in layout."
            logger.exception(msg)
            raise


class _LayoutBuilder:
    """Helper class to build a Layout from an expression tree."""

    def __init__(self) -> None:
        """Initialize an empty _LayoutBuilder."""
        self.params_meta: list[ParamMeta] = []
        self.current_index = 0

    def add_node_params(self, node: BaseModel) -> None:
        """Add the parameters of the input node to the layout.

        The local name of the parameters are maintained if possible, but the
        global name is made unique by prefixing with the node name and adding a
        suffix if needed to avoid conflicts with existing global names. This
        allows for better interpretability of the parameters in the context of
        the model expression tree, while ensuring that each parameter has a
        unique global name in the layout. The global name is used for the final
        parameter vector, while the local name is used for the parameters within
        the context of the node.

        Parameters
        ----------
        node : BaseModel
            The model node whose parameters are being added to the layout.

        """
        for local_index, local_name in enumerate(node.param_names()):
            params_names = {param.global_name for param in self.params_meta}
            global_name = (
                f"{node.name}.{local_name}"
                if local_name in params_names
                else local_name
            )
            i = 1
            while global_name in params_names:
                global_name = f"{node.name}{i}.{local_name}"
                i += 1
            self.params_meta.append(
                ParamMeta(
                    index=self.current_index + local_index,
                    local_index=local_index,
                    local_name=local_name,
                    global_name=global_name,
                ),
            )
        self.current_index += len(node.param_names())

    def build(self) -> Layout:
        return Layout(ndim=self.current_index, params_meta=self.params_meta)


class BaseModel:
    """Base class for all models."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize the BaseModel by assigning a name.

        Parameters
        ----------
        name : str | None, optional
            The name of the model, by default None

        """
        self._name = name

    # To be implemented by subclasses --------------------------------------------------
    @property
    def n_params(self) -> int:
        """Number of parameters in the model."""
        msg = "Subclasses of BaseModel must implement n_params property."
        logger.exception(msg)
        raise NotImplementedError(msg)

    def param_names(self) -> list[str]:
        """Parameter names for the model.

        Should be overridden by subclasses to provide meaningful parameter
        names. By default, it returns generic names like p0, p1, ...,
        p{n_params-1}.
        """
        return [f"p{i}" for i in range(self.n_params)]

    def evaluate(
        self,
        x: float | list | np.ndarray,  # noqa: ARG002
        theta: list | np.ndarray,  # noqa: ARG002
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> np.ndarray:
        """Evaluate the model or submodel at the given input `x` and parameters `theta`.

        Parameters
        ----------
        x : float | list | np.ndarray
            The input values at which to evaluate the model.
        theta : list | np.ndarray
            The model parameters.
        **kwargs : dict[str, Any]
            Additional keyword arguments that may be needed for evaluation
            (e.g., time of observation, etc.).

        Returns
        -------
        np.ndarray
            Array of model predictions for the corresponding inputs.


        Raises
        ------
        NotImplementedError
            Models needs to implement this method

        """
        msg = "Subclasses of BaseModel must implement evaluate method."
        logger.exception(msg)
        raise NotImplementedError(msg)

    # Implemented methods --------------------------------------------------------------
    @property
    def name(self) -> str:
        """Get the name of the model."""
        if self._name:
            return self._name
        return self.__class__.__name__.lower()

    def layout(self) -> Layout:
        """Build and return the Layout of the model."""
        b = _LayoutBuilder()
        _walk_layout(self, b)
        return b.build()

    @property
    def ndim(self) -> int:
        """Total number of parameters in the model."""
        return self.layout().ndim

    def evaluate_local(
        self,
        x: float | list | np.ndarray,
        theta_local: list | np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Evaluate the model at the given input `x` and local parameters `theta_local`.

        This method is used internally by the BinaryOp and UnaryOp classes to
        evaluate the model with the correct subset of parameters.

        Parameters
        ----------
        x : float | list | np.ndarray
            The input values at which to evaluate the model.
        theta_local : list | np.ndarray
            The model parameters for the local submodel.
        **kwargs : dict[str, Any]
            Additional keyword arguments that may be needed for evaluation
            (e.g., time of observation, etc.).


        Returns
        -------
        np.ndarray
            Array of model predictions for the corresponding inputs.

        Raises
        ------
        ValueError
            When the length of `theta_local` does not match the expected number
            of parameters for the model.

        """
        theta_local = np.asarray(theta_local, float).ravel()
        if theta_local.size != self.n_params:
            msg = f"{self.name} expected {self.n_params} params, got {theta_local.size}"
            logger.error(msg)
            raise ValueError(msg)
        return self.evaluate(x, theta_local, **kwargs)

    # Algebraic composition operators --------------------------------------------------
    def __add__(self, other: float | np.number | BaseModel) -> BinaryOp:
        """Addition operator for model composition."""
        return BinaryOp(self, _wrap_scalar(other), np.add, "+")

    def __radd__(self, other: float | np.number | BaseModel) -> BinaryOp:
        """Right addition operator for model composition."""
        return BinaryOp(_wrap_scalar(other), self, np.add, "+")

    def __sub__(self, other: float | np.number | BaseModel) -> BinaryOp:
        """Subtraction operator for model composition."""
        return BinaryOp(self, _wrap_scalar(other), np.subtract, "-")

    def __rsub__(self, other: float | np.number | BaseModel) -> BinaryOp:
        """Right subtraction operator for model composition."""
        return BinaryOp(_wrap_scalar(other), self, np.subtract, "-")

    def __mul__(self, other: float | np.number | BaseModel) -> BinaryOp:
        """Multiplication operator for model composition."""
        return BinaryOp(self, _wrap_scalar(other), np.multiply, "*")

    def __rmul__(self, other: float | np.number | BaseModel) -> BinaryOp:
        """Right multiplication operator for model composition."""
        return BinaryOp(_wrap_scalar(other), self, np.multiply, "*")

    def __truediv__(self, other: float | np.number | BaseModel) -> BinaryOp:
        """Division operator for model composition."""
        return BinaryOp(self, _wrap_scalar(other), np.divide, "/")

    def __rtruediv__(self, other: float | np.number | BaseModel) -> BinaryOp:
        """Right division operator for model composition."""
        return BinaryOp(_wrap_scalar(other), self, np.divide, "/")

    def __pow__(self, other: float | np.number | BaseModel) -> BinaryOp:
        """Power operator for model composition."""
        return BinaryOp(self, _wrap_scalar(other), np.power, "**")

    def __neg__(self) -> UnaryOp:
        """Negation operator for model composition."""
        return UnaryOp(self, np.negative, "-")


def _wrap_scalar(obj: float | np.number | BaseModel) -> BaseModel:
    """Return a `BaseModel` instance representing the input object.

    Create a `FixedConstant` model from a scalar value, or return the input
    if it's already a `BaseModel`.

    Parameters
    ----------
    obj : float | np.number | BaseModel
        The object to convert into a `BaseModel`. If it's a scalar (float or
        numpy number), it will be wrapped in a `FixedConstant` model. If it's
        already a `BaseModel`, it will be returned as is.

    Returns
    -------
    BaseModel
        The wrapped `BaseModel` instance.

    Raises
    ------
    TypeError
        When the input object is not a scalar value or a `BaseModel`.

    """
    if isinstance(obj, BaseModel):
        return obj
    if isinstance(obj, (int, float, np.number)):
        return FixedConstant(value=float(obj))
    msg = f"Cannot convert object of type {type(obj)} into a BaseModel"
    logger.error(msg)
    raise TypeError(msg)


def _walk_layout(m: BaseModel, b: _LayoutBuilder) -> None:
    """Walk the binary tree of a composite model to set the _LayoutBuilder.

    Parameters
    ----------
    m : BaseModel
        The model node to process in the layout building. This can be a leaf
        node (primitive or user-defined model) or an internal node (UnaryOp or
        BinaryOp).
    b : _LayoutBuilder
        The _LayoutBuilder instance that is being used to build the layout. This object
        is modified in-place as the tree is walked, with parameters from each node
        being added to the builder.

    """
    if isinstance(m, FixedConstant):
        return

    if isinstance(m, UnaryOp):
        _walk_layout(m.operand, b)
        return

    # Use model label instead of left/right/operand
    if isinstance(m, BinaryOp):
        _walk_layout(m.left, b)
        _walk_layout(m.right, b)
        return

    # leaf / primitive or user-defined model: allocate its local block
    b.add_node_params(m)


class FixedConstant(BaseModel):
    """Fixed constant model.

    Used when a scalar value is used in a model expression. It has no parameters
    and always evaluates to the fixed value. This allows for seamless
    integration of scalar constants into model expressions without needing to
    treat them as special cases. The value is stored as a float, and the model
    can be evaluated at any input `x` to return an array of the same shape
    filled with the constant value.
    """

    def __init__(self, value: float, name: str | None = None) -> None:
        """Initialize the FixedConstant model with a fixed value.

        Parameters
        ----------
        value : float
            The constant value that this model will always return when evaluated.
        name : str | None, optional
            The name of the model. If None, the name is set to "fixed_c".

        """
        self._value = float(value)
        if name is None:
            name = "fixed_c"
        super().__init__(name=name)

    @property
    def n_params(self) -> int:
        """Number of parameters in the model (always 0 for FixedConstant)."""
        return 0

    def evaluate(
        self,
        x: float | list | np.ndarray,
        theta: list | np.ndarray,  # noqa: ARG002
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> np.ndarray:
        """Evaluate constant function.

        Parameters
        ----------
        x : float | list | np.ndarray
            Array of input values at which to evaluate the model. The output
            will have the same shape as `x`, but the values will be constant
            regardless of `x`.
        theta : list | np.ndarray
            The model parameters (not used for FixedConstant, but included for
            compatibility with the evaluate method signature).
        **kwargs : dict[str, Any]
            Additional keyword arguments that may be needed for evaluation (not
            used for FixedConstant, but included for compatibility with the
            evaluate method signature).

        Returns
        -------
        np.ndarray
            Return a constant array of the same shape as `x` filled with the
            fixed value of the model.

        """
        return np.full_like(x, self._value, dtype=float)


class Constant(BaseModel):
    def __init__(self, name: str = "const") -> None:
        super().__init__(name=name)

    @property
    def n_params(self) -> int:
        return 1

    def evaluate(
        self,
        x: np.ndarray,
        theta_local: np.ndarray,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> np.ndarray:
        return np.full_like(x, float(theta_local[0]), dtype=float)

    def param_names(self) -> list[str]:
        return ["c"]


class UnaryOp(BaseModel):
    def __init__(self, operand: BaseModel, op_func: np.ufunc, symbol: str) -> None:
        self.operand = operand
        self.op_func = op_func
        self.symbol = symbol
        super().__init__(name=f"{symbol}({operand.name})")

    @property
    def n_params(self) -> int:
        return self.operand.n_params

    def evaluate(
        self,
        x: float | list | np.ndarray,
        theta: list | np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        return self.op_func(self.operand.evaluate(x, theta, **kwargs))


class BinaryOp(BaseModel):
    def __init__(
        self,
        left: BaseModel,
        right: BaseModel,
        op_func: np.ufunc,
        symbol: str,
    ) -> None:
        self.left = left
        self.right = right
        self.op_func = op_func
        self.symbol = symbol
        super().__init__(name=f"({left.name} {symbol} {right.name})")

    @property
    def n_params(self) -> int:
        return self.left.n_params + self.right.n_params

    def evaluate(
        self,
        x: float | list | np.ndarray,
        theta: list | np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        index_left = self.left.n_params
        theta_left = theta[:index_left]
        theta_right = theta[index_left:]

        return self.op_func(
            self.left.evaluate_local(x, theta_left, **kwargs),
            self.right.evaluate_local(x, theta_right, **kwargs),
        )
