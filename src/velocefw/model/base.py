"""Base classes and utilities for model definition and composition.

BaseModel is the core class that all models inherit from. It defines the basic
interface for models, including parameter handling, evaluation, and algebraic
composition.

The ParamMeta class stores metadata for each parameter, including its index in
the full global theta vector, its local index within the node's parameter block,
and optional human-readable names.

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

import inspect
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class ParamMeta:
    """Metadata for a parameter.

    Parameter in the full global theta vector, which is the concatenation of all
    node parameters.
    """

    node: int  # counter for the node to ensure
    model_name: str  # name of the model node this parameter belongs to
    model_counter: int  # counter for the model node to ensure unique global names
    local_name: str  # optional human-readable name within the node
    local_index: int  # index within the node's local theta block
    index: int  # global index in theta_full
    global_name: str | None = None  # optional human-readable global name


@dataclass
class Layout:
    """Layout of the global parameter vector."""

    ndim: int
    params_meta: list[ParamMeta]
    name_to_index: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        """Complete global names for parameters."""
        _ = self.names

    @property
    def names(self) -> list[str]:
        """List of parameter names in the global theta vector."""
        out: list[str] = []
        self.name_to_index = {}  # build name_to_index mapping while generating names
        repeated_local_names = defaultdict(int)
        for param in self.params_meta:
            repeated_local_names[param.local_name] += 1

        repeated_models = {
            param.model_name for param in self.params_meta if param.model_counter > 0
        }

        for param in self.params_meta:
            local_name = param.local_name
            model_name = param.model_name
            if model_name in repeated_models:
                out.append(f"{model_name}{param.model_counter}.{local_name}")
            elif repeated_local_names[local_name] > 1:
                out.append(f"{model_name}.{local_name}")
            else:
                out.append(f"{local_name}")
            param.global_name = out[-1]
            self.name_to_index[param.global_name] = param.index
        return out

    def _available_names_mapped(self) -> tuple[set[str], dict[str, list[int]]]:
        """Set of model names in the layout and how they map to parameter indices.

        Returns
        -------
        set[str]
            A set of unique model names present in the layout, with counters
            appended to ensure uniqueness when the same model appears multiple
            times in the expression tree.
        dict[str, list[int]]
            A mapping from each unique model name to a list of global parameter
            indices corresponding to that model. This allows for easy retrieval
            of the parameters associated with each model in the layout.
            unique model name -> list of indices in theta_full of that model

        """
        counts = Counter(param.model_name for param in self.params_meta)
        available_names = set()
        map_model_to_index = defaultdict(list)

        for param in self.params_meta:
            effective_name = (
                f"{param.model_name}{param.model_counter}"
                if counts[param.model_name] > 1
                else param.model_name
            )
            available_names.add(effective_name)
            map_model_to_index[effective_name].append(param.index)
        return available_names, map_model_to_index

    def model_names(self) -> set[str]:
        """Set of unique model names in the layout, with counters for duplicates.

        Returns
        -------
        set[str]
            A set of unique model names present in the layout, with counters
            appended to ensure uniqueness when the same model appears multiple
            times in the expression tree. This allows for easy identification of
            the different models that are part of the overall model expression.

        """
        available_names, _ = self._available_names_mapped()
        return available_names

    def mask(self, model_name: str) -> np.ndarray:
        """Boolean mask for the parameters of a given model name.

        Parameters
        ----------
        model_name : str
            The name of the model for which to create the mask. This should be
            one of the unique model names returned by the `model_names` method,
            which may include counters if the same model appears multiple times
            in the expression tree.

        Returns
        -------
        np.ndarray
            A boolean array of length `ndim` where True indicates that the
            parameter at that index belongs to the specified model, and False
            otherwise. This allows for easy selection of the parameters
            associated with a particular model when working with the full
            parameter vector.

        """
        mask = np.zeros(self.ndim, dtype=bool)
        available_names, map_model_to_index = self._available_names_mapped()

        if model_name not in available_names:
            msg = f"Model name '{model_name}' not found in layout."
            msg += f" Available model names: {available_names}."
            logger.error(msg)
            raise ValueError(msg)

        for index in map_model_to_index[model_name]:
            mask[index] = True

        return mask

    def get_index(self, name: str) -> int:
        """Global index of a parameter by full name.

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
        node_counter = 0
        model_counter = 0
        for param in self.params_meta:
            node_counter = max(node_counter, param.node + 1)
            if param.model_name == node.name:
                model_counter = max(model_counter, param.model_counter + 1)

        for local_index, local_name in enumerate(node.param_names):
            self.params_meta.append(
                ParamMeta(
                    node=node_counter,
                    model_name=node.name,
                    model_counter=model_counter,
                    local_name=local_name,
                    local_index=local_index,
                    index=self.current_index + local_index,
                ),
            )

        self.current_index += len(node.param_names)

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

    def __init_subclass__(cls) -> None:
        """Check that subclasses implement the required interface."""
        super().__init_subclass__()

        # Only check if subclass overrides evaluate
        if "evaluate" in cls.__dict__:
            sig = inspect.signature(cls.evaluate)
            minimum_params = {"self", "theta", "x", "**kwargs"}
            if len(sig.parameters) < len(minimum_params):
                msg = f"{cls.__name__}.evaluate must have at least 4 parameters:"
                msg += " (self, theta, x, **kwargs)"
                logger.error(msg)
                raise TypeError(msg)

            kind_kwargs = inspect.Parameter.VAR_KEYWORD
            set_kinds = {p.kind for p in sig.parameters.values()}
            if kind_kwargs not in set_kinds:
                msg = f"{cls.__name__}.evaluate must accept **kwargs."
                logger.error(msg)
                raise TypeError(msg)
        else:
            msg = f"{cls.__name__} does not implement evaluate method."
            logger.error(msg)
            raise NotImplementedError(msg)

        if "n_params" not in cls.__dict__:
            msg = f"{cls.__name__} does not implement n_params property."
            logger.error(msg)
            raise NotImplementedError(msg)

    # To be implemented by subclasses --------------------------------------------------
    @property
    def n_params(self) -> int:
        """Number of parameters in the model."""
        msg = "Subclasses of BaseModel must implement n_params property."
        logger.exception(msg)
        raise NotImplementedError(msg)

    @property
    def param_names(self) -> list[str]:
        """Parameter names for the model.

        Should be overridden by subclasses to provide meaningful parameter
        names. By default, it returns generic names like p0, p1, ...,
        p{n_params-1}.
        """
        return [f"p{i}" for i in range(self.n_params)]

    def evaluate(
        self,
        theta: list | np.ndarray,  # noqa: ARG002
        x: float | list | np.ndarray,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> np.ndarray:
        """Evaluate the model or submodel at the given parameters `theta` and input `x`.

        Parameters
        ----------
        theta : list | np.ndarray
            The model parameters.
        x : float | list | np.ndarray
            The input values at which to evaluate the model.
        **kwargs : object
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

    @property
    def layout(self) -> Layout:
        """Build and return the Layout of the model."""
        b = _LayoutBuilder()
        _walk_layout(self, b)
        return b.build()

    @property
    def ndim(self) -> int:
        """Total number of parameters in the model."""
        return self.layout.ndim

    def evaluate_local(
        self,
        theta_local: list | np.ndarray,
        x: float | list | np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Evaluate the model at the given input `x` and local parameters `theta_local`.

        This method is used internally by the BinaryOp and UnaryOp classes to
        evaluate the model with the correct subset of parameters.

        Parameters
        ----------
        theta_local : list | np.ndarray
            The model parameters for the local submodel.
        x : float | list | np.ndarray
            The input values at which to evaluate the model.
        **kwargs : object
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
        return self.evaluate(theta_local, x, **kwargs)

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

    def __rpow__(self, other: float | np.number | BaseModel) -> BinaryOp:
        """Right power operator for model composition."""
        return BinaryOp(_wrap_scalar(other), self, np.power, "**")

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
    def value(self) -> float:
        """Get the fixed constant value of the model."""
        return self._value

    @property
    def n_params(self) -> int:
        """Number of parameters in the model (always 0 for FixedConstant)."""
        return 0

    def evaluate(
        self,
        theta: list | np.ndarray,  # noqa: ARG002
        x: float | list | np.ndarray,
        **kwargs: object,  # noqa: ARG002
    ) -> np.ndarray:
        """Evaluate constant function.

        Parameters
        ----------
        theta : list | np.ndarray
            The model parameters (not used for FixedConstant, but included for
            compatibility with the evaluate method signature).
        x : float | list | np.ndarray
            Array of input values at which to evaluate the model. The output
            will have the same shape as `x`, but the values will be constant
            regardless of `x`.
        **kwargs : object
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


class UnaryOp(BaseModel):
    """Handle unary operations of a given BaseModel."""

    def __init__(self, operand: BaseModel, op_func: np.ufunc, symbol: str) -> None:
        """Class that handles unary operations of a given BaseModel.

        Parameters
        ----------
        operand : BaseModel
            The model to which the unary operation will be applied.
        op_func : np.ufunc
            The numpy ufunc that defines the unary operation (e.g., np.exp, np.log,
            np.negative).
        symbol : str
            The symbol representing the unary operation (e.g., "exp", "log", "-"),
            used for naming the composed model.

        """
        self.operand = operand
        self.op_func = op_func
        self.symbol = symbol
        super().__init__(name=f"{symbol}({operand.name})")

    @property
    def n_params(self) -> int:
        """Number of parameters.

        Same as the `operand` model.

        Returns
        -------
        int
            The number of parameters in the unary operation, which is the same as
            the number of parameters in the `operand` model.

        """
        return self.operand.n_params

    @property
    def param_names(self) -> list[str]:
        """Parameter names.

        Same as the `operand` model.

        Returns
        -------
        list[str]
            The parameter names for the unary operation, which are the same as
            the parameter names of the `operand` model.

        """
        return self.operand.param_names

    def evaluate(
        self,
        theta: list | np.ndarray,
        x: float | list | np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Evaluate the function.

        Evaluate the function by applying the unary operation to the operand
        model.

        Parameters
        ----------
        theta : list | np.ndarray
            The model parameters.
        x : float | list | np.ndarray
            Array of input values at which to evaluate the model.
        **kwargs : object
            Additional keyword arguments that may be needed for evaluation.

        Returns
        -------
        np.ndarray
            Return an array of the same shape as `x` filled with the result of
            applying the unary operation to the operand model evaluated at `x`.

        """
        return self.op_func(self.operand.evaluate(theta, x, **kwargs))


class BinaryOp(BaseModel):
    """Handle binary operations of two given BaseModels.

    Two BaseModel instances are combined using a specified binary operation
    defined by a numpy ufunc, and the resulting model can be evaluated at any
    input `x` with the correct parameters. The class takes care of correctly
    partitioning the parameter vector for the left and right models when
    evaluating the composed model.

    A series of BinaryOp instances can be combined to create complex model
    expressions using standard operators like +, -, *, /, and **, allowing for
    flexible and intuitive model composition.
    """

    def __init__(
        self,
        left: BaseModel,
        right: BaseModel,
        op_func: np.ufunc,
        symbol: str,
    ) -> None:
        """Class that handles binary operations of two given BaseModels.

        Parameters
        ----------
        left : BaseModel
            The left model in the binary operation.
        right : BaseModel
            The right model in the binary operation.
        op_func : np.ufunc
            The numpy ufunc that defines the binary operation (e.g., np.add,
            np.subtract, np.multiply, np.divide, np.power).
        symbol : str
            The symbol representing the binary operation (e.g., "+", "-", "*",
            "/", "**"), used for naming the composed model.

        """
        self.left = left
        self.right = right
        self.op_func = op_func
        self.symbol = symbol
        super().__init__(name=f"({left.name} {symbol} {right.name})")

    @property
    def n_params(self) -> int:
        """Number of parameters.

        Returns
        -------
        int
            The number of parameters in the binary operation, which is the sum
            of the number of parameters in the `left` and `right` models.

        """
        return self.left.n_params + self.right.n_params

    @property
    def param_names(self) -> list[str]:
        """Global parameter names.

        Global parameter names as generated by the layout.

        Returns
        -------
        list[str]
            The global parameter names for the binary operation, which are the
            concatenation of the parameter names of the `left` and `right` models.
            The names take into account the structure of the model expression
            tree to ensure uniqueness and interpretability.

        """
        return self.layout.names

    def evaluate(
        self,
        theta: list | np.ndarray,
        x: float | list | np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Evaluate the function.

        Evaluate the function by evaluating the left and right models at the
        given input `x` and parameters `theta`, and then applying the binary
        operation to the results. The parameter vector `theta` is partitioned
        into two parts: the first part corresponds to the parameters of the
        `left` model and the second part corresponds to the parameters of the
        `right` model. The method ensures that the correct subset of parameters
        is passed to each model when evaluating them.

        Parameters
        ----------
        theta : list | np.ndarray
            The model parameters (both left and right models).
        x : float | list | np.ndarray
            Array of input values at which to evaluate the model.
        **kwargs : object
            Additional keyword arguments that may be needed for evaluation.

        Returns
        -------
        np.ndarray
            Return an array of the same shape as `x` filled with the result of
            applying the binary operation to the left and right models evaluated at `x`.

        """
        index_left = self.left.n_params
        theta_left = theta[:index_left]
        theta_right = theta[index_left:]

        return np.atleast_1d(
            self.op_func(
                self.left.evaluate_local(theta_left, x, **kwargs),
                self.right.evaluate_local(theta_right, x, **kwargs),
            ),
        )
