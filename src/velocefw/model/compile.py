"""Compile the model for faster evaluation and to specify parametrization.

The `compile_model` function compiles a given model into a `CompiledModel` that
can be evaluated more efficiently. The compilation process involves recursively
traversing the model's expression tree and creating a single callable function
that evaluates the entire model. This avoids the overhead of walking the tree
and allocating slices every time the model is evaluated, which can be
particularly beneficial when fitting with MCMC methods. The compiled model also
supports a `Parametrization` that defines how the free parameters map to the
full global theta vector, allowing for tied parameters and other constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .base import BinaryOp, UnaryOp
from .parametrization import Parametrization

if TYPE_CHECKING:
    from .base import BaseModel, Layout

logger = logging.getLogger(__name__)


class EvaluateCallable(Protocol):
    """Protocol for the callable returned by the compiled model."""

    def __call__(
        self,
        theta_full: np.ndarray,
        x: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Callable for evaluating the compiled model.

        Parameters
        ----------
        theta_full : np.ndarray
            The full global theta vector, which is the concatenation of all node
            parameters.
        x : np.ndarray
            The input values at which to evaluate the model.
        **kwargs : object
            Additional keyword arguments that may be needed for evaluation.

        Returns
        -------
        np.ndarray
            Return an array of model predictions for the corresponding inputs.

        """
        ...


# Faster evaluation with parametrization -------------------------------------------
def compile_model(
    model: BaseModel,
    parametrization: Parametrization | None = None,
) -> CompiledModel:
    """Compile the model for faster evaluation.

    It returns a CompiledModel that avoids solving the possible model tree
    everytime the model is evaluated. A Parametrization can be provided to
    specify how the free parameters map to the full global theta vector,
    allowing for tied parameters and other constraints. If no
    parametrization is provided to the compile method, a parametrization
    based on the model's layout will be used by default (identity mapping
    with no shared parameters).

    Parameters
    ----------
    model : BaseModel
        The model to compile.
    parametrization : Parametrization | None
        The parametrization to use for compilation. If None, the default
        parametrization is used.

    Returns
    -------
    CompiledModel
        The compiled model.

    """
    layout = model.layout()
    parametrization = parametrization or Parametrization.identity(layout.ndim)

    # Check that the number of parameters used in the compiled function
    # matches the expected number from the layout
    if parametrization.full_ndim != layout.ndim:
        msg = f"Expected {layout.ndim} parameters in parametrization, "
        msg += f"got {parametrization.full_ndim}"
        logger.error(msg)
        raise ValueError(msg)

    def compile_fn(
        node: BaseModel,
        counter_params: int,
    ) -> tuple[EvaluateCallable, int]:
        """Recursively compile the model expression tree into a single function."""
        if isinstance(node, UnaryOp):
            operand_fn, counter_params = compile_fn(node.operand, counter_params)
            op = node.op_func

            def fn(
                theta_full: np.ndarray,
                x: np.ndarray,
                **kwargs: object,
            ) -> np.ndarray:
                return op(operand_fn(theta_full, x, **kwargs))

            return (fn, counter_params)
        if isinstance(node, BinaryOp):
            left_fn, counter_params = compile_fn(node.left, counter_params)
            right_fn, counter_params = compile_fn(node.right, counter_params)
            op = node.op_func

            def fn(
                theta_full: np.ndarray,
                x: np.ndarray,
                **kwargs: object,
            ) -> np.ndarray:
                return op(
                    left_fn(theta_full, x, **kwargs),
                    right_fn(theta_full, x, **kwargs),
                )

            return fn, counter_params

        # leaf / primitive or user-defined model: allocate its local block
        local_ndim = node.n_params
        local_indices = slice(counter_params, counter_params + local_ndim)

        def fn(
            theta_full: np.ndarray,
            x: np.ndarray,
            **kwargs: object,
        ) -> np.ndarray:
            theta_local = theta_full[local_indices]
            # Call node.evaluate, not node.evaluate_local, to avoid double
            # checks in hot loops
            return node.evaluate(theta_local, x, **kwargs)

        return fn, counter_params + local_ndim

    fn_full, counter_params = compile_fn(model, 0)

    return CompiledModel(model, layout, parametrization, fn_full)


@dataclass
class CompiledModel:
    """Compiled model ready for evaluation.

    Allow to produce a faster callable to evaluate the model at given inputs and
    parameters, by compiling the model expression tree into a single function.
    This can be particularly useful when fitting with MCMC methods. A compiled
    model avoids re-walking the tree and re-allocating slices every call.
    """

    model: BaseModel
    layout: Layout
    parametrization: Parametrization
    fn_full: EvaluateCallable
    # x_fixed
    # fn_full
    # kwargs_fixed

    @property
    def n_params_free(self) -> int:
        """Number of free parameters in the model."""
        return self.parametrization.free_ndim

    @property
    def n_params_full(self) -> int:
        """Number of parameters in the full global theta vector."""
        return self.parametrization.full_ndim

    def __call__(
        self,
        theta_free: list | np.ndarray,
        *,
        x: float | list | np.ndarray | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        """Evaluate the compiled model at the given free parameters and input `x`.

        Parameters
        ----------
        theta_free : list | np.ndarray
            The free parameters of the model, which will be expanded to the full
            global theta vector using the parametrization.
        x : np.ndarray | None, optional
            The input values at which to evaluate the model. If None, it will be
            determined based on the model's requirements (e.g., from kwargs or
            default).
        **kwargs : object
            Additional keyword arguments that may be needed for evaluation.

        Returns
        -------
        np.ndarray
            Return an array of model predictions for the corresponding inputs.

        """
        # Handle theta
        theta_free = np.asarray(theta_free, float).ravel()
        theta_full = self.parametrization.expand(theta_free)
        # Possibly add logic to have a default x if not provided.
        if x is None:
            msg = "Input x must be provided when evaluating the compiled model."
            logger.error(msg)
            raise ValueError(msg)
        x_use = np.asarray(x, float)
        return self.fn_full(theta_full, x_use, **kwargs)
