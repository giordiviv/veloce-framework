"""Implementation of the Maximum A Posteriori estimation.

Based on scipy.optimize.minimize
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

"""

import logging
from collections.abc import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from velocefw.fit.objective import ModelObjectiveFunction
from velocefw.fit.results import FailedFitResult, SuccessfulFitResult
from velocefw.model import BaseModel, CompiledModel, compile_model

logger = logging.getLogger(__name__)


def fit_map(  # noqa: PLR0913
    compiled_model: BaseModel | CompiledModel,
    x: list | np.ndarray,
    y: list | np.ndarray,
    yerr: list | np.ndarray | None = None,
    theta_free_init: list | np.ndarray | None = None,
    logprior: Callable[[np.ndarray], float] | None = None,
    **kwargs,  # noqa: ANN003
) -> SuccessfulFitResult | FailedFitResult:
    """Estimate Maximum A Posteriori for the model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float) if yerr is not None else None
    if isinstance(compiled_model, BaseModel):
        compiled_model = compile_model(compiled_model)
    if logprior is None:
        msg = "No log_prior was given. Fit is equal to Maximum Likelihood Estimation"
        logger.info(msg)

    if theta_free_init is None:
        theta0 = np.zeros(compiled_model.n_params_free)
    else:
        theta0 = np.asarray(theta_free_init, dtype=float).ravel()

    model_evaluator = ModelObjectiveFunction(
        model=compiled_model,
        x=x,
        y=y,
        yerr=yerr,
        logprior=logprior,
    )

    optimization_result: OptimizeResult = minimize(
        lambda x: -model_evaluator.logposterior(x),
        theta0,
        **kwargs,
    )

    # If the optimization was not successful, return a FailedFitResult
    if not optimization_result.success:
        return FailedFitResult(
            method="map",
            model=compiled_model,
            success=optimization_result.success,
            message=optimization_result.message,
            x=x,
            y=y,
            yerr=yerr,
            optimizer_result=optimization_result,
        )

    # If the optimization was successful, compute the fit statistics
    fit_statistics = model_evaluator.statistics(optimization_result.x)
    logprior_value = model_evaluator.logprior(optimization_result.x)
    loglikelihood_value = model_evaluator.loglikelihood(optimization_result.x)
    logposterior_value = model_evaluator.logposterior(optimization_result.x)

    extra = {
        "logprior": logprior_value,
        "loglikelihood": loglikelihood_value,
        "logposterior": logposterior_value,
    }

    return SuccessfulFitResult(
        method="least_squares",
        model=compiled_model,
        optimizer_result=optimization_result,
        success=optimization_result.success,
        message=optimization_result.message,
        theta_free=optimization_result.x,
        theta_full=compiled_model.parametrization.expand(optimization_result.x),
        y_model=model_evaluator.model_y(optimization_result.x),
        x=x,
        y=y,
        yerr=yerr,
        stats=fit_statistics,
        extra=extra,
    )
