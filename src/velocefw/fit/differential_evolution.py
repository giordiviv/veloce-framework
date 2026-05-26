"""Implementation of the Differential Evolution (DE) algorithm.

Used to find the global minimum of a multivariate function.


Based on scipy.optimize.differential_evolution
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

"""

import logging
from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution

from velocefw.fit.objective import ModelObjectiveFunction
from velocefw.fit.results import FailedFitResult, SuccessfulFitResult
from velocefw.model import BaseModel, CompiledModel, compile_model

logger = logging.getLogger(__name__)


def differential_evolution_posterior(  # noqa: PLR0913
    compiled_model: BaseModel | CompiledModel,
    bounds: Sequence | list | np.ndarray,
    x: list | np.ndarray,
    y: list | np.ndarray,
    yerr: list | np.ndarray | None = None,
    logprior: Callable[[np.ndarray], float] | None = None,
    **kwargs,  # noqa: ANN003
) -> SuccessfulFitResult | FailedFitResult:
    """Global minimum via DE."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float) if yerr is not None else None
    if isinstance(compiled_model, BaseModel):
        compiled_model = compile_model(compiled_model)
    if logprior is None:
        msg = "No log_prior was given. "
        msg += "Fit is equal to Maximum Likelihood Estimation via DE."
        logger.info(msg)

    model_evaluator = ModelObjectiveFunction(
        model=compiled_model,
        x=x,
        y=y,
        yerr=yerr,
        logprior=logprior,
    )

    kwargs_de = {
        "strategy": "rand1bin",
        "maxiter": 5000,
        "popsize": 15,
        "tol": 1e-3,
    }
    kwargs_de.update(kwargs)

    optimization_result: OptimizeResult = differential_evolution(
        lambda x: -model_evaluator.logposterior(x),
        bounds,
        **kwargs_de,
    )

    # If the optimization was not successful, return a FailedFitResult
    if not optimization_result.success:
        return FailedFitResult(
            method="de_posterior",
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
        "bounds": bounds,
        "kwargs_de": kwargs_de,
    }

    return SuccessfulFitResult(
        method="de_posterio",
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
