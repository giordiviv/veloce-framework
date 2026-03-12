"""Implementation of the least squares fitting.

Based on scipy.optimize.least_squares:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

"""

import logging
from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from velocefw.fit.objective import ModelObjectiveFunction
from velocefw.fit.results import FailedFitResult, SuccessfulFitResult
from velocefw.model import BaseModel, CompiledModel, compile_model

logger = logging.getLogger(__name__)


def fit_least_squares(
    compiled_model: BaseModel | CompiledModel,
    x: list | np.ndarray,
    y: list | np.ndarray,
    yerr: list | np.ndarray | None = None,
    theta_free_init: list | np.ndarray | None = None,
    **kwargs,  # noqa: ANN003
) -> SuccessfulFitResult | FailedFitResult:
    """Fit the model to the data using least squares optimization.

    Based on scipy.optimize.least_squares:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html


    Parameters
    ----------
    compiled_model : BaseModel or CompiledModel
        The model to be fitted. If a BaseModel is provided, it will be compiled.
    x : list or np.ndarray
        The input values at which to evaluate the model.
    y : list or np.ndarray
        The observed values corresponding to the input values.
    yerr : list or np.ndarray, optional
        The uncertainties associated with the observed values. If not provided,
        it is assumed that all observations have equal uncertainty.
    theta_free_init : list or np.ndarray, optional
        Initial guess for the free parameters of the model. If not provided, it
        is assumed to be a vector of zeros.
    **kwargs : object
        Additional keyword arguments that will be passed to
        scipy.optimize.least_squares.

    Returns
    -------
    FitResult
        A dataclass containing the results of the fit, including the best-fit
        parameters, the fit statistics, and the optimization result.

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = None if yerr is None else np.asarray(yerr, dtype=float)
    if isinstance(compiled_model, BaseModel):
        compiled_model = compile_model(compiled_model)

    theta0 = np.asarray(theta_free_init, dtype=float).ravel()
    if theta_free_init is None:
        theta0 = np.zeros(compiled_model.n_params_free)

    model_evaluator = ModelObjectiveFunction(
        model=compiled_model,
        x=x,
        y=y,
        yerr=yerr,
    )

    optimization_result: OptimizeResult = least_squares(
        model_evaluator.weighted_residuals,
        theta0,
        **kwargs,
    )
    optimization_result = cast("OptimizeResult", optimization_result)

    # If the optimization was not successful, return a FailedFitResult with the
    # error message
    if not optimization_result.success:
        return FailedFitResult(
            method="least_squares",
            success=optimization_result.success,
            message=optimization_result.message,
            x=x,
            y=y,
            yerr=yerr,
            optimizer_result=optimization_result,
        )

    # If the optimization was successful, compute the fit statistics and error estimates
    fit_statistics = model_evaluator.statistics(optimization_result.x)

    # Extra: error estimates from the covariance matrix if available
    # For covariance matrix, see:
    # https://stackoverflow.com/questions/42388139/how-to-compute-standard-deviation-errors-with-scipy-optimize-least-squares
    covariance_matrix = np.linalg.inv(
        optimization_result.jac.T @ optimization_result.jac,
    )
    theta_err = np.sqrt(np.diag(covariance_matrix))

    # Extra: store the covariance matrix and parameter errors in the FitResult.extra
    extra = {"theta_err": theta_err, "covariance_matrix": covariance_matrix}

    return SuccessfulFitResult(
        method="least_squares",
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
