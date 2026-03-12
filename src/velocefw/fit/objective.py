"""Class that define the objective function to be minimized during the fitting process.

The class is called `ModelObjectiveFunction`, to increase clarity in the context of the
framework, where `Objective` could be difficult to understand. The
`ModelObjectiveFunction` class is responsible for computing the objective function to be
minimized during the fitting process.

Objective functions available:
- Residual sum of squares (RSS)
- Chi-squared statistic (chi2)
- Root mean square error (RMSE)
- Log-likelihood (gaussian likelihood)
- Log-posterior (log-prior + log-likelihood)
"""

import logging
from collections.abc import Callable

import numpy as np

from velocefw.fit.results import FitStatistics
from velocefw.model import BaseModel, CompiledModel, compile_model

logger = logging.getLogger(__name__)


class ModelObjectiveFunction:
    """Class that calculates the objective functions for fitters."""

    def __init__(
        self,
        model: BaseModel | CompiledModel,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray | None = None,
        log_prior: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        """Initialize the ModelObjectiveFunction.

        Parameters
        ----------
        model : BaseModel or CompiledModel
            The model to be evaluated.
        x : np.ndarray
            The input values at which to evaluate the model.
        y : np.ndarray
            The observed values corresponding to the input values.
        yerr : np.ndarray, optional
            The uncertainties associated with the observed values. If not provided,
            it is assumed that all observations have equal uncertainty.
        log_prior: Callable[[np.ndarray], float], optional
            A function that computes the log prior probability of the model parameters.
            If not provided, it is assumed that the prior is uniform over the
            parameter space.

        """
        if isinstance(model, BaseModel):
            model = compile_model(model)
        self.model = model
        self.x = x
        self.y = y
        self.yerr = yerr
        self.log_prior = log_prior

    def model_y(self, theta: list | np.ndarray) -> np.ndarray:
        """Evaluate the model at the given parameters.

        Parameters
        ----------
        theta : list or np.ndarray
            The parameters at which to evaluate the model.

        Returns
        -------
        np.ndarray
            The model predictions corresponding to the input parameters.

        """
        return self.model(theta, x=self.x)

    def theta_full(self, theta_free: list | np.ndarray) -> np.ndarray:
        """Full theta vector corresponding to the given free parameters.

        Parameters
        ----------
        theta_free : list or np.ndarray
            The free parameters for which to compute the full theta vector.

        Returns
        -------
        np.ndarray
            The full theta vector corresponding to the given free parameters.

        """
        return self.model.parametrization.expand(np.asarray(theta_free))

    def residuals(self, theta_free: np.ndarray) -> np.ndarray:
        """Compute the residuals: y_obs - y_model."""
        return self.y - self.model_y(theta_free)

    def weighted_residuals(self, theta_free: np.ndarray) -> np.ndarray:
        """Compute the weighted residuals."""
        r = self.residuals(theta_free)
        if self.yerr is None:
            return r
        return r / self.yerr

    def rss(self, theta_free: np.ndarray) -> float:
        """Compute the residual sum of squares (RSS)."""
        r = self.residuals(theta_free)
        return float(np.sum(r**2))

    def chi2(self, theta_free: np.ndarray) -> float:
        """Compute the chi-squared statistic."""
        r = self.weighted_residuals(theta_free)
        return float(np.sum(r**2))

    def reduced_chi2(self, theta_free: np.ndarray) -> float:
        """Compute the reduced chi-squared statistic."""
        dof = self.y.size - len(np.asarray(theta_free, dtype=float).ravel())
        if dof > 0:
            return self.chi2(theta_free) / dof
        return float("inf")

    def rmse(self, theta_free: np.ndarray) -> float:
        """Compute the root mean square error (RMSE)."""
        r = self.residuals(theta_free)
        return float(np.sqrt(np.mean(r**2)))

    def loglikelihood(self, theta_free: np.ndarray) -> float:
        """Compute the log-likelihood (gaussian likelihood)."""
        if self.yerr is None:
            return -0.5 * self.rss(theta_free)
        r = self.weighted_residuals(theta_free)
        norm = np.sum(np.log(2.0 * np.pi * self.yerr**2))
        return float(-0.5 * (np.sum(r**2) + norm))

    def logposterior(self, theta_free: np.ndarray) -> float:
        """Compute the log-posterior (log-prior + log-likelihood)."""
        lp = 0.0 if self.log_prior is None else float(self.log_prior(theta_free))
        if not np.isfinite(lp):
            return -np.inf
        ll = self.loglikelihood(theta_free)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def statistics(self, theta_free: np.ndarray) -> FitStatistics:
        """Compute all fit statistics for the given parameters.

        Parameters
        ----------
        theta_free : np.ndarray
            The free parameters for which to compute the fit statistics.

        Returns
        -------
        FitStatistics
            A dataclass containing all computed fit statistics for the given parameters.

        """
        residuals = self.residuals(theta_free)
        rss = self.rss(theta_free)
        rmse = self.rmse(theta_free)
        chi2 = self.chi2(theta_free)
        dof = self.y.size - len(np.asarray(theta_free, dtype=float).ravel())
        reduced_chi2 = self.reduced_chi2(theta_free)
        return FitStatistics(
            residuals=residuals,
            rss=rss,
            rmse=rmse,
            chi2=chi2,
            reduced_chi2=reduced_chi2,
            dof=dof,
        )
