"""Module that handles the fitting results."""

import logging
from dataclasses import dataclass

import numpy as np

from velocefw.model import BaseModel, CompiledModel, compile_model

logger = logging.getLogger(__name__)


@dataclass
class FitStatistics:
    """Class to store the statistics of a fit."""

    residuals: np.ndarray
    rss: float
    rmse: float
    chi2: float | None = None
    reduced_chi2: float | None = None
    dof: int | None = None


@dataclass
class FitResult:
    """General class to store the results of a fit."""

    method: str
    optimizer_result: object
    success: bool
    message: str
    theta_free: np.ndarray
    theta_full: np.ndarray
    y_model: np.ndarray
    stats: FitStatistics
    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None = None
    extra: dict | None = None

    @property
    def rmse(self) -> float:
        """Root mean square error (RMSE) of the fit."""
        return self.stats.rmse

    @property
    def chi2(self) -> float | None:
        """Chi-squared statistic of the fit."""
        return self.stats.chi2

    @property
    def reduced_chi2(self) -> float | None:
        """Reduced chi-squared statistic of the fit."""
        return self.stats.reduced_chi2

    def evaluate(
        self,
        compiled_model: BaseModel | CompiledModel,
        x: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Evaluate the model at the given parameters."""
        if isinstance(compiled_model, BaseModel):
            compiled_model = compile_model(compiled_model)
        return compiled_model(self.theta_free, x=x, **kwargs)
