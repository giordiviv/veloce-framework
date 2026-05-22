"""Module that handles the fitting results."""

import logging
from dataclasses import dataclass, field

import numpy as np

from velocefw.model import CompiledModel

logger = logging.getLogger(__name__)


class FailedFitError(Exception):
    """Exception raised when a fit fails."""

    def __init__(self, message: str) -> None:
        """Initialize the FailedFitError with a message."""
        self.msg = message

    def __str__(self) -> str:
        """Return the error message."""
        return f"Failed fit: {self.msg}"


@dataclass
class FitStatistics:
    """Class to store the statistics of a fit."""

    residuals: np.ndarray
    rss: float
    rmse: float
    chi2: float
    reduced_chi2: float
    dof: int


@dataclass(slots=True)
class _FitResultBase:
    """Common information returned by any fitter.

    Should not be used directly, but rather through the SuccessfulFitResult and
    FailedFitResult subclasses.

    """

    method: str
    model: CompiledModel
    optimizer_result: object
    success: bool
    message: str
    x: np.ndarray
    y: np.ndarray


@dataclass(slots=True)
class FailedFitResult(_FitResultBase):
    """Result of a failed fit."""

    yerr: np.ndarray | None = None
    extra: dict | None = None

    def __post_init__(self) -> None:
        """Ensure that the fit result is marked as failed."""
        if self.success:
            msg = "FailedFitResult must have success=False."
            logger.error(msg)
            raise ValueError(msg)


@dataclass(slots=True)
class SuccessfulFitResult(_FitResultBase):
    """Result of a successful fit."""

    theta_free: np.ndarray
    theta_full: np.ndarray
    y_model: np.ndarray
    stats: FitStatistics
    yerr: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure that the fit result is marked as successful."""
        if not self.success:
            msg = "SuccessfulFitResult must have success=True."
            logger.error(msg)
            raise ValueError(msg)

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
        x: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Evaluate the model at the given parameters."""
        return self.model(self.theta_free, x=x, **kwargs)
