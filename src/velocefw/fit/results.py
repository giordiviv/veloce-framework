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


@dataclass(slots=True)
class _FitResultBase:
    """Common information returned by any fitter.

    Should not be used directly, but rather through the SuccessfulFitResult and
    FailedFitResult subclasses.

    """

    method: str
    optimizer_result: object
    success: bool
    message: str
    x: np.ndarray
    y: np.ndarray

    def evaluate(
        self,
        _compiled_model: BaseModel | CompiledModel,
        _x: np.ndarray,
        **_kwargs,  # noqa: ANN003
    ) -> np.ndarray:
        """Evaluate the fitted model on new x values.

        Raises
        ------
        RuntimeError
            If the fit did not converge successfully.

        """
        msg = (
            "This fit result does not contain fitted parameters. "
            f"The optimization was not successful: {self.message}."
        )
        logger.error(msg)
        raise RuntimeError(msg)


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
    extra: dict | None = None

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
        compiled_model: BaseModel | CompiledModel,
        x: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Evaluate the model at the given parameters."""
        if isinstance(compiled_model, BaseModel):
            compiled_model = compile_model(compiled_model)
        return compiled_model(self.theta_free, x=x, **kwargs)
