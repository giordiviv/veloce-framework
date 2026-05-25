"""Fitting module for the Veloce Framework."""

from velocefw.fit.leastsq import fit_least_squares
from velocefw.fit.map import fit_map
from velocefw.fit.objective import ModelObjectiveFunction
from velocefw.fit.results import (
    FailedFitError,
    FailedFitResult,
    FitStatistics,
    SuccessfulFitResult,
)

__all__ = [
    "FailedFitError",
    "FailedFitResult",
    "FitStatistics",
    "ModelObjectiveFunction",
    "SuccessfulFitResult",
    "fit_least_squares",
    "fit_map",
]
