"""Fitting module for the Veloce Framework."""

from velocefw.fit.leastsq import fit_least_squares
from velocefw.fit.objective import ModelObjectiveFunction
from velocefw.fit.results import FailedFitResult, FitStatistics, SuccessfulFitResult

__all__ = [
    "FailedFitResult",
    "FitStatistics",
    "ModelObjectiveFunction",
    "SuccessfulFitResult",
    "fit_least_squares",
]
