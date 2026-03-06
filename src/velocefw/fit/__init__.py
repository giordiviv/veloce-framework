"""Fitting module for the Veloce Framework."""

from velocefw.fit.leastsq import fit_least_squares
from velocefw.fit.objective import ModelEvaluator
from velocefw.fit.results import FitResult, FitStatistics

__all__ = ["FitResult", "FitStatistics", "ModelEvaluator", "fit_least_squares"]
