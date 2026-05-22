"""Prior construction: connecting densities to model parameters.

Two classes are provided:

    ParameterPrior   — wraps any density object and maps its variables to
                       indices in theta_free. The result is a callable
                       (theta_free -> float) suitable for ModelObjectiveFunction.

    CombinedLogPrior — sums independent log-prior callables. Accepts any
                       mixture of ParameterPrior instances and plain functions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


class ParameterPrior:
    """Connect a density object to specific parameters in theta_free.

    Parameters
    ----------
    density :
        Any density object with ``.var_names`` and ``.log_pdf(dict) -> float``.
        Typically a ``KDEDensity``, ``ConditionalDensity``, or ``SplineDensity``.
    var_to_index :
        Mapping from each variable name in the density to its index in
        ``theta_free``. Must cover **all** variables in ``density.var_names``.

    Examples
    --------
    Single-variable prior (pc1 conditioned on log_P):
    >>> cond = kde_logP_pc1.conditional(log_P=np.log10(period))
    >>> fast = SplineDensity(cond)
    >>> prior = ParameterPrior(fast, {"pc1": 1})

    Joint two-variable prior (pc1 and pc2 jointly):
    >>> prior = ParameterPrior(kde_pc12, {"pc1": 1, "pc2": 2})

    """

    def __init__(
        self,
        density: object,
        var_to_index: dict[str, int],
    ) -> None:
        """Initialize the ParameterPrior from a object describing a density.

        Parameters
        ----------
        density : object
            Any density object with ``.var_names`` and ``.log_pdf(dict) -> float``.
            Typically a ``KDEDensity``, ``ConditionalDensity``, or ``SplineDensity``.
        var_to_index : dict[str, int]
            Mapping from each variable name in the density to its index in
            ``theta_free``. Must cover **all** variables in ``density.var_names``.

        Raises
        ------
        ValueError
            If any variable in ``density.var_names`` is absent from ``var_to_index``.

        """
        missing = [v for v in density.var_names if v not in var_to_index]
        if missing:
            msg = f"var_to_index is missing entries for: {missing}. "
            msg += f"All of {density.var_names} must map to a theta_free index."
            logger.error(msg)
            raise ValueError(msg)

        self._density = density
        # Filter additional/extra unused var names that maybe be present
        self._var_to_index = {v: var_to_index[v] for v in density.var_names}

    def __call__(self, theta_free: np.ndarray) -> float:
        """Evaluate the log-prior for the given parameter vector."""
        values = {v: float(theta_free[idx]) for v, idx in self._var_to_index.items()}
        return self._density.log_pdf(values)

    @staticmethod
    def index_from_names(
        density: object,
        param_names: list[str],
    ) -> dict[str, int]:
        """Build ``var_to_index`` automatically from a parameter name list.

        Parameters
        ----------
        density :
            Density object whose ``.var_names`` should be mapped.
        param_names :
            Parameter names, e.g. ``compiled_model.param_names_free``.

        Returns
        -------
        dict
            Mapping from density variable name to its position in
            ``param_names``.

        Raises
        ------
        ValueError
            If any density variable is absent from ``param_names``.

        """
        result: dict[str, int] = {}
        missing: list[str] = []
        for var in density.var_names:  # type: ignore[attr-defined]
            try:
                result[var] = param_names.index(var)
            except ValueError:
                missing.append(var)
        if missing:
            msg = f"Variables {missing} not found in param_names. "
            msg += f"Available: {param_names}"
            logger.error(msg)
            raise ValueError(msg)
        return result


class CombinedLogPrior:
    """Sum of independent log-prior callables.

    Treats each component as contributing an independent log-probability term.
    Short-circuits to ``-inf`` as soon as any component returns a non-finite
    value, avoiding unnecessary evaluations.

    Accepts any mixture of ``ParameterPrior`` instances and plain callables
    that take ``theta_free: np.ndarray`` and return a ``float``.

    Parameters
    ----------
    log_priors :
        Sequence of callables (theta_free -> float).

    Examples
    --------
    >>> log_prior = CombinedLogPrior([prior_pc1, prior_pc2, prior_phase])
    >>> obj = ModelObjectiveFunction(model, x, y, yerr, log_prior=log_prior)

    """

    def __init__(
        self,
        log_priors: Sequence[Callable[[np.ndarray], float]],
    ) -> None:
        """Sequence of log priors functions on theta_free."""
        self._log_priors = list(log_priors)

    def __call__(self, theta_free: np.ndarray) -> float:
        """Evaluate the log-prior for the given parameter vector."""
        lp = 0.0
        for prior in self._log_priors:
            lp += prior(theta_free)
            if not np.isfinite(lp):
                return -np.inf
        return lp
