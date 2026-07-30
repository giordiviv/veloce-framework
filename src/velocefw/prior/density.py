"""Density estimation objects.

Three types are provided, sharing a common interface
(var_names, domain, log_pdf):

    KDEDensity            — fits a kernel density estimate to named variables.
    KDEConditionalDensity — a KDEDensity with some variables fixed (lazy slice).
    SplineDensity         — fast 1-D pre-computed spline, suitable for MCMC.

Typical workflow::

    kde = KDEDensity(["log_P", "pc1"]).fit(train_df)
    cond = kde.conditional(log_P=1.2)      # KDEConditionalDensity over pc1
    fast = SplineDensity(cond)             # pre-compute spline (MCMC speed)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp
import statsmodels.api as sm

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _padded_domain(values: np.ndarray, padding: float) -> tuple[float, float]:
    """Return array extremes after padding them by a percentage.

    Parameters
    ----------
    values : np.ndarray
        Array whose extremes will be padded.
    padding : float
        Percentage of differnce of the extremes that will be padded on both sides.

    Returns
    -------
    tuple[float, float]
        Padded extremes.

    """
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = padding * (hi - lo)
    return lo - pad, hi + pad


class KDEDensity:
    """N-dimensional kernel density estimate over a named set of variables."""

    def __init__(
        self,
        var_names: list[str],
        bw: str | np.ndarray | list | float = "normal_reference",
        padding: float = 0.1,
        var_type: str | None = None,
    ) -> None:
        """N-dimensional kernel density estimate over a named set of variables.

        Parameters
        ----------
        var_names : list[str]
            Names of the variables, in the order they appear in the training data.
        bw : str or array-like, optional
            Bandwidth selection method passed to
            ``statsmodels.nonparametric.KDEMultivariate``.
            ``"normal_reference"`` is fast; ``"cv_ml"`` is cross-validated.
        padding : float, optional
            Padding add to the domain of the variables to avoid cutting the
            distribution right at the extreme values.
            Specified as a percentage of the difference between extremes.
            Default: 0.1.
        var_type: str or None, optional
            The type of variables. The string should contain a type specifier
            for each variable.
            Use `c` for continuous, `u` for unordered, `o` for ordered.
            Default: assumes all variables as continuous.

        """
        self.var_names: list[str] = list(var_names)
        if isinstance(bw, list | float | int):
            bw = np.array(bw)
        self.bw = bw
        self.domain: dict[str, tuple[float, float]] = {}
        self.padding = padding
        if var_type is None:
            var_type = "c" * len(self.var_names)
        self.var_type = var_type
        self._kde = None  # statsmodels KDEMultivariate, set by fit()

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has been called."""
        return self._kde is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            msg = "KDEDensity is not fitted — call .fit(data) first."
            logger.error(msg)
            raise RuntimeError(msg)

    def fit(self, data: pd.DataFrame) -> KDEDensity:
        """Fit the KDE to data and return self (for chaining).

        Rows containing NaN in any of the KDE variables are dropped.

        Parameters
        ----------
        data :
            DataFrame with columns for every variable in ``var_names``.

        """
        missing = [v for v in self.var_names if v not in data.columns]
        if missing:
            msg = f"Columns not found in data: {missing}"
            logger.error(msg)
            raise ValueError(msg)

        arr = data[self.var_names].to_numpy(dtype=float)

        # Find and remove rows containing NaN
        valid = ~np.any(np.isnan(arr), axis=1)
        if not np.any(valid):
            msg = "No valid (non-NaN) rows after filtering."
            logger.error(msg)
            raise ValueError(msg)
        arr = arr[valid]

        # Set domain of each variable
        self.domain = {
            v: _padded_domain(arr[:, i], self.padding)
            for i, v in enumerate(self.var_names)
        }

        # Apply KDE
        self._kde = sm.nonparametric.KDEMultivariate(
            arr,
            var_type=self.var_type,
            bw=self.bw,
        )
        logger.debug("Fitted %dD KDE for %s.", len(self.var_names), self.var_names)
        return self

    def log_pdf(self, values: dict[str, float]) -> float:
        """Evaluate the log density at a single point.

        Parameters
        ----------
        values :
            Mapping from variable name to value. Must cover all ``var_names``.

        """
        self._require_fitted()
        for v, (lo, hi) in self.domain.items():
            if not (lo <= float(values[v]) <= hi):
                return -np.inf
        row = np.array([values[v] for v in self.var_names])
        pdf_val = float(np.atleast_1d(self._kde.pdf(row))[0])
        return float(np.log(max(pdf_val, 1e-300)))

    def conditional(self, **fixed_values: float) -> KDEConditionalDensity:
        """Fix some variables and return a lower-dimensional conditional density.

        Parameters
        ----------
        **fixed_values :
            Variable name -> value for every variable to be conditioned on.
            At least one variable must remain free.

        Returns
        -------
        ConditionalDensity
            Density over the remaining (free) variables.

        Examples
        --------
        >>> kde = KDEDensity(["log_P", "pc1"]).fit(data)
        >>> cond = kde.conditional(log_P=1.2)   # density over pc1 only

        """
        self._require_fitted()
        invalid = [v for v in fixed_values if v not in self.var_names]
        if invalid:
            msg = f"Variables not in this density: {invalid} (vars: {self.var_names})"
            logger.error(msg)
            raise ValueError(msg)
        free = [v for v in self.var_names if v not in fixed_values]
        if not free:
            msg = "All variables are fixed — no free variables remain."
            logger.error(msg)
            raise ValueError(msg)
        # Check that all values provided are within the domain
        for name, value in fixed_values.items():
            low, high = self.domain[name]
            if not (low <= value <= high):
                msg = f"Fixed value for {name} is outside the domain: ({low}, {high})"
                logger.error(msg)
                raise ValueError(msg)

        return KDEConditionalDensity(
            parent=self,
            fixed=dict(fixed_values),
            var_names=free,
        )

    def _pdf_batch(self, rows: np.ndarray) -> np.ndarray:
        """Evaluate PDF on a 2-D array of rows (n_rows, n_vars). Internal use."""
        self._require_fitted()
        return np.asarray(self._kde.pdf(rows), dtype=float)


class KDEConditionalDensity:
    """A KDE density with some variables fixed at specific values.

    Produced by ``KDEDensity.conditional(**fixed)``. Shares the same
    ``log_pdf`` interface as ``KDEDensity`` but only accepts the free
    (unfixed) variables as input.

    Note: ``log_pdf`` returns the *joint* log density of the parent KDE,
    which differs from the true conditional by a constant (the marginal
    density of the fixed variables). This constant is irrelevant for MCMC
    and MAP because it cancels in all ratio computations.

    """

    def __init__(
        self,
        parent: KDEDensity,
        fixed: dict[str, float],
        var_names: list[str],
    ) -> None:
        """Marginal density of the KDE density at fixed values of a subset of variables.

        Use the function KDE.conditional to initialize this object.

        Parameters
        ----------
        parent : KDEDensity
            Fitted KDE density over multiple variables.
        fixed : dict[str, float]
            Fixed variables and their values. At least a free variable must remain.
        var_names : list[str]
            Names of the remaining free variables.

        """
        self._parent = parent
        self._fixed = fixed
        self.var_names: list[str] = var_names
        self.domain: dict[str, tuple[float, float]] = {
            v: parent.domain[v] for v in var_names
        }

    def log_pdf(self, values: dict[str, float]) -> float:
        """Evaluate the log (joint) density at the given free-variable values."""
        return self._parent.log_pdf({**self._fixed, **values})

    def _pdf_batch(self, grid: np.ndarray) -> np.ndarray:
        """Vectorised PDF evaluation over a 1-D grid of the single free variable.

        Only callable when this density has exactly one free variable.
        """
        if len(self.var_names) != 1:
            msg = "_pdf_batch only supports 1-D ConditionalDensity."
            raise RuntimeError(msg)
        n = len(grid)
        rows = np.column_stack(
            [
                np.full(n, self._fixed[v]) if v in self._fixed else grid
                for v in self._parent.var_names
            ],
        )

        return self._parent._pdf_batch(rows)  # noqa: SLF001


class SplineDensity:
    """Fast 1-D density backed by a pre-computed normalised cubic spline.

    Wraps any 1-D density object (``KDEDensity`` or ``ConditionalDensity``)
    and pre-computes a cubic spline so that every subsequent call to
    ``log_pdf`` is a O(1) spline lookup rather than a full KDE evaluation.
    This makes it suitable for use inside MCMC hot loops.

    Returns ``-inf`` for values outside the domain.

    Examples
    --------
    >>> cond = kde_2d.conditional(log_P=1.2)
    >>> fast = SplineDensity(cond, n_grid=1000)
    >>> fast.log_pdf({"pc1": 0.5})

    """

    def __init__(
        self,
        density: KDEDensity | KDEConditionalDensity,
        n_grid: int = 1000,
    ) -> None:
        """Initialize a SplineDensity for a one-dimensional density distribution.

        Parameters
        ----------
        density : KDEDensity | KDEConditionalDensity
            Any 1-D density object (``var_names`` must have exactly one entry).
        n_grid : int, optional
            Number of grid points for spline construction, by default 1000.

        Raises
        ------
        ValueError
            If the density distribution is not 1-dimensional.
        ValueError
            If the density cannot be normalized.

        """
        if len(density.var_names) != 1:
            msg = f"SplineDensity requires a 1-D density, got {len(density.var_names)}."
            logger.error(msg)
            raise ValueError(msg)

        var = density.var_names[0]
        lo, hi = density.domain[var]
        grid = np.linspace(lo, hi, n_grid)

        # Vectorised PDF evaluation (avoids n_grid separate KDE calls)
        if hasattr(density, "_pdf_batch"):
            raw_pdf = density._pdf_batch(grid)  # noqa: SLF001
        else:
            raw_pdf = np.array([np.exp(density.log_pdf({var: x})) for x in grid])

        # Normalise then take log (numerically stable via max-shift)
        raw_pdf = np.clip(raw_pdf, 0.0, None)
        norm = sp.integrate.simpson(raw_pdf, x=grid)
        if norm <= 0:
            msg = f"Density integrates to {norm} — cannot normalise."
            logger.error(msg)
            raise ValueError(msg)

        log_pdf_norm = np.log(np.clip(raw_pdf / norm, 1e-300, None))

        self.var_names: list[str] = [var]
        self.domain: dict[str, tuple[float, float]] = {var: (lo, hi)}
        self._var = var
        self._lo = lo
        self._hi = hi
        self._spline = sp.interpolate.CubicSpline(grid, log_pdf_norm)
        self._grid = grid

    def log_pdf(self, values: dict[str, float]) -> float:
        """Evaluate the normalised log density (spline lookup)."""
        val = float(values[self._var])
        if val < self._lo or val > self._hi:
            return -np.inf
        return float(self._spline(val))

    def pdf(self, values: dict[str, float]) -> float:
        """Evaluate the normalised probability density."""
        lp = self.log_pdf(values)
        return float(np.exp(lp)) if np.isfinite(lp) else 0.0


class UniformDensity:
    """Uniform density on [lo, hi) — same interface as SplineDensity."""

    def __init__(self, var_name: str, lo: float, hi: float) -> None:
        """Initialize a uniform density between ``lo`` and ``hi``.

        Only the lower bound is included. Half-open interval.

        Parameters
        ----------
        var_name: str
            Name of the variable.
        lo: float
            Lower bound.
        hi: float
            Upper bound

        """
        self.var_names = [var_name]
        self.domain = {var_name: (lo, hi)}
        self._lo, self._hi = lo, hi
        self._log_norm = -np.log(hi - lo)  # 0.0 for U(0,1)

    def log_pdf(self, values: dict) -> float:
        """Evaluate the normalised log density (between the extremes)."""
        val = float(values[self.var_names[0]])
        return self._log_norm if self._lo <= val <= self._hi else -np.inf

    def pdf(self, values: dict[str, float]) -> float:
        """Evaluate the normalised probability density."""
        lp = self.log_pdf(values)
        return float(np.exp(lp)) if np.isfinite(lp) else 0.0
