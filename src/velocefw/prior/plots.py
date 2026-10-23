"""Visualisation utilities for KDE densities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from velocefw.prior.density import KDEDensity


def plot_kde_joint(
    kde: KDEDensity,
    data: pd.DataFrame,
    ax: Axes | None = None,
    n_grid: int = 80,
    cmap: str = "PuRd",
    scatter_kw: dict | None = None,
    test_data: pd.DataFrame | None = None,
    test_scatter_kw: dict | None = None,
) -> tuple[Figure, Axes]:
    """Plot a 2-D KDE density as a filled-contour overlay on a scatter plot.

    The first variable in ``kde.var_names`` is mapped to the x-axis and the
    second to the y-axis. Suitable for visualising ``(conditioning, target)``
    distributions before building priors.

    Parameters
    ----------
    kde :
        A fitted 2-D ``KDEDensity``.
    data :
        Training data (must contain the KDE's variable columns).
    ax :
        Existing axes to draw on. A new figure is created if ``None``.
    n_grid :
        Grid resolution for the contour plot.
    cmap :
        Matplotlib colormap for the density contours.
    scatter_kw :
        Extra keyword arguments forwarded to the training-data scatter call.
    test_data :
        Optional held-out data to overlay with a different marker.
    test_scatter_kw :
        Extra keyword arguments for the test-data scatter call.

    Returns
    -------
    (Figure, Axes)

    Raises
    ------
    ValueError
        If the KDE does not have exactly 2 variables.

    """
    if len(kde.var_names) != 2:  # noqa: PLR2004
        raise ValueError(
            f"plot_kde_joint requires a 2-D KDE, got {len(kde.var_names)} variables.",
        )

    x_var, y_var = kde.var_names

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # ── Training scatter ─────────────────────────────────────────────────────
    _scatter_kw = dict(
        s=14,
        marker="o",
        edgecolors="k",
        linewidths=0.6,
        zorder=2,
        label="Train",
    )
    if scatter_kw:
        _scatter_kw.update(scatter_kw)
    x_train = data[x_var].to_numpy(dtype=float)
    y_train = data[y_var].to_numpy(dtype=float)
    valid = ~(np.isnan(x_train) | np.isnan(y_train))
    ax.scatter(x_train[valid], y_train[valid], **_scatter_kw)

    # ── Density contour ───────────────────────────────────────────────────────
    x_grid = np.linspace(*kde.domain[x_var], n_grid)
    y_grid = np.linspace(*kde.domain[y_var], n_grid)
    xg, yg = np.meshgrid(x_grid, y_grid, indexing="ij")
    rows = np.column_stack([xg.ravel(), yg.ravel()])
    pdf = kde._pdf_batch(rows).reshape(xg.shape)
    ax.contourf(xg, yg, pdf, levels=12, cmap=cmap, zorder=0, alpha=0.85)

    # ── Test scatter (optional) ───────────────────────────────────────────────
    if test_data is not None:
        _test_kw = {
            "s": 18,
            "marker": "^",
            "edgecolors": "k",
            "linewidths": 0.6,
            "zorder": 3,
            "label": "Test",
        }
        if test_scatter_kw:
            _test_kw.update(test_scatter_kw)
        x_test = test_data[x_var].to_numpy(dtype=float)
        y_test = test_data[y_var].to_numpy(dtype=float)
        valid_test = ~(np.isnan(x_test) | np.isnan(y_test))
        ax.scatter(x_test[valid_test], y_test[valid_test], **_test_kw)

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)

    # pyrefly: ignore [bad-return]
    return fig, ax
