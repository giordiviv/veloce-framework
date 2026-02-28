"""Handle parametrization of model parameters.

The Parametrization class defines how free parameters map to the full global
theta vector.  It includes a static method for creating an identity
parametrization (where all parameters are free) and a method for creating a
parametrization with shared parameters based on specified groups of indices that
should share the same free parameter. The mapping from free to full parameters
is implemented in the expand function, which constructs the full theta vector
from the free parameters according to the defined sharing structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Parametrization:
    """Parametrization of a model.

    Maps the free parameters to the full global theta vector, which is the
    concatenation of all node parameters.
    """

    full_ndim: int
    free_ndim: int
    expand: Callable[[np.ndarray], np.ndarray]  # maps free theta to full theta

    @staticmethod
    def identity(full_ndim: int) -> Parametrization:
        """Parametrization where all parameters are free (no sharing).

        Parameters
        ----------
        full_ndim : int
            Number of parameters in the full global theta vector.

        Returns
        -------
        Parametrization
            Identity parametrization where all parameters are free and there is
            a 1-to-1 mapping between free and full parameters.

        """

        def expand(theta_free: np.ndarray) -> np.ndarray:
            if len(theta_free) != full_ndim:
                msg = f"Expected free theta of len {full_ndim}, got {len(theta_free)}"
                logger.error(msg)
                raise ValueError(msg)
            return theta_free

        return Parametrization(full_ndim=full_ndim, free_ndim=full_ndim, expand=expand)

    @staticmethod
    def from_shared(  # noqa: C901
        full_ndim: int,
        shared_groups: Sequence[Sequence[int]],
    ) -> Parametrization:
        """Parametrization with shared parameters.

        Parameters
        ----------
        full_ndim : int
            Total number of parameters in the full global theta vector.
        shared_groups : Sequence[Sequence[int]]
            Each inner sequence is a group of indices in the full theta vector
            that should share the same free parameter. For example,
            [[0, 2], [1, 3, 4]] means that indices 0 and 2 share one free
            parameter, and indices 1, 3, and 4 share another free parameter.

        Returns
        -------
        Parametrization
            Parametrization where parameters in the same shared group are tied
            together and share the same free parameter.
            The mapping from free to full parameters is defined by the shared
            groups, and any indices not included in any group are treated as
            independent free parameters.

        Raises
        ------
        ValueError
            When any index in shared_groups is out of bounds (not in [0,
            full_ndim-1]) or when the size of the vector of free parameters does
            not match the expected number based on the shared groups.
            `free_ndim = full_ndim - (total_shared_indices - number_of_shared_groups)`

        """
        # Step 1: Initialize union-find structure
        parent = list(range(full_ndim))

        def find(index: int) -> int:
            """Find the root of the set containing 'index', with path compression."""
            while parent[index] != index:
                parent[index] = parent[parent[index]]  # Path compression
                index = parent[index]
            return index

        def union(index_a: int, index_b: int) -> None:
            """Merge the sets containing 'index_a' and 'index_b'."""
            root_a, root_b = find(index_a), find(index_b)
            if root_a != root_b:
                parent[root_b] = root_a

        # Step 2: Merge indices in the same shared group
        # Handle shared groups
        for group in shared_groups:
            if len(group) >= 2:  # noqa: PLR2004
                for i in group[1:]:
                    union(group[0], i)

        # Step 3: Collect all indices in each set
        classes: dict[int, list[int]] = {}
        for index in range(full_ndim):
            root = find(index)
            classes.setdefault(root, []).append(index)

        # Step 4: Assign a unique free parameter to each set
        # (Each set gets one free parameter)
        unique_roots = sorted(classes.keys())
        root_to_free_index = {root: i for i, root in enumerate(unique_roots)}
        num_free_params = len(unique_roots)

        # Build mapping from free to full
        def expand(theta_free: np.ndarray) -> np.ndarray:
            theta_free = np.asarray(theta_free, dtype=float).ravel()
            if len(theta_free) != num_free_params:
                msg = (
                    "Expected free theta of len "
                    f"{num_free_params}, got {len(theta_free)}"
                )
                logger.error(msg)
                raise ValueError(msg)
            theta_full = np.empty(full_ndim, dtype=float)
            for root, indices in classes.items():
                free_index = root_to_free_index[root]
                theta_full[indices] = theta_free[free_index]
            return theta_full

        return Parametrization(
            full_ndim=full_ndim,
            free_ndim=num_free_params,
            expand=expand,
        )
