"""Phased Principal Components model."""  # noqa: INP001

import numpy as np
from sklearn.decomposition import PCA

import velocefw as vf


class ClusterOffsets(vf.model.BaseModel):
    """Independent constant offset (systemic velocity) per cluster of epochs.

    Each observation belongs to a cluster (e.g. an observing campaign or
    instrument grouping) and every cluster gets its own free offset parameter.
    The cluster ids of the observations used for fitting are stored at
    construction time, so the model can be evaluated by the fitters without
    extra keyword arguments. When evaluating on a different input (e.g. a
    phase grid for plotting), pass ``cluster_ids`` explicitly as a keyword
    argument with one integer id per input point.

    With ``n_clusters=1`` the model reduces to a single shared systemic
    velocity ``v_gamma``.
    """

    def __init__(
        self,
        n_clusters: int = 1,
        cluster_ids: np.ndarray | None = None,
        name: str = "offsets",
    ) -> None:
        """Initialize the model with the number of clusters and default ids."""
        super().__init__(name=name)
        self.n_clusters = int(n_clusters)
        if cluster_ids is not None:
            cluster_ids = np.asarray(cluster_ids, dtype=int)
            if cluster_ids.min() < 0 or cluster_ids.max() >= self.n_clusters:
                msg = "cluster_ids must be integers in [0, n_clusters)."
                raise ValueError(msg)
        self.default_cluster_ids = cluster_ids

    @property
    def n_params(self) -> int:
        """One offset parameter per cluster."""
        return self.n_clusters

    @property
    def param_names(self) -> list[str]:
        """Parameter names: a single v_gamma or one offset per cluster."""
        if self.n_clusters == 1:
            return ["v_gamma"]
        return [f"offset_{i}" for i in range(self.n_clusters)]

    def evaluate(
        self,
        theta: list | np.ndarray,
        x: float | list | np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """Evaluate the offset of each input point from its cluster id."""
        theta = np.asarray(theta, float).ravel()
        x = np.atleast_1d(np.asarray(x, float))
        if self.n_clusters == 1:
            return np.full_like(x, theta[0])
        cluster_ids = kwargs.get("cluster_ids", self.default_cluster_ids)
        if cluster_ids is None:
            msg = f"{self.name}: cluster_ids required (none stored at init)."
            raise ValueError(msg)
        cluster_ids = np.asarray(cluster_ids, dtype=int)
        if cluster_ids.shape != x.shape:
            msg = f"{self.name}: cluster_ids shape {cluster_ids.shape} "
            msg += f"does not match x shape {x.shape}."
            raise ValueError(msg)
        return theta[cluster_ids]


class PhaseShiftLCPC(vf.model.BaseModel):
    """Phase Shifted Linear combination of the PCA components."""

    def __init__(
        self,
        pca: PCA,
        mean_training_sample: np.ndarray,
        phase_grid: np.ndarray | None = None,
        name: str = "pslcc",
    ) -> None:
        """Initialize the model with the PCA components and the mean training sample."""
        super().__init__(name=name)
        self.pca = pca
        if phase_grid is None:
            phase_grid = np.linspace(0.0, 1.0, 1000)
        self.phase_grid = phase_grid
        self.components = pca.components_.copy()
        self._components_T = self.components.T.copy()
        self.removed_mean = mean_training_sample

    @property
    def n_params(self) -> int:
        """Number of parameters is equal to the number of PCA components."""
        return self.components.shape[0] + 1

    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
        # pc1, pc2, ..., pcN
        return [f"pc{i}" for i in range(1, self.n_params)] + ["delta_phi"]

    def evaluate(
        self,
        theta: list | np.ndarray,
        x: float | list | np.ndarray,
        **kwargs: object,  # noqa: ARG002
    ) -> np.ndarray:
        """Evaluate the model."""
        # Reshape theta to be a 1-dimensional array
        theta = np.asarray(theta).ravel()
        new_x = (x + theta[-1]) % 1.0
        # Evaluate the linear combination
        return np.interp(
            new_x,
            self.phase_grid,
            self.removed_mean + np.dot(self._components_T, theta[:-1]),
        )
