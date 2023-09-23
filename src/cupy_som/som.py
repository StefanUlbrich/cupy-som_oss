# pyright: reportMissingImports=false

from dataclasses import dataclass
from enum import Enum
import logging

import numpy as np

from cupy_som.numeric import Array, xp

logger = logging.getLogger(__name__)


class Topology(Enum):
    #: Topology using the Euclidean norm
    EUCLIDEAN = 1
    #: Topology using the cosine distance (experimental)
    COSINE = 2


@dataclass
class SelfOrganizingMap:
    """Data class for a self-organizing map

    args:
        latent (xp.ndarray): Topology of the network
        codebook (xp.ndarray): Feature vectors of the neurons
    """

    n_neurons: int | tuple[int]
    latent_dim: int
    n_features: int
    topology: Topology = Topology.EUCLIDEAN
    chunk_size: int = 1000

    latent: Array | None = None
    codebook: Array | None = None

    def __post_init__(self):
        match self.topology:
            case Topology.EUCLIDEAN:
                if not isinstance(self.n_neurons, tuple):
                    self.n_neurons = (self.n_neurons,) * self.latent_dim
                self.latent = xp.array(list(xp.ndindex(*self.n_neurons))).astype(xp.double) / (
                    xp.array(self.n_neurons) - 1
                )
            case Topology.COSINE:
                if self.latent is None:
                    raise NotImplementedError(
                        "No default topology implemented for the cosine distance metric. "
                        "Consider a torus or ico-shpere."
                    )
            case _:
                raise NotImplementedError("This topology is not implemented")

        assert self.latent is not None
        self.codebook = xp.random.rand(self.latent.shape[0], self.n_features)

    def adapt(self, sample: Array, rate: float, influence: float) -> None:
        """A single update step

        Args:
            samples (xp.ndarray): The training samples
            rate (float): Learning rate
            influence (float): Influence
        """

        assert self.latent is not None
        assert self.codebook is not None

        _, output_dim = self.codebook.shape  # (3)

        # # Differences between the neurons weights and the sample
        diffs = self.codebook - sample[xp.newaxis, :]  # (4)

        # Neural coordinates of the winning neuron (BMU)
        winning = self.latent[xp.argmin(xp.linalg.norm(diffs, axis=1))]  # (5)
        # winning = self.get_winning(sample)

        # TODO compute the new influence based on scalar product
        # Influence of the BMU ∀ neurons

        match self.topology:
            case Topology.EUCLIDEAN:
                dist = xp.sum((self.latent - winning[xp.newaxis, :]) ** 2, axis=1)
            case Topology.COSINE:
                dist = xp.arccos(xp.clip(self.latent @ winning.T, -1.0, 1.0)) ** 2
            case _:
                raise NotImplementedError("This topology is not implemented")

        neighborhood = xp.exp(-0.5 * dist / influence**2)  # (6)

        # Update the codebook
        self.codebook -= rate * diffs * xp.tile(neighborhood[:, xp.newaxis], (1, output_dim))  # (7)

    def get_winning(self, samples: Array, k: int = 1) -> Array:
        """Get winning neurons for a set of samples

        Args:
            samples (Array): Samples, ``(n_samples, input_dim)``
            k(int): Retrieve the `k` best matching neurons

        Returns:
            (Array, Array):
                The :math:`k` best matching units (shape ``(n_samples, k, latent_dim)``)
                and their indices (``(n_samples, k)``)
        """
        # TODO check whether works with  1D too!
        samples = xp.atleast_2d(samples)
        assert self.latent is not None
        assert self.codebook is not None
        n_neurons, output_dim = self.codebook.shape  # (3)

        n_samples = samples.shape[0]

        winning = xp.zeros((n_samples, k, self.latent.shape[1]))
        sorted = xp.zeros((n_samples, k))

        for start in range(0, n_samples, self.chunk_size):
            end = min(start + self.chunk_size, n_samples)

            diffs = self.codebook[xp.newaxis, :, :] - samples[start:end, np.newaxis, :]  # (4)
            dists = xp.linalg.norm(diffs, axis=2)

            # winning[start:end, :] = self.latent[xp.argmin(dists, axis=1)]

            _sorted = np.argsort(dists, axis=1)[:, :k]
            winning[start:end, :] = self.latent[_sorted]
            sorted[start:end, :] = _sorted

        return winning, sorted

    def get_nearest_neurons(self, winning: Array, k: int) -> Array:
        """Get the :math:`k` nearest neighbors of :math:`l` winning neurons in *latent* space.

        Args:
            winning (Array): The *weights* of the :math:`l` winning neurons. Shape ``(n_samples, l, latent_dim)``
            k (int): Determins how many neighbors to return :math:`k`

        Returns:
            Array: Shape: ``(n_samples, l, k)``
        """

        match self.topology:
            case Topology.COSINE:
                # (n_samples, l, latent_dim) @ (latent_dim, n_neurons) = (n_samples, l, n_neurons)
                neighbors = np.argsort(winning @ self.latent.T, axis=2)
            case _:
                raise NotImplementedError("This topology is not implemented")

        # needs to be reversed
        return self.latent[neighbors[:, :, :-k:-1]], neighbors[:, :, :-k:-1]

    def batch(self, samples: Array):
        ...

    def online(
        self,
        samples: Array,
        rate: tuple[float, float] = (0.7, 0.1),
        influence: tuple[float, float] | None = None,
        epochs: int = 5,
    ) -> None:
        """Train a SOM.

        Args:
            samples (xp.ndarray): The training samples
            rate_start (float): Learning rate at the start (≤ 1 ∧ > 0)
            rate_end (float): Learning rate at the end.
            influence_start (float): Influence of the BMU at the beggining (>0)
            influence_end (float): Influence of the BMU at the end.
        """
        assert self.latent is not None
        assert self.codebook is not None

        if influence is None:
            influence = (1.0 / self.latent.shape[0] * 30, 1.0 / self.latent.shape[0] * 0.5)

        n_samples, output_dim = samples.shape
        assert output_dim == self.codebook.shape[1]

        for e in range(epochs):
            logger.debug("Epoch %d", e)

            for i, x in enumerate(samples):
                rate_ = rate[0] * (rate[1] / rate[0]) ** (float(e * n_samples + i) / (epochs * n_samples))  # (8)
                influence_ = influence[0] * (influence[1] / influence[0]) ** (
                    float(e * n_samples + i) / (epochs * n_samples)
                )  # (9)
                self.adapt(x, rate_, influence_)

                if i % 10000 == 0:
                    logger.debug("Sample %d, rate: %.2f, influence: %.2f", i, rate_, influence_)