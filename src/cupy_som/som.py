# pyright: reportMissingImports=false

from dataclasses import dataclass
from enum import Enum
from collections.abc import Generator

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
        n_neuron (int | tuple[int, ...]):
            Number of neurons. In case of Euclidean topologies, integer
            values become converted into a tuple of length ``latent_dim``.
            Alternatively, one can specify the number of neurons for each
            dimension indepedently as a tuple of length `latent_dim`. For
            sperical topologies only a integer value is allowed.
        latent_dim (int): Dimensionality of the latent space
        n_features (int): Dimensionality of the feature space
        topology (:class:`Topology`): Topology of the the latent space
        chunk_size (int):
            Split up computation on the GPU to respect memory. Note: this should
            rather be converted to memory size in the future (or read the available
            space automatically)
        latent (Array | None):
            The neurons' coordinates in the latent space. Automatically initialized
            according to the `topology` parameter if omitted. If given, `n_neuron`,
            `topology` and `latent_dim` are ignored.
        codebook (Array | None):
            The neurons' weights/connections in the feature space. Automatically initialized
            with random values if omitted. If not, the `n_features` parameter is ignored.
    """

    n_neurons: int | tuple[int, ...]
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
                if self.latent is None:
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
        self.latent_dim = self.latent.shape[1]

        if self.codebook is None:
            self.codebook = xp.random.rand(self.latent.shape[0], self.n_features)
        else:
            if self.codebook.shape[0] != self.latent.shape[0]:
                raise ValueError("Dimensions of the input does not match")
            self.n_features = self.codebook.shape[1]

    def adapt(self, sample: Array, rate: float, influence: float) -> None:
        """A single update step

        Args:
            samples (xp.ndarray): The training samples
            rate (float): Learning rate
            influence (float): Influence
        """

        # formula numbers correspond to
        # https://www.lemonfold.io/posts/2023/citrate/cerebral/cerebral_part1_motivation/#first-version-of-the-algorithm

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

    def get_winning_chunks(
        self, samples: Array, k: int
    ) -> Generator[tuple[Array, Array, Array, tuple[int, int]], None, None]:
        """Get winning neurons for a set of samples

        Args:
            samples (Array): Samples, ``(n_samples, input_dim)``
            k(int): Retrieve the `k` best matching neurons

        Yields:
            (Array, Array, Array, (int,int)):
                For each chunk of samples of size ``self.chunk_size``,
                the indices of the :math:`k` winning neurons, their latent coordinates,
                the differences of the codebook to the samples, a tuple with the start
                and stop index of chunk.
        """

        assert self.latent is not None
        assert self.codebook is not None

        n_samples = samples.shape[0]

        for start in range(0, n_samples, self.chunk_size):
            end = min(start + self.chunk_size, n_samples)

            diffs = self.codebook[xp.newaxis, :, :] - samples[start:end, np.newaxis, :]  # (4)
            dists = xp.linalg.norm(diffs, axis=2)

            indices = np.argsort(dists, axis=1)[:, :k]
            latent = self.latent[indices]

            yield indices, latent, diffs, (start, end)

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

        n_samples = samples.shape[0]

        latent = xp.zeros((n_samples, k, self.latent.shape[1]))
        indices = xp.zeros((n_samples, k))

        for chunk_indices, chunk_latent, _, idx in self.get_winning_chunks(samples, k):
            latent[idx[0] : idx[1], :] = chunk_latent
            indices[idx[0] : idx[1], :] = chunk_indices

        return latent, indices

    def get_nearest_neurons(self, winning: Array, k: int) -> Array:
        """Get the :math:`k` nearest neighbors of :math:`l` winning neurons in *latent* space.

        Args:
            winning (Array): The *weights* of the :math:`l` winning neurons. Shape ``(n_samples, l, latent_dim)``
            k (int): Determins how many neighbors to return :math:`k`

        Returns:
            Array: Shape: ``(n_samples, l, k)``
        """

        assert self.latent is not None
        match self.topology:
            case Topology.COSINE:
                # (n_samples, l, latent_dim) @ (latent_dim, n_neurons) = (n_samples, l, n_neurons)
                neighbors = np.argsort(winning @ self.latent.T, axis=2)
            case _:
                raise NotImplementedError("This topology is not implemented")

        # FIXME: k should not include the neuron itself and has an off-by-one error

        # needs to be reversed
        return self.latent[neighbors[:, :, :-k:-1]], neighbors[:, :, :-k:-1]

    def batch(self, samples: Array, influences: tuple[float, ...]) -> None:
        """Batch learning similar to expectation maximization

        Args:
            samples (Array): The samples to learn from
            influences (tuple[float, ...]):
                To avoid "folds" in the map, multiple trainings
                with decreasing ranges of influence should be done
                (cf. "from exploration to exploitation"). See Section
                7.2, p. 79 in `MATLAB Implementations and Applications
                of the Self-Organizing
                Map <http://docs.unigrafia.fi/publications/kohonen_teuvo/>`_

        """
        assert self.latent is not None
        assert self.codebook is not None

        ## exploration -> exploitation
        for influence in influences:
            # stop criterion: if indices don't change anymore
            last_indices = xp.ones((samples.shape[0], 1)) * self.n_neurons

            max_iterations = 30
            for i in range(max_iterations):
                ## expectation step (finding bmu)
                logger.debug("Iteration: %i, indices: %s", i, last_indices.flatten())

                # allocate/initializes indices, update

                update = xp.zeros_like(self.codebook)
                indices = xp.zeros_like(last_indices)

                for chunk_indices, winning, diffs, idx in self.get_winning_chunks(samples, k=1):
                    indices[idx[0] : idx[1], :] = chunk_indices

                    winning = winning[:, 0, :]  # remove unnecessary dim (k=1)

                    # (chunk_size, latent_dim) , (latent_dim, n_neurons) -> (chunk_size, n_neurons)
                    dist = xp.arccos(np.clip(winning @ self.latent.T, -1.0, 1.0))

                    # (n_samples, n_neurons)
                    neighborhood = xp.exp(-0.5 * dist**2 / influence**2)
                    neighborhood /= xp.sum(neighborhood, axis=0)

                    update += xp.sum(neighborhood[:, :, np.newaxis] * diffs, axis=0)

                if xp.allclose(indices, last_indices):
                    logger.info("Converged after %i iterations", i)
                    break  # converged

                # winning, indices = self.get_winning(samples, k=1)

                # # Check convergence

                # winning = winning[:, 0, :]  # remove unnecessary dim (k=1)

                # # (n_samples, latent_dim) , (latent_dim, n_neurons) -> (n_samples, n_neurons)
                # dist = xp.arccos(np.clip(winning @ self.latent.T, -1.0, 1.0))

                # ## maximization step (updating the neurons)
                # # (n_samples, □, n_features), (□, n_neurons, n_features) -> ...
                # diffs = samples[:, xp.newaxis, :] - self.codebook[np.newaxis, :, :]

                # # (n_samples, n_neurons)
                # neighborhood = xp.exp(-0.5 * dist**2 / influence**2)
                # neighborhood /= xp.sum(neighborhood, axis=0)

                # logger.debug("NaN in neighborhood: %s", np.count_nonzero(~np.isnan(neighborhood)))

                # # (n_samples, n_features, □), (n_samples, n_neurons, n_features) -> (n_neurons, n_features)
                # update = xp.sum(neighborhood[:, :, np.newaxis] * diffs, axis=0)

                self.codebook += update
                last_indices = indices
            else:
                logger.warning("Not converged for influence: %f", influence)

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
