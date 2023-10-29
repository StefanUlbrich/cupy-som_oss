# pyright: reportMissingImports=false

import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum

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
    # TODO: feature_topology = Topology.EUCLIDEAN
    chunk_size: int = 1000

    latent: Array | None = None
    codebook: Array | None = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def __post_init__(self) -> None:
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

        DEPRECATED

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
        # winning, _ = self.get_winning(sample)

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

    def iter_winning(
        self, samples: Array, k: int, random: bool = False
    ) -> Generator[tuple[Array, Array, Array, tuple[int, int]], None, None]:
        """Get winning neurons for a set of samples

        Args:
            samples (Array): Samples, ``(n_samples, input_dim)``
            k(int): Retrieve the `k` best matching neurons
            random (bool):
                Whether to choose the winners randomly rather
                than based on distance.

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

            # match (self.feature_topology):
            #   case TOPOLOGY.EUCLEDIAN: ...
            diffs = self.codebook[xp.newaxis, :, :] - samples[start:end, xp.newaxis, :]  # (4)
            dists = xp.linalg.norm(diffs, axis=2)
            #   case TOPOLOGY.ANGULAR: ...

            if not random:
                if k == 1:
                    indices = xp.argmin(dists, axis=1)[:, xp.newaxis]
                else:
                    indices = xp.argsort(dists, axis=1)[:, :k]
            else:
                indices = xp.array(self.rng.integers(0, self.n_neurons, (end - start, k)))
                if start == 0:
                    logger.debug("Random initialization")

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

        for chunk_indices, chunk_latent, _, idx in self.iter_winning(samples, k):
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

    def batch(
        self,
        data: Array,
        influences: tuple[float, ...],
        max_iterations: int = 30,
        sampling_ratio: float | None = None,
        min_change: float = 0.0,
    ) -> None:
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
            sampling_ratio: float | None:
                Stochastic gradient descent: When set to value in :math:`(0,1)`,
                a subset of the data set is used in each EM step.
                The data sub set needs to be representative and should not be too small:
                The algorithm determines convergence if all samples are assigned to the
                same BMU as in the previoius run—with a small sample set this is more
                probable to prematurely happen.

        """
        assert self.latent is not None
        assert self.codebook is not None

        n_samples, _ = data.shape

        self.codebook[:] = 0
        ## exploration -> exploitation

        for i, influence in enumerate(influences):
            denominator = xp.zeros(self.n_neurons)
            last_indices = xp.ones((data.shape[0], 1)) * self.n_neurons

            for j in range(max_iterations):
                # On the first iteration each samle gets
                # a BMU randomly assigned to it. That way,
                # we get a better random initializaiton of the weights
                init_random = i == 0 and j == 0

                # sampling: Random sub set of the data
                # Note: on the very first iteration (i==0,j==0),
                # we need to process all samples
                if sampling_ratio is not None and not init_random:
                    end = int(n_samples * sampling_ratio)
                    permutation = xp.asarray(self.rng.permutation(n_samples)[:end])
                    samples = data[permutation]
                else:
                    permutation = xp.arange(data.shape[0]).reshape(1, -1)
                    samples = data

                # stop criterion: if indices don't change anymore, so first let's get current winners
                # for the sample in this loop
                # _, last_indices = self.get_winning(samples, 1)

                # allocate/initializes indices, update
                update = xp.zeros_like(self.codebook)
                indices = xp.zeros((samples.shape[0], 1))

                ## Iterate over chunks that fit into the memory.
                for chunk_indices, latent, diffs, idx in self.iter_winning(samples, 1, random=init_random):
                    # logger.debug("start: %i, end: %i, winning: %s\n%s", idx[0], idx[1], winning.shape, winning)

                    # logger.debug("%s", diffs.shape)

                    indices[idx[0] : idx[1], :] = chunk_indices

                    latent = latent[:, 0, :]  # remove unnecessary dim (special case k=1)

                    # (chunk_size, latent_dim) , (latent_dim, n_neurons) -> (chunk_size, n_neurons)
                    dist = xp.arccos(xp.clip(latent @ self.latent.T, -1.0, 1.0))

                    # (n_samples, n_neurons)
                    neighborhood = xp.exp(-0.5 * dist**2 / influence**2)
                    denominator += xp.sum(neighborhood, axis=0)
                    # neighborhood /= xp.sum(neighborhood, axis=0)

                    update -= xp.sum(neighborhood[:, :, xp.newaxis] * diffs, axis=0)

                if xp.allclose(indices, last_indices[permutation]):
                    logger.info("Converged after %i iterations", j)
                    break  # converged

                if xp.sum(xp.isclose(indices, last_indices[permutation])) / samples.shape[0] > 1.0 - min_change:
                    logger.info("Converged after %i iterations", j)
                    break  # converged

                logger.debug(
                    "Iteration: %i, indices: %s, indices: %s, stable: %d (%.2f%%)",
                    j,
                    last_indices[permutation].flatten(),
                    last_indices.flatten(),
                    xp.sum(xp.isclose(indices, last_indices[permutation])),
                    xp.sum(xp.isclose(indices, last_indices[permutation])) / samples.shape[0] * 100,
                )

                last_indices[permutation] = indices
                self.codebook += update / denominator[:, xp.newaxis]
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
