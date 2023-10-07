"""Handles import of cupy. Exports ``cupy_som.numeric.xp`` which is either ``numpy`` or ``cupy`` (if available)."""
__all__ = ["Array", "xp", "HAS_CUPY"]

import logging
import sys
from dataclasses import dataclass
from math import ceil, prod
from typing import TypeAlias

import numpy as np

try:
    import cupy as xp  # noqa
    from cupy.typing import NDArray

    #: indicates whether cupy could be imported
    HAS_CUPY = True

except ImportError:
    import numpy as xp
    from numpy.typing import NDArray

    HAS_CUPY = False

logger = logging.getLogger(__name__)

if HAS_CUPY:
    logging.info("Using Cupy")
else:
    logging.warning("Not using Cupy")

#: Alias for the array type used in this package
Array: TypeAlias = NDArray[xp.double]


# Automatic memory handling


@dataclass
class Memory:
    """Memory limit in bytes"""

    bytes: int


@dataclass
class Rows:
    """Row limit to manually set the chunk size."""

    value: int


@dataclass
class Device:
    """Limits per device (GPU)"""

    device: int = 0


def bytes_to_gb(bytes: int) -> float:
    """Helper function to convert bytes to gigabytes"""
    return bytes / 1024 / 1024


#: Algebraic data type / Variadic to describe memory limits
ChunkLimit: TypeAlias = Memory | Rows | Device | None


def get_chunk_size(
    shape: tuple[int, ...],
    chunk_size: ChunkLimit = None,
    axis: int = 0,
    dtype: Array | np.dtype | str = "float64",
) -> tuple[int, int]:
    """Computes the optimal size of chunks (e.g., rows) that fit into memory
    given limits.

    Args:
        shape (tuple[int, ...]): The shape of the total memory to be alloted (if it were a single array).
        chunk_size (ChunkLimit, optional): Memory constraints. Defaults to None.
        axis (int, optional): Along which axis chunking is desired. Defaults to 0.
        dtype (Array | np.dtype | str, optional): The datatype of memory. Defaults to "float64".

    Returns:
        tuple[int, int]: The chunk size and the resulting number of iterations
    """
    match chunk_size:
        case Memory(bytes):
            itemsize = np.dtype(dtype).itemsize
            cells_per_slice = prod(v for i, v in enumerate(shape) if i != axis)
            bytes_per_slice = cells_per_slice * np.dtype(dtype).itemsize
            rows = bytes // bytes_per_slice
            total_size = prod(shape) * itemsize
            logger.debug(
                "Total size: %.2f (%.2f GB), Chunk size: %d (%.2f GB)",
                total_size,
                bytes_to_gb(total_size),
                rows,
                bytes_to_gb(rows * bytes_per_slice),
            )
            return rows, ceil(shape[axis] / rows)
        case Rows(value):
            print("Rows")
            return value, ceil(shape[axis] / value)
        case Device(device):
            print("Device")
            if not HAS_CUPY:
                raise ValueError("Cupy not enabled")

            used, total = xp.cuda.Device(device).mem_info
            return get_chunk_size(shape, Memory(total - used))
        case None:
            return sys.maxsize, 1
        case _:
            raise ValueError("Wrong argument type")
