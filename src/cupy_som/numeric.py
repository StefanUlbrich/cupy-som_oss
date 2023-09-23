"""Handles import of cupy. Exports ``cupy_som.numeric.xp`` which is either ``numpy`` or ``cupy`` (if available)."""
__all__ = ["Array", "xp", "HAS_CUPY"]

import logging
from typing import TypeAlias

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
