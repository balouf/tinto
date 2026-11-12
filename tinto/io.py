"""Deterministic seeding helpers.

:func:`set_seed` reseeds both the :mod:`random` standard library module and
:mod:`numpy.random`, which together cover every RNG call inside Tinto when
Numba JIT is disabled. Under JIT, Numba ships its own RNG that this function
also reaches into through :func:`numpy.random.seed`, since Numba's
:mod:`numpy`-namespaced random calls share state with NumPy's global RNG.

Use :func:`set_seed` at the top of any docstring example that exercises
randomized code (shuffled node IDs, random initial coloring, Bernoulli
schedulers, Monte Carlo runs) to keep doctests reproducible.
"""

import random
import numpy as np
from numba import njit


@njit
def set_seed(seed=42):
    """Set the seed for all random number generators used in Tinto.

    This includes the Python built-in ``random`` module and NumPy's
    ``numpy.random`` module. Setting the seed ensures reproducibility of
    randomized algorithms in Tinto.

    Parameters
    ----------
    seed: :class:`int`
        Seed value to initialize the random number generators.

    Examples
    --------

    >>> set_seed()
    >>> random.randint(0, 100)
    81
    >>> np.random.randint(0, 100)
    51
    >>> set_seed(51)
    >>> random.randint(0, 100)
    31
    >>> np.random.randint(0, 100)
    57
    >>> set_seed()
    >>> random.randint(0, 100)
    81
    >>> np.random.randint(0, 100)
    51
    """
    random.seed(seed)
    np.random.seed(seed)
