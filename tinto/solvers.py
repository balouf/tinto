"""Solvers that recolor a batch of nodes and refresh the enabled cache.

A *solver* is a callable ``solver(config, decisions) -> None`` that, given
the indices returned by a scheduler, picks a new color for each decision
through a *colorer* function, writes the colors back into
:attr:`~tinto.configs.Config.colors`, and then invokes
:meth:`~tinto.configs.Config.check` on each touched node and its neighbors
to refresh the enabled mask.

A few colorers are provided:

- :func:`colorer_lowest` always picks the smallest color absent from the
  neighborhood.
- :func:`colorer_next` starts the search at ``me + 1`` and wraps around,
  favoring a *change* over a "canonical" choice.
- :func:`colorer_random` picks a random color absent from the neighborhood.
- :func:`colorer_random_blind` picks a random color different from the
  current one, without checking the neighborhood.
- :func:`colorer_next_blind` picks the next color (cyclic) after the
  current one, without checking the neighborhood.

:func:`colorer_lowest`, :func:`colorer_next` and :func:`colorer_random`
raise :class:`ValueError` if no free color exists, which can only happen
when the current configuration uses fewer colors than needed. The
``_blind`` variants never raise but may leave clashes behind.
Use :func:`make` to wrap a colorer into a compiled solver.
"""

from functools import lru_cache

from numba import njit
import numpy as np


@njit(fastmath=True)
def colorer_lowest(config, i):
    """Return the smallest color of ``range(k)`` absent from the neighborhood of `i`.

    Parameters
    ----------
    config: :class:`~tinto.configs.Config`
        Current coloring configuration.
    i: :class:`int`
        Node whose new color is requested.

    Returns
    -------
    :class:`int64`
        Smallest ``c`` such that no neighbor of `i` is currently
        colored `c`.

    Raises
    ------
    ValueError
        If every color in ``range(k)`` is represented in the
        neighborhood of `i`.

    Examples
    --------
    On a 3-node line with every node at color 0, node 1 has two
    neighbors of color 0 and k=3 colors available; the lowest free
    color is 1:

    >>> from tinto.graphs import line
    >>> from tinto.configs import make as make_config
    >>> Config = make_config()
    >>> c = Config(line(3))
    >>> int(colorer_lowest(c, 1))
    1

    On the 3-node complete graph with only 2 colors, every color is
    occupied in the neighborhood and the call raises:

    >>> import numpy as np
    >>> from tinto.graphs import Graph
    >>> ptr = np.array([0, 2, 4, 6], dtype=np.int32)
    >>> ind = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
    >>> c = Config(Graph(ptr, ind), k=2)
    >>> c.colors[:] = [0, 0, 1]
    >>> c.check_all()
    >>> colorer_lowest(c, 0)
    Traceback (most recent call last):
        ...
    ValueError: Not Enough Colors!!!
    """
    counts = config.count(i)
    for c in range(config.k):
        if counts[c] == 0:
            return c
    else:
        raise ValueError("Not Enough Colors!!!")


colorer_lowest.uses_ids = False


@njit(fastmath=True)
def colorer_next(config, i):
    """Return the next free color after ``config.colors[i]`` (cyclic).

    Scans ``(me + 1) % k, (me + 2) % k, ...`` and returns the first free
    value. Favors a change relative to the current color, which matters
    for rules that must avoid self-fixed points.

    Parameters
    ----------
    config: :class:`~tinto.configs.Config`
        Current coloring configuration.
    i: :class:`int`
        Node whose new color is requested.

    Returns
    -------
    :class:`int64`
        First free color encountered when iterating ``(me + 1 + c) % k``
        for ``c`` in ``range(k)``.

    Raises
    ------
    ValueError
        If no free color exists.

    Examples
    --------
    On a 3-node line with every node at color 0, node 1's first
    candidate is ``(0 + 1) % 3 = 1``, which is free:

    >>> from tinto.graphs import line
    >>> from tinto.configs import make as make_config
    >>> Config = make_config()
    >>> c = Config(line(3))
    >>> int(colorer_next(c, 1))
    1

    The cyclic search wraps around. On a 3-node line with
    ``colors = [0, 2, 0]``, node 1's search starts at color 0 (taken)
    and only finds 1 on the next iteration:

    >>> c.colors[:] = [0, 2, 0]
    >>> c.check_all()
    >>> int(colorer_next(c, 1))
    1

    The saturated K3-with-k=2 case raises:

    >>> import numpy as np
    >>> from tinto.graphs import Graph
    >>> ptr = np.array([0, 2, 4, 6], dtype=np.int32)
    >>> ind = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
    >>> c = Config(Graph(ptr, ind), k=2)
    >>> c.colors[:] = [0, 0, 1]
    >>> c.check_all()
    >>> colorer_next(c, 0)
    Traceback (most recent call last):
        ...
    ValueError: Not Enough Colors!!!
    """
    counts = config.count(i)
    me = config.colors[i]
    for c in range(config.k):
        prop = (me + 1 + c) % config.k
        if counts[prop] == 0:
            return prop
    else:
        raise ValueError("Not Enough Colors!!!")


colorer_next.uses_ids = False


@njit(fastmath=True)
def colorer_random(config, i):
    """Return a random free color for node `i`.

    Parameters
    ----------
    config: :class:`~tinto.configs.Config`
        Current coloring configuration.
    i: :class:`int`
        Node whose new color is requested.

    Returns
    -------
    :class:`int64`
        A random color in ``range(k)`` that is absent from the neighborhood
        of `i`.

    Raises
    ------
    ValueError
        If no free color exists.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make as make_config
    >>> from tinto.io import set_seed
    >>> set_seed(42)
    >>> Config = make_config()
    >>> c = Config(line(3))
    >>> solver = make(colorer=colorer_random)
    >>> decisions = np.array([0, 1, 2], dtype=np.int64)
    >>> solver(c, decisions)
    >>> c.colors.tolist()
    [1, 2, 1]
    >>> c.total
    0

    On the 3-node complete graph with only 2 colors, no color is free:

    >>> from tinto.graphs import Graph
    >>> ptr = np.array([0, 2, 4, 6], dtype=np.int32)
    >>> ind = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
    >>> c = Config(Graph(ptr, ind), k=2)
    >>> c.colors[:] = [0, 0, 1]
    >>> c.check_all()
    >>> colorer_random(c, 0)
    Traceback (most recent call last):
        ...
    ValueError: Not Enough Colors!!!
    """
    counts = config.count(i)
    free = np.array([c for c in range(config.k) if counts[c] == 0])
    size = len(free)
    if size > 0:
        return free[np.random.randint(size)]
    raise ValueError("Not Enough Colors!!!")


colorer_random.uses_ids = False


@njit(fastmath=True)
def colorer_random_blind(config, i):
    """Return a random color different from ``config.colors[i]`` without checking the neighborhood.

    Parameters
    ----------
    config: :class:`~tinto.configs.Config`
        Current coloring configuration.
    i: :class:`int`
        Node whose new color is requested.

    Returns
    -------
    :class:`int64`
        A color drawn uniformly from ``range(k)`` excluding the current
        color ``config.colors[i]``. Always guarantees a change.

    Notes
    -----
    The returned color may still clash with a neighbor: this colorer
    skips the neighborhood scan and trades correctness for speed. Useful
    when modelling agents that detect a conflict locally but cannot
    enumerate which colors are free.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make as make_config
    >>> from tinto.io import set_seed
    >>> set_seed(40)
    >>> Config = make_config()
    >>> c = Config(line(3))
    >>> c.color_randomize()
    >>> c.colors.tolist()
    [2, 1, 0]
    >>> solver = make(colorer=colorer_random_blind)
    >>> decisions = np.array([0, 1, 2], dtype=np.int64)
    >>> solver(c, decisions)
    >>> c.colors.tolist()
    [0, 0, 2]
    >>> c.total
    2
    """
    prop = np.random.randint(config.k - 1)
    if prop >= config.colors[i]:
        prop += 1
    return prop


colorer_random_blind.uses_ids = False


@njit(fastmath=True)
def colorer_next_blind(config, i):
    """Return the next color after ``config.colors[i]`` (cyclic) without checking the neighborhood.

    Parameters
    ----------
    config: :class:`~tinto.configs.Config`
        Current coloring configuration.
    i: :class:`int`
        Node whose new color is requested.

    Returns
    -------
    :class:`int64`
        The color ``(config.colors[i] + 1) % config.k``.

    Notes
    -----
    The returned color may clash with a neighbor: this colorer skips
    the neighborhood scan and trades correctness for speed. Useful for
    modelling agents that can detect a conflict locally but cannot
    enumerate which colors are free.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make as make_config
    >>> from tinto.io import set_seed
    >>> set_seed(51)
    >>> Config = make_config()
    >>> c = Config(line(3))
    >>> c.color_randomize()
    >>> c.colors.tolist()
    [2, 1, 1]
    >>> c.total
    2
    >>> solver = make(colorer=colorer_next_blind)
    >>> decisions = np.array([0, 1, 2], dtype=np.int64)
    >>> solver(c, decisions)
    >>> c.colors.tolist()
    [0, 2, 2]
    >>> c.total
    2
    """
    return (config.colors[i] + 1) % config.k


colorer_next_blind.uses_ids = False


def make(colorer=None):
    """Build a compiled solver wired to `colorer`.

    The returned callable applies `colorer` to every decision, writes
    the new colors into :attr:`~tinto.configs.Config.colors` at once,
    then calls :meth:`~tinto.configs.Config.check` on every touched
    node and its neighbors to refresh the enabled mask.

    Parameters
    ----------
    colorer: callable or ``None``, default=``None``
        A ``colorer(config, i) -> int64`` picking the new color for
        node `i`. When ``None``, :func:`colorer_lowest` is used.

    Returns
    -------
    callable
        A :func:`numba.njit`-compiled ``solver(config, decisions) -> None``.

    Examples
    --------
    With the default colorer, recoloring node 1 of a 3-node line reaches
    a stable coloring in one step:

    >>> from tinto.graphs import line
    >>> from tinto.configs import make as make_config
    >>> Config = make_config()
    >>> c = Config(line(3))
    >>> solver = make()
    >>> decisions = np.array([1], dtype=np.int64)
    >>> solver(c, decisions)
    >>> c.colors.tolist()
    [0, 1, 0]
    >>> c.total
    0

    A custom colorer is accepted:

    >>> solver_next = make(colorer=colorer_next)
    >>> c = Config(line(3))
    >>> solver_next(c, decisions)
    >>> c.colors.tolist()
    [0, 1, 0]

    Repeated calls with the same (normalized) colorer hit the cache:

    >>> make() is make()
    True
    >>> make() is make(colorer_lowest)
    True
    >>> make(colorer_lowest) is make(colorer_next)
    False

    Notes
    -----
    Two calls with the same (normalized) colorer return the same
    compiled solver. Colorers are keyed by identity; ``None`` and
    :func:`colorer_lowest` are treated as the same key.
    """
    if colorer is None:
        colorer = colorer_lowest
    return _make_solver(colorer)


@lru_cache(maxsize=None)
def _make_solver(colorer):
    def solver(config, decisions):
        colors = np.empty(len(decisions), dtype=np.int64)
        for k, i in enumerate(decisions):
            colors[k] = colorer(config, i)
        for i, c in zip(decisions, colors):
            config.colors[i] = c
        for i in decisions:
            config.check(i)
            for j in config.graph.neighbors(i):
                config.check(j)

    compiled = njit(solver, fastmath=True)
    compiled.uses_ids = getattr(colorer, "uses_ids", False)
    return compiled
