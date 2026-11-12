"""Coloring configuration of a :class:`~tinto.graphs.Graph`.

This module defines :func:`make`, a factory that builds a :mod:`numba`
:class:`~numba.experimental.jitclass` container. A :class:`Config` tracks the
color assigned to each node of a graph and maintains, in sync, the set of
nodes for which a user-supplied *guard* predicate currently fires.
"""

from functools import lru_cache

from numba.experimental import jitclass
from numba import int64, bool
import numpy as np

from tinto.guards import color_clash
from tinto.graphs import Graph


def make(guard=None):
    """Build a :class:`Config` jitclass wired to `guard`.

    Because :mod:`numba` requires every :func:`~numba.experimental.jitclass`
    method to be specialized at class-definition time, the guard predicate is
    captured here as a closure rather than passed at instantiation.

    Parameters
    ----------
    guard: callable or ``None``, default=``None``
        Predicate ``guard(config, i) -> bool`` deciding whether node `i` is
        *enabled*. When ``None``, :func:`~tinto.guards.color_clash` is
        used, which enables a node iff it shares its color with a neighbor.

    Returns
    -------
    type
        A jitclass implementing the API documented on :class:`Config`.

    Notes
    -----
    Two calls with the same (normalized) guard return the same compiled
    jitclass — the class is built only once per distinct guard. Guards
    are keyed by identity; ``None`` and :func:`~tinto.guards.color_clash`
    are treated as the same key. Use
    ``tinto.configs._make_config.cache_clear()`` to drop the cache in the
    rare cases it is needed.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> Config = make()
    >>> c = Config(line(4))
    >>> c.total
    4

    Passing an explicit guard wires the returned class to it rather
    than to :func:`~tinto.guards.color_clash`:

    >>> from tinto.guards import color_min
    >>> ConfigMin = make(color_min)
    >>> cm = ConfigMin(line(4))
    >>> cm.total > 0
    True

    The number of available colors is adjustable post-construction:

    >>> c.set_k(5)
    >>> c.k
    5
    >>> c.local_count.shape
    (5,)

    And :meth:`Config.shuffle_ids` reseeds the per-Config id permutation
    without rebuilding the graph — useful between replicas when a guard
    breaks ties on :attr:`ids`. With the default guard
    (:func:`~tinto.guards.color_clash`, which ignores ids), :attr:`ids`
    starts empty and is populated on demand:

    >>> c.ids.tolist()
    []
    >>> c.shuffle_ids()
    >>> sorted(c.ids.tolist())
    [0, 1, 2, 3]

    A guard that reads ids (e.g. :func:`~tinto.guards.color_clash_tie`)
    flips the :attr:`Config.uses_ids` flag and makes
    :class:`Config.__init__` draw a fresh permutation directly, so that
    :meth:`Config.check_all` sees a valid tie-breaking order on the
    first call:

    >>> from tinto.guards import color_clash_tie
    >>> ConfigTie = make(color_clash_tie)
    >>> ConfigTie.uses_ids
    True
    >>> ct = ConfigTie(line(4))
    >>> sorted(ct.ids.tolist())
    [0, 1, 2, 3]

    Repeated calls with the same (normalized) guard hit the cache:

    >>> make() is make()
    True
    >>> make() is make(color_clash)
    True
    >>> make(color_clash) is make(color_min)
    False
    """
    if guard is None:
        guard = color_clash
    return _make_config(guard)


@lru_cache(maxsize=None)
def _make_config(guard):
    uses_ids = getattr(guard, "uses_ids", True)

    @jitclass
    class Config:
        """Dynamic coloring configuration of a :class:`~tinto.graphs.Graph`.

        A node is said to be *enabled* when the guard predicate captured at
        :func:`make` time fires on it. :class:`Config` keeps the
        enabled mask :attr:`enabled` and the enabled count :attr:`total` in
        sync with the current coloring :attr:`colors`; they can be recomputed
        from scratch by :meth:`check_all`, or updated one node at a time by
        :meth:`check`.

        Parameters
        ----------
        graph: :class:`~tinto.graphs.Graph`
            Underlying graph. All nodes start with color ``0``.
        k: :class:`int`, default=0
            Number of available colors. When left at the default, it is set
            to ``graph.max_degree + 1`` so that a proper coloring is always
            reachable by a greedy algorithm. Any strictly positive value
            overrides this default.

        Attributes
        ----------
        graph: :class:`~tinto.graphs.Graph`
            See the corresponding parameter.
        colors: :class:`~numpy.ndarray` of :class:`int64`, shape (n,)
            Current color of each node. Values lie in ``range(k)``.
        enabled: :class:`~numpy.ndarray` of :class:`bool`, shape (n,)
            Boolean mask: ``enabled[i]`` is ``True`` iff the guard fires on
            node `i` under the current coloring.
        total: :class:`int64`
            Number of enabled nodes, i.e. ``enabled.sum()``.
        k: :class:`int64`
            Number of available colors.
        local_count: :class:`~numpy.ndarray` of :class:`int64`, shape (k,)
            Scratch buffer holding the color histogram of a node's neighbors,
            (re)filled by :meth:`count` and reused by some guards.

        Examples
        --------
        On a line of 4 nodes all starting at color ``0``, every node clashes
        with a neighbor under the default guard:

        >>> from tinto.graphs import line
        >>> from tinto.configs import make
        >>> from tinto.io import set_seed
        >>> set_seed(0)
        >>> Config = make()
        >>> c = Config(line(4))
        >>> c.k
        3
        >>> c.total
        4
        >>> c.enabled.tolist()
        [True, True, True, True]
        >>> c.ids.tolist()
        []
        >>> c.shuffle_ids()
        >>> sorted(c.ids.tolist())
        [0, 1, 2, 3]
        """

        graph: Graph
        enabled: bool[:]  # type: ignore
        total: int64  # type: ignore
        k: int64  # type: ignore
        colors: int64[:]  # type: ignore
        local_count: int64[:]  # type: ignore
        ids: int64[:]  # type: ignore
        uses_ids: bool  # type: ignore

        def __init__(self, graph, k=0):
            self.graph = graph
            self.uses_ids = uses_ids
            if uses_ids:
                self.ids = np.random.permutation(graph.n)
            else:
                self.ids = np.empty(0, dtype=np.int64)
            self.enabled = np.zeros(graph.n, np.bool)
            self.colors = np.zeros(graph.n, dtype=np.int64)
            self.k = k if k > 0 else graph.max_degree + 1
            self.local_count = np.zeros(self.k, dtype=np.int64)
            self.total = 0
            self.check_all()

        def shuffle_ids(self):
            """Replace :attr:`ids` by a uniform random permutation of ``range(graph.n)``.

            Useful between replicas when the guard or colorer relies on
            node IDs to break ties, as in
            :func:`~tinto.guards.color_clash_tie`, so that successive runs
            do not share the same tie-breaking order. When the guard does
            not use ids, :attr:`ids` starts empty (see
            :class:`tinto.configs.Config`); calling this method populates
            it on demand.

            Examples
            --------
            >>> from tinto.graphs import line
            >>> from tinto.configs import make
            >>> Config = make()
            >>> c = Config(line(4))
            >>> c.ids.tolist()
            []
            >>> c.shuffle_ids()
            >>> sorted(c.ids.tolist())
            [0, 1, 2, 3]
            """
            self.ids = np.random.permutation(self.graph.n)

        def set_k(self, k):
            """Change the number of available colors and reallocate :attr:`local_count`.

            Existing values in :attr:`colors` are left untouched, so callers
            should follow up with :meth:`color_uniform` or
            :meth:`color_randomize` to ensure every node stays in ``range(k)``.

            Parameters
            ----------
            k: :class:`int`
                New number of colors. Must be strictly positive.

            Examples
            --------
            >>> from tinto.graphs import line
            >>> from tinto.configs import make
            >>> Config = make()
            >>> c = Config(line(4))
            >>> c.k
            3
            >>> c.set_k(5)
            >>> c.k
            5
            >>> c.local_count.shape
            (5,)
            """
            self.k = k
            self.local_count = np.zeros(self.k, dtype=np.int64)

        def count(self, i):
            """Return the color histogram of the neighbors of node `i`.

            The scratch buffer :attr:`local_count` is zeroed and refilled,
            then returned as a view: read the result before the next call to
            :meth:`count`, which will overwrite it.

            Parameters
            ----------
            i: :class:`int`
                Node index, with ``0 <= i < graph.n``.

            Returns
            -------
            :class:`~numpy.ndarray` of :class:`int64`, shape (k,)
                View of :attr:`local_count` where entry `c` holds the number
                of neighbors of `i` currently colored with `c`.

            Examples
            --------
            >>> from tinto.graphs import line
            >>> from tinto.configs import make
            >>> Config = make()
            >>> c = Config(line(4))
            >>> c.colors[:] = [0, 1, 2, 0]
            >>> c.check_all()
            >>> c.count(1).tolist()
            [1, 0, 1]
            """
            self.local_count[:] = 0
            for j in self.graph.neighbors(i):
                self.local_count[self.colors[j]] += 1
            return self.local_count

        def check(self, i):
            """Refresh ``enabled[i]`` and adjust :attr:`total` accordingly.

            Evaluates the guard on node `i`, stores the result in
            ``enabled[i]``, and increments or decrements :attr:`total` to
            reflect any change relative to the previous value.

            Parameters
            ----------
            i: :class:`int`
                Node index, with ``0 <= i < graph.n``.

            Returns
            -------
            :class:`bool`
                New value of ``enabled[i]``.

            Examples
            --------
            >>> from tinto.graphs import line
            >>> from tinto.configs import make
            >>> Config = make()
            >>> c = Config(line(3))
            >>> c.check(0)
            True
            >>> c.colors[0] = 1
            >>> c.check(0)
            False
            """
            old = self.enabled[i]
            new = guard(self, i)
            if new and not old:
                self.total += 1
            elif old and not new:
                self.total -= 1
            self.enabled[i] = new
            return new

        def check_all(self):
            """Recompute :attr:`enabled` and :attr:`total` from :attr:`colors`.

            Resets the bookkeeping and calls :meth:`check` on every node. Use
            this after any bulk edit of :attr:`colors` so that the enabled
            mask reflects the current coloring.

            Examples
            --------
            >>> from tinto.graphs import line
            >>> from tinto.configs import make
            >>> Config = make()
            >>> c = Config(line(3))
            >>> c.colors[:] = [0, 1, 0]
            >>> c.check_all()
            >>> c.total
            0
            >>> c.enabled.tolist()
            [False, False, False]
            """
            self.total = 0
            self.enabled[:] = False
            for i in range(self.graph.n):
                self.check(i)

        def color_uniform(self, color=0):
            """Reset every node to `color`, defaulting to ``0``, and refresh the enabled cache.

            Under the default guard, any two adjacent nodes then share the
            same color and are both enabled.

            Parameters
            ----------
            color: :class:`int`, default=0
                Color assigned to every node. Must lie in ``range(k)``.

            Examples
            --------
            >>> from tinto.graphs import line
            >>> from tinto.configs import make
            >>> Config = make()
            >>> c = Config(line(4))
            >>> c.colors[:] = [0, 1, 0, 1]
            >>> c.check_all()
            >>> c.total
            0
            >>> c.color_uniform()
            >>> c.total
            4

            A non-zero color is equally accepted, as long as it stays in
            ``range(k)``:

            >>> c.color_uniform(color=2)
            >>> c.colors.tolist()
            [2, 2, 2, 2]
            """
            assert 0 <= color < self.k
            self.colors[:] = color
            self.check_all()

        def color_randomize(self, k_init=0):
            """Assign a uniform random color to every node.

            Colors are drawn in ``range(k_init)``, the first `k_init` colors
            of the palette, or in ``range(k)`` when ``k_init`` is 0. A start
            with fewer colors than the palette (``k_init < Delta + 1``) is
            how the exact stabilization-time paper samples its initial
            colorings. The enabled cache is refreshed from the new coloring.

            Parameters
            ----------
            k_init: :class:`int`, default=0
                Number of initial colors, at most :attr:`k`; 0 means all of
                them.

            Examples
            --------
            >>> import numpy as np
            >>> from tinto.graphs import line
            >>> from tinto.configs import make
            >>> Config = make()
            >>> np.random.seed(0)
            >>> c = Config(line(10))
            >>> c.color_randomize()
            >>> bool(((c.colors >= 0) & (c.colors < c.k)).all())
            True
            >>> c.total == int(c.enabled.sum())
            True

            Two initial colors out of a palette of three:

            >>> c.color_randomize(2)
            >>> c.k, sorted(set(c.colors.tolist()))
            (3, [0, 1])
            """
            n_colors = self.k if k_init == 0 else k_init
            assert n_colors <= self.k
            for i in range(self.graph.n):
                self.colors[i] = np.random.randint(n_colors)
            self.check_all()

    Config.uses_ids = uses_ids
    return Config
