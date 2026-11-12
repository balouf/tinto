"""Guard predicates enabling nodes under the current coloring.

A *guard* is a :mod:`numba`-compatible predicate ``guard(config, i) -> bool``
deciding whether node `i` of a :class:`~tinto.configs.Config` is currently
*enabled*, i.e. eligible for recoloring. The guard is captured by
:func:`tinto.configs.make` at class-definition time and evaluated by
:meth:`~tinto.configs.Config.check`.

Four ready-made guards are provided, expressing increasingly conservative
enabling rules. :func:`color_clash` is the default used by
:func:`tinto.configs.make` when no guard is passed.
"""

from numba import njit


@njit(fastmath=True)
def color_clash(config, i):
    """Enable node `i` iff it shares its color with at least one neighbor.

    Parameters
    ----------
    config: :class:`~tinto.configs.Config`
        Coloring configuration whose :attr:`colors` and
        :attr:`graph` are read.
    i: :class:`int`
        Node index, with ``0 <= i < config.graph.n``.

    Returns
    -------
    :class:`bool`
        ``True`` iff at least one neighbor of `i` currently shares the
        color of `i`.

    Examples
    --------
    On a 3-node line with every node at color ``0``, every node clashes
    with a neighbor:

    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> Config = make()
    >>> c = Config(line(3))
    >>> color_clash(c, 0)
    True

    With a proper coloring, no node clashes:

    >>> c.colors[:] = [0, 1, 0]
    >>> color_clash(c, 1)
    False
    """
    me = config.colors[i]
    for j in config.graph.neighbors(i):
        if config.colors[j] == me:
            return True
    return False


color_clash.uses_ids = False


@njit(fastmath=True)
def color_clash_tie(config, i):
    """Enable node `i` iff it clashes with a neighbor and owns the lowest ID among the clashing cluster.

    Identical to :func:`color_clash` except that, among a group of nodes
    sharing the same color, only the one with the smallest
    :attr:`~tinto.graphs.Graph.ids` entry is enabled. This breaks ties
    deterministically, which matters for solvers that would otherwise
    recolor several conflicting nodes simultaneously into the same new
    color.

    Parameters
    ----------
    config: :class:`~tinto.configs.Config`
        Coloring configuration.
    i: :class:`int`
        Node index.

    Returns
    -------
    :class:`bool`
        ``True`` iff `i` clashes with a neighbor and no clashing neighbor
        has a smaller :attr:`~tinto.graphs.Graph.ids` value.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> import numpy as np
    >>> Config = make()
    >>> c = Config(line(3))

    Let's cheat and set the ids to the natural order:

    >>> c.ids = np.arange(3)
    >>> c.ids.tolist()
    [0, 1, 2]

    All three nodes share color 0. Node 0 wins the tie (smallest id):

    >>> color_clash_tie(c, 0)
    True

    Node 1 clashes with node 0 whose id is smaller, so it yields:

    >>> color_clash_tie(c, 1)
    False

    With a proper coloring no node clashes at all:

    >>> c.colors[:] = [0, 1, 0]
    >>> color_clash_tie(c, 0)
    False
    """
    me = config.colors[i]
    meid = config.ids[i]
    conflict = False
    for j in config.graph.neighbors(i):
        if config.colors[j] == me:
            if config.ids[j] < meid:
                return False
            conflict = True
    return conflict


color_clash_tie.uses_ids = True


@njit(fastmath=True)
def color_min(config, i):
    """Enable node `i` iff its own color is not the lowest one missing among its neighbors.

    More precisely, walk ``range(k)`` in order and stop at the first color
    `c` absent from the neighbor histogram; the node is enabled iff
    ``c`` differs from its current color.

    Parameters
    ----------
    config: :class:`~tinto.configs.Config`
        Coloring configuration.
    i: :class:`int`
        Node index.

    Returns
    -------
    :class:`bool`
        ``True`` iff there exists a color strictly lower than the first
        one missing from the neighbor histogram that `i` currently
        holds, or equivalently iff `i` is not already using that minimum
        available color.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> Config = make()
    >>> c = Config(line(3))

    With ``colors = [0, 0, 0]``, node 1 has two neighbors colored 0.
    The smallest missing color is 1, which differs from ``me = 0``:

    >>> color_min(c, 1)
    True

    Swap node 1 to color 1 ; now it holds the smallest missing color:

    >>> c.colors[:] = [0, 1, 0]
    >>> c.check_all()
    >>> color_min(c, 1)
    False

    On a complete graph K4 with ``k = 3`` colors, every color is
    represented in the neighborhood of node 0, so no color is
    "missing" at all:

    >>> import numpy as np
    >>> from tinto.graphs import Graph
    >>> ptr = np.array([0, 3, 6, 9, 12], dtype=np.int32)
    >>> ind = np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], dtype=np.int32)
    >>> g = Graph(ptr, ind)
    >>> c = Config(g, k=3)
    >>> c.colors[:] = [0, 1, 2, 0]
    >>> c.check_all()
    >>> color_min(c, 0)
    False
    """
    me = config.colors[i]
    counts = config.count(i)
    for c in range(config.k):
        if counts[c] == 0:
            return c != me
    return False


color_min.uses_ids = False


@njit(fastmath=True)
def color_min_tie(config, i):
    """Enable node `i` under :func:`color_min` with a tie-break on :attr:`~tinto.graphs.Graph.ids`.

    Same rule as :func:`color_min`, except that when a neighbor sharing
    the color of `i` has a smaller :attr:`~tinto.graphs.Graph.ids`, `i`
    yields priority and is disabled.

    Parameters
    ----------
    config: :class:`~tinto.configs.Config`
        Coloring configuration.
    i: :class:`int`
        Node index.

    Returns
    -------
    :class:`bool`
        ``True`` iff `i` satisfies :func:`color_min` and owns the
        smallest :attr:`~tinto.graphs.Graph.ids` among same-colored
        neighbors.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> import numpy as np
    >>> Config = make()
    >>> c = Config(line(3))
    >>> c.ids = np.arange(3)

    With uniform color 0, node 0 wins the tie and is enabled:

    >>> color_min_tie(c, 0)
    True

    Node 1 clashes with node 0 whose id is smaller, and yields:

    >>> color_min_tie(c, 1)
    False

    Holding already the minimum missing color makes the node disabled,
    whatever its id:

    >>> c.colors[:] = [0, 1, 0]
    >>> c.check_all()
    >>> color_min_tie(c, 1)
    False

    And on a saturated neighborhood where no color is missing at all,
    the node is also disabled:

    >>> import numpy as np
    >>> from tinto.graphs import Graph
    >>> ptr = np.array([0, 3, 6, 9, 12], dtype=np.int32)
    >>> ind = np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], dtype=np.int32)
    >>> g = Graph(ptr, ind)
    >>> c = Config(g, k=3)
    >>> c.colors[:] = [0, 1, 2, 0]
    >>> c.check_all()
    >>> color_min_tie(c, 0)
    False
    """
    me = config.colors[i]
    counts = config.count(i)
    for c in range(config.k):
        if counts[c] == 0:
            if c != me:
                for j in config.graph.neighbors(i):
                    if config.colors[j] == me and config.ids[j] < config.ids[i]:
                        return False
                return True
            return False
    return False


color_min_tie.uses_ids = True
