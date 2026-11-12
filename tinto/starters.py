"""Factories producing initial :class:`~tinto.configs.Config` instances.

A *starter* is a :mod:`numba` jitclass exposing ``draw() -> Config`` that
returns a freshly initialized :class:`~tinto.configs.Config`. Starters are
consumed by :func:`tinto.runners.mc`, which calls :meth:`draw` once per
replica to seed the Monte-Carlo samples.

:func:`from_graph` covers both regimes from a fixed user-supplied graph:
deterministic uniform coloring (``color`` given as an integer, including
``0`` for the all-zeros default) and stochastic coloring (``color=None``).

:func:`from_cayley`, :func:`from_erdos_renyi` and :func:`from_bipartite`
follow the same coloring contract but *generate* a fresh random graph on
every :meth:`draw` call, so that :func:`tinto.runners.mc` samples over a
distribution of graphs (one Cayley tree, one :math:`G(n, p)` or one
:math:`G(n_1, n_2, p)` per replica) rather than over a single fixed
topology. They take the generator's parameters at construction time,
leaving ``(Config, color)`` as the sole compilation key — exactly as the
graph is a construction-time argument of :func:`from_graph`.

Every starter also takes, at construction time, the palette size ``k`` and
the number of initial colors ``k_init`` (both 0 by default). In the
stochastic regime, the initial colors are drawn among the first ``k_init``
colors (all of them when ``k_init`` is 0). When ``k`` is 0, the palette has
``max(k_init, Delta + 1)`` colors, so that a free color always exists: a
start with ``k_init < Delta + 1`` colors, as in the exact stabilization-time
paper, keeps a palette of ``Delta + 1``.
"""

from functools import lru_cache

from numba import bool, float64, int64, njit
from numba.experimental import jitclass

from tinto.graphs import Graph, bipartite, cayley, erdos_renyi


@njit
def _palette(graph, k, k_init):
    """Palette size: `k` when given, else ``max(k_init, Delta + 1)``.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> _palette(line(5), 0, 0), _palette(line(5), 0, 2), _palette(line(5), 0, 5), _palette(line(5), 4, 2)
    (3, 3, 5, 4)
    """
    if k > 0:
        return k
    return max(k_init, graph.max_degree + 1)


def from_graph(Config, color=None):
    """Build a starter class drawing configs over a fixed graph.

    The returned jitclass takes a :class:`~tinto.graphs.Graph` at
    construction time; each :meth:`draw` call allocates a fresh
    :class:`~tinto.configs.Config` and seeds it according to `color`:

    - ``color=None`` (default): the config is randomized via
      :meth:`~tinto.configs.Config.color_randomize`.
    - ``color=0``: the bare :class:`~tinto.configs.Config` constructor
      is used, which already initializes every node to color 0.
    - any other ``int``: the config is forced to that uniform color via
      :meth:`~tinto.configs.Config.color_uniform`.

    The ``None`` sentinel for the stochastic regime follows the usual
    Python idiom (``np.random.seed(None)``, ``open(..., encoding=None)``,
    ...) and avoids overloading a positive integer to mean "no fixed
    color".

    Parameters
    ----------
    Config: type
        Jitclass returned by :func:`tinto.configs.make`.
    color: :class:`int` or ``None``, default=``None``
        Initial color. ``None`` randomizes every node; any integer in
        ``range(k)`` forces a uniform coloring at that value.

    Returns
    -------
    type
        A jitclass ``FromGraphStarter`` with constructor ``(graph,
        k=0, k_init=0)`` and method ``draw() -> Config``.

    Examples
    --------
    Stochastic regime (default, ``color=None``):

    >>> from tinto.io import set_seed
    >>> from tinto.graphs import line
    >>> from tinto.configs import make as make_config
    >>> set_seed(0)
    >>> Config = make_config()
    >>> Starter = from_graph(Config)
    >>> c = Starter(line(8)).draw()
    >>> bool(((c.colors >= 0) & (c.colors < c.k)).all())
    True
    >>> c.total == int(c.enabled.sum())
    True

    Two initial colors on a path, whose palette stays ``Delta + 1 = 3``:

    >>> c = Starter(line(8), k_init=2).draw()
    >>> c.k, bool((c.colors < 2).all())
    (3, True)

    Uniform regime at color 0:

    >>> c = from_graph(Config, color=0)(line(4)).draw()
    >>> c.colors.tolist()
    [0, 0, 0, 0]

    A non-zero uniform color is equally accepted:

    >>> c = from_graph(Config, color=1)(line(4)).draw()
    >>> c.colors.tolist()
    [1, 1, 1, 1]

    Repeated calls with the same ``(Config, color)`` hit the cache:

    >>> from_graph(Config) is from_graph(Config)
    True
    >>> from_graph(Config, color=None) is from_graph(Config)
    True
    >>> from_graph(Config, color=0) is from_graph(Config, color=1)
    False

    Notes
    -----
    Two calls with the same ``(Config, color)`` return the same
    compiled starter. `Config` is keyed by identity and `color` by
    value.
    """
    return _from_graph(Config, color)


@lru_cache(maxsize=None)
def _from_graph(Config, color):
    randomize = color is None
    init_color = 0 if randomize else color
    starter_uses_ids = getattr(Config, "uses_ids", True)

    @jitclass
    class FromGraphStarter:
        graph: Graph
        k: int64  # type: ignore
        k_init: int64  # type: ignore
        uses_ids: bool  # type: ignore

        def __init__(self, graph, k=0, k_init=0):
            self.graph = graph
            self.k = k
            self.k_init = k_init
            self.uses_ids = starter_uses_ids

        def draw(self):
            config = Config(self.graph, _palette(self.graph, self.k, self.k_init))
            if randomize:
                config.color_randomize(self.k_init)
            elif init_color > 0:
                config.color_uniform(color=init_color)
            return config

    return FromGraphStarter


def from_cayley(Config, color=None):
    """Build a starter drawing configs over a fresh random Cayley tree.

    The returned jitclass takes the tree size ``n`` at construction time;
    each :meth:`draw` call generates a *new* random Cayley tree via
    :func:`tinto.graphs.cayley`, allocates a fresh
    :class:`~tinto.configs.Config` over it, and seeds it according to
    `color` with the exact same contract as :func:`from_graph`:

    - ``color=None`` (default): the config is randomized via
      :meth:`~tinto.configs.Config.color_randomize`.
    - ``color=0``: the bare :class:`~tinto.configs.Config` constructor is
      used, which already initializes every node to color 0.
    - any other ``int``: the config is forced to that uniform color via
      :meth:`~tinto.configs.Config.color_uniform`.

    Because a fresh graph is drawn per replica, :func:`tinto.runners.mc`
    samples the dynamics over the distribution of Cayley trees on ``n``
    nodes rather than over a single fixed tree.

    Parameters
    ----------
    Config: type
        Jitclass returned by :func:`tinto.configs.make`.
    color: :class:`int` or ``None``, default=``None``
        Initial color. ``None`` randomizes every node; any integer in
        ``range(k)`` forces a uniform coloring at that value.

    Returns
    -------
    type
        A jitclass ``FromCayleyStarter`` with constructor ``(n=100,
        k=0, k_init=0)`` and method ``draw() -> Config``.

    Examples
    --------
    Stochastic regime (default, ``color=None``); a fresh tree on ``n``
    nodes is drawn each call:

    >>> from tinto.io import set_seed
    >>> from tinto.configs import make as make_config
    >>> set_seed(0)
    >>> Config = make_config()
    >>> c = from_cayley(Config)(8).draw()
    >>> c.graph.n
    8
    >>> bool(((c.colors >= 0) & (c.colors < c.k)).all())
    True
    >>> c.total == int(c.enabled.sum())
    True

    Uniform regimes behave like :func:`from_graph`, independently of the
    drawn topology:

    >>> from_cayley(Config, color=0)(4).draw().colors.tolist()
    [0, 0, 0, 0]
    >>> from_cayley(Config, color=1)(4).draw().colors.tolist()
    [1, 1, 1, 1]

    Repeated calls with the same ``(Config, color)`` hit the cache:

    >>> from_cayley(Config) is from_cayley(Config)
    True
    >>> from_cayley(Config, color=0) is from_cayley(Config, color=1)
    False

    Notes
    -----
    Two calls with the same ``(Config, color)`` return the same compiled
    starter. `Config` is keyed by identity and `color` by value.
    """
    return _from_cayley(Config, color)


@lru_cache(maxsize=None)
def _from_cayley(Config, color):
    randomize = color is None
    init_color = 0 if randomize else color
    starter_uses_ids = getattr(Config, "uses_ids", True)

    @jitclass
    class FromCayleyStarter:
        n: int64  # type: ignore
        k: int64  # type: ignore
        k_init: int64  # type: ignore
        uses_ids: bool  # type: ignore

        def __init__(self, n=100, k=0, k_init=0):
            self.n = n
            self.k = k
            self.k_init = k_init
            self.uses_ids = starter_uses_ids

        def draw(self):
            graph = cayley(self.n)
            config = Config(graph, _palette(graph, self.k, self.k_init))
            if randomize:
                config.color_randomize(self.k_init)
            elif init_color > 0:
                config.color_uniform(color=init_color)
            return config

    return FromCayleyStarter


def from_erdos_renyi(Config, color=None):
    """Build a starter drawing configs over a fresh random :math:`G(n, p)` graph.

    The returned jitclass takes the node count ``n`` and edge probability
    ``p`` at construction time; each :meth:`draw` call generates a *new*
    random Erdős–Rényi graph via :func:`tinto.graphs.erdos_renyi`,
    allocates a fresh :class:`~tinto.configs.Config` over it, and seeds it
    according to `color` with the exact same contract as
    :func:`from_graph`:

    - ``color=None`` (default): the config is randomized via
      :meth:`~tinto.configs.Config.color_randomize`.
    - ``color=0``: the bare :class:`~tinto.configs.Config` constructor is
      used, which already initializes every node to color 0.
    - any other ``int``: the config is forced to that uniform color via
      :meth:`~tinto.configs.Config.color_uniform`.

    Because a fresh graph is drawn per replica, :func:`tinto.runners.mc`
    samples the dynamics over the :math:`G(n, p)` ensemble rather than
    over a single fixed graph.

    Parameters
    ----------
    Config: type
        Jitclass returned by :func:`tinto.configs.make`.
    color: :class:`int` or ``None``, default=``None``
        Initial color. ``None`` randomizes every node; any integer in
        ``range(k)`` forces a uniform coloring at that value.

    Returns
    -------
    type
        A jitclass ``FromErdosRenyiStarter`` with constructor ``(n=100,
        p=0.5, k=0, k_init=0)`` and method ``draw() -> Config``.

    Examples
    --------
    Stochastic regime (default, ``color=None``); a fresh :math:`G(n, p)`
    graph is drawn each call:

    >>> from tinto.io import set_seed
    >>> from tinto.configs import make as make_config
    >>> set_seed(0)
    >>> Config = make_config()
    >>> c = from_erdos_renyi(Config)(8, p=0.5).draw()
    >>> c.graph.n
    8
    >>> bool(((c.colors >= 0) & (c.colors < c.k)).all())
    True
    >>> c.total == int(c.enabled.sum())
    True

    Uniform regimes behave like :func:`from_graph`, independently of the
    drawn topology (``p=1.0`` guarantees enough colors for ``color=1``):

    >>> from_erdos_renyi(Config, color=0)(4, p=0.5).draw().colors.tolist()
    [0, 0, 0, 0]
    >>> from_erdos_renyi(Config, color=1)(4, p=1.0).draw().colors.tolist()
    [1, 1, 1, 1]

    Repeated calls with the same ``(Config, color)`` hit the cache:

    >>> from_erdos_renyi(Config) is from_erdos_renyi(Config)
    True
    >>> from_erdos_renyi(Config, color=0) is from_erdos_renyi(Config, color=1)
    False

    Notes
    -----
    Two calls with the same ``(Config, color)`` return the same compiled
    starter. `Config` is keyed by identity and `color` by value.
    """
    return _from_erdos_renyi(Config, color)


@lru_cache(maxsize=None)
def _from_erdos_renyi(Config, color):
    randomize = color is None
    init_color = 0 if randomize else color
    starter_uses_ids = getattr(Config, "uses_ids", True)

    @jitclass
    class FromErdosRenyiStarter:
        n: int64  # type: ignore
        p: float64  # type: ignore
        k: int64  # type: ignore
        k_init: int64  # type: ignore
        uses_ids: bool  # type: ignore

        def __init__(self, n=100, p=0.5, k=0, k_init=0):
            self.n = n
            self.p = p
            self.k = k
            self.k_init = k_init
            self.uses_ids = starter_uses_ids

        def draw(self):
            graph = erdos_renyi(self.n, self.p)
            config = Config(graph, _palette(graph, self.k, self.k_init))
            if randomize:
                config.color_randomize(self.k_init)
            elif init_color > 0:
                config.color_uniform(color=init_color)
            return config

    return FromErdosRenyiStarter


def from_bipartite(Config, color=None):
    """Build a starter drawing configs over a fresh random bipartite :math:`G(n_1, n_2, p)` graph.

    Same contract as :func:`from_erdos_renyi`, with the generator
    :func:`tinto.graphs.bipartite`: the constructor takes the side sizes
    ``n1``, ``n2`` and the edge probability ``p``, and each :meth:`draw`
    call generates a *new* graph, allocates a fresh
    :class:`~tinto.configs.Config` over it and seeds it according to
    `color` (``None`` randomizes, ``0`` keeps the bare constructor, any
    other ``int`` forces a uniform coloring).

    Parameters
    ----------
    Config: type
        Jitclass returned by :func:`tinto.configs.make`.
    color: :class:`int` or ``None``, default=``None``
        Initial color. ``None`` randomizes every node; any integer in
        ``range(k)`` forces a uniform coloring at that value.

    Returns
    -------
    type
        A jitclass ``FromBipartiteStarter`` with constructor ``(n1=50,
        n2=50, p=0.5, k=0, k_init=0)`` and method ``draw() -> Config``.

    Examples
    --------

    >>> from tinto.io import set_seed
    >>> from tinto.configs import make as make_config
    >>> set_seed(0)
    >>> Config = make_config()
    >>> c = from_bipartite(Config)(3, 5, p=0.5).draw()
    >>> c.graph.n
    8
    >>> c.total == int(c.enabled.sum())
    True
    >>> from_bipartite(Config, color=0)(2, 2, p=1.0).draw().colors.tolist()
    [0, 0, 0, 0]
    >>> from_bipartite(Config, color=1)(2, 2, p=1.0).draw().colors.tolist()
    [1, 1, 1, 1]
    >>> from_bipartite(Config) is from_bipartite(Config)
    True
    """
    return _from_bipartite(Config, color)


@lru_cache(maxsize=None)
def _from_bipartite(Config, color):
    randomize = color is None
    init_color = 0 if randomize else color
    starter_uses_ids = getattr(Config, "uses_ids", True)

    @jitclass
    class FromBipartiteStarter:
        n1: int64  # type: ignore
        n2: int64  # type: ignore
        p: float64  # type: ignore
        k: int64  # type: ignore
        k_init: int64  # type: ignore
        uses_ids: bool  # type: ignore

        def __init__(self, n1=50, n2=50, p=0.5, k=0, k_init=0):
            self.n1 = n1
            self.n2 = n2
            self.p = p
            self.k = k
            self.k_init = k_init
            self.uses_ids = starter_uses_ids

        def draw(self):
            graph = bipartite(self.n1, self.n2, self.p)
            config = Config(graph, _palette(graph, self.k, self.k_init))
            if randomize:
                config.color_randomize(self.k_init)
            elif init_color > 0:
                config.color_uniform(color=init_color)
            return config

    return FromBipartiteStarter
