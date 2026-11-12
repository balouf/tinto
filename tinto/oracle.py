"""Peak oracle: samples of the stabilization time drawn without running the dynamics.

The reduction lemma of the exact stabilization-time paper reads as follows.
Run the conflict-guard rule (a node sharing its color with a neighbor
recolors itself with a color absent from its neighborhood) under the random
central daemon, from a uniform random coloring with `k` colors and a palette
of at least ``Delta + 1`` colors. Then the number of moves until
stabilization is ``T = n - S``, where `S` counts the *peaks* of a uniform
random labeling of the nodes: a node is a peak when its label exceeds the
labels of all its neighbors that share its initial color.

Sampling `T` therefore needs no dynamics. Draw the colors and the labels,
then count the nodes that are beaten by a same-colored neighbor with a
larger label. The samplers below return samples of ``T`` (the number of
beaten nodes), for a fixed graph or for a random graph drawn afresh at each
trial. The peak statistic is defined for every `k`; the palette condition
only matters for the dynamics, which the oracle does not run.

:func:`~tinto.schedulers.shuffled` gives the same law as the random central
daemon for this rule: a disabled node never becomes enabled again, so each
node moves at most once, at its first visit, exactly when a same-colored
neighbor is still unvisited.

The samplers run in parallel over a fixed number of chunks, each seeded with
``seed + chunk``, so the samples depend on the seed only, not on the number
of threads. Without a seed, the base seed is drawn from the generator that
:func:`~tinto.io.set_seed` controls.

This module belongs to the simulation side of tinto: it imports neither
:mod:`tinto.exact` nor :mod:`tinto.theory`, the two other sources of the
companion checks.
"""

import numpy as np
from numba import njit, prange

from tinto.graphs import bipartite_edges, cayley_edges, erdos_renyi_edges

CHUNKS = 64
"""Number of independently seeded chunks a sampling run is split into."""

_FIXED, _ERDOS_RENYI, _BIPARTITE, _CAYLEY = 0, 1, 2, 3
_NO_EDGES = np.zeros((0, 2), dtype=np.int64)


@njit
def count_peaks(graph, colors, labels):
    """Number of peaks of a colored and labeled graph.

    Parameters
    ----------
    graph: :class:`~tinto.graphs.Graph`
        Undirected graph (each edge stored in both directions).
    colors: :class:`~numpy.ndarray` of :class:`int`, shape (n,)
        Initial color of each node.
    labels: :class:`~numpy.ndarray`, shape (n,)
        Label of each node; labels are pairwise distinct.

    Returns
    -------
    :class:`int`
        Number of nodes whose label exceeds the labels of all their
        same-colored neighbors.

    Examples
    --------
    On a monochrome path ``0 - 1 - 2``, only the node with the largest label
    among its neighbors survives:

    >>> from tinto.graphs import line
    >>> count_peaks(line(3), np.zeros(3, dtype=np.int64), np.array([0.1, 0.5, 0.3]))
    1

    Two colors split the path into a single edge and an isolated node:

    >>> count_peaks(line(3), np.array([0, 0, 1]), np.array([0.1, 0.5, 0.3]))
    2
    """
    s = 0
    for i in range(graph.n):
        peak = True
        for j in graph.neighbors(i):
            if colors[j] == colors[i] and labels[j] > labels[i]:
                peak = False
                break
        s += peak
    return s


@njit
def _beaten(n, edges, k):
    """One sample of ``T``: draw colors and labels, count the beaten nodes."""
    colors = np.random.randint(0, k, n)
    labels = np.random.random(n)
    beaten = np.zeros(n, dtype=np.bool_)
    for e in range(edges.shape[0]):
        u, v = edges[e, 0], edges[e, 1]
        if colors[u] == colors[v]:
            if labels[u] < labels[v]:
                beaten[u] = True
            else:
                beaten[v] = True
    return beaten.sum()


@njit(parallel=True)
def _sample(family, n, n1, n2, p, fixed, k, trials, seed):
    out = np.empty(trials, dtype=np.int64)
    for c in prange(CHUNKS):
        np.random.seed(seed + c)
        for t in range(c * trials // CHUNKS, (c + 1) * trials // CHUNKS):
            if family == _ERDOS_RENYI:
                edges = erdos_renyi_edges(n, p)
            elif family == _BIPARTITE:
                edges = bipartite_edges(n1, n2, p)
            elif family == _CAYLEY:
                edges = cayley_edges(n)
            else:
                edges = fixed
            out[t] = _beaten(n, edges, k)
    return out


@njit
def _draw_seeds():
    return np.random.randint(0, 2**30), np.random.randint(0, 2**30)


@njit
def _reseed(seed):
    np.random.seed(seed)


def _run(family, n, n1, n2, p, fixed, k, trials, seed):
    """Seed handling around :func:`_sample`.

    The chunks reseed the generator of every thread they run on, the calling
    thread included. Two numbers are drawn first: the base seed (unless
    `seed` is given) and a seed with which the calling thread resumes, so
    that a sequence of calls stays reproducible under :func:`~tinto.io.set_seed`.
    """
    base, resume = _draw_seeds()
    out = _sample(
        family, n, n1, n2, p, fixed, k, trials, base if seed is None else seed
    )
    _reseed(resume)
    return out


@njit
def _graph_edges(graph):
    """Edge list ``(i, j)``, ``i < j``, of an undirected graph."""
    edges = np.empty((len(graph.ind) // 2, 2), dtype=np.int64)
    e = 0
    for i in range(graph.n):
        for j in graph.neighbors(i):
            if i < j:
                edges[e, 0], edges[e, 1] = i, j
                e += 1
    return edges[:e].copy()


def sample_graph(graph, k=1, trials=1000, seed=None):
    """Sample the stabilization time on a fixed graph.

    Parameters
    ----------
    graph: :class:`~tinto.graphs.Graph`
        Undirected graph.
    k: :class:`int`, default=1
        Number of colors of the uniform random initial coloring.
    trials: :class:`int`, default=1000
        Number of samples.
    seed: :class:`int` or ``None``, default=``None``
        Base seed of the chunks; ``None`` draws it from the generator that
        :func:`~tinto.io.set_seed` controls.

    Returns
    -------
    :class:`~numpy.ndarray` of :class:`int64`, shape (trials,)
        Samples of ``T = n - S``.

    Examples
    --------
    On a monochrome complete graph, only the largest label is a peak:

    >>> from tinto.graphs import adja_to_graph
    >>> K5 = adja_to_graph(~np.eye(5, dtype=np.bool_))
    >>> sample_graph(K5, trials=10, seed=0).tolist()
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    The samples depend on the seed only:

    >>> from tinto.graphs import line
    >>> a = sample_graph(line(20), k=3, trials=200, seed=7)
    >>> bool((a == sample_graph(line(20), k=3, trials=200, seed=7)).all())
    True
    """
    return _run(_FIXED, graph.n, 0, 0, 0.0, _graph_edges(graph), k, trials, seed)


def sample_erdos_renyi(n, p, k=1, trials=1000, seed=None):
    """Sample the stabilization time on :math:`G(n, p)`, a fresh graph per trial.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    p: :class:`float`
        Edge probability.
    k: :class:`int`, default=1
        Number of colors of the uniform random initial coloring.
    trials: :class:`int`, default=1000
        Number of samples.
    seed: :class:`int` or ``None``, default=``None``
        Base seed (see :func:`sample_graph`).

    Returns
    -------
    :class:`~numpy.ndarray` of :class:`int64`, shape (trials,)
        Samples of ``T = n - S`` over the joint law of the graph, the
        coloring and the labeling.

    Examples
    --------
    With ``p = 1`` the graph is complete:

    >>> sample_erdos_renyi(6, 1.0, trials=5, seed=0).tolist()
    [5, 5, 5, 5, 5]
    """
    return _run(_ERDOS_RENYI, n, 0, 0, p, _NO_EDGES, k, trials, seed)


def sample_bipartite(n1, n2, p, k=1, trials=1000, seed=None):
    """Sample the stabilization time on :math:`G(n_1, n_2, p)`, a fresh graph per trial.

    Parameters
    ----------
    n1: :class:`int`
        Size of the first side.
    n2: :class:`int`
        Size of the second side.
    p: :class:`float`
        Probability of an edge across the sides.
    k: :class:`int`, default=1
        Number of colors of the uniform random initial coloring.
    trials: :class:`int`, default=1000
        Number of samples.
    seed: :class:`int` or ``None``, default=``None``
        Base seed (see :func:`sample_graph`).

    Returns
    -------
    :class:`~numpy.ndarray` of :class:`int64`, shape (trials,)
        Samples of ``T = n1 + n2 - S``.

    Examples
    --------
    On the monochrome star :math:`K_{1, 3}`, the three leaves are beaten
    when the center has the largest label; otherwise the center is beaten,
    with the leaves below it, so ``T`` takes the values 1, 2 and 3:

    >>> sorted(set(sample_bipartite(1, 3, 1.0, trials=50, seed=0).tolist()))
    [1, 2, 3]
    """
    return _run(_BIPARTITE, n1 + n2, n1, n2, p, _NO_EDGES, k, trials, seed)


def sample_cayley(n, k=1, trials=1000, seed=None):
    """Sample the stabilization time on a uniform labeled tree, a fresh tree per trial.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    k: :class:`int`, default=1
        Number of colors of the uniform random initial coloring.
    trials: :class:`int`, default=1000
        Number of samples.
    seed: :class:`int` or ``None``, default=``None``
        Base seed (see :func:`sample_graph`).

    Returns
    -------
    :class:`~numpy.ndarray` of :class:`int64`, shape (trials,)
        Samples of ``T = n - S``.

    Examples
    --------
    A tree on two nodes is a single edge:

    >>> sample_cayley(2, trials=5, seed=0).tolist()
    [1, 1, 1, 1, 1]
    """
    return _run(_CAYLEY, n, 0, 0, 0.0, _NO_EDGES, k, trials, seed)
