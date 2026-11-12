"""Exact enumeration of the peak statistic on small graphs.

The third source of the companion checks, next to the closed forms of
:mod:`tinto.theory` and the Monte Carlo of :mod:`tinto.oracle`. Everything
here is exact integer counting, no sampling: all ``n!`` labelings, all
``k^n`` colorings, all labeled trees or all graphs on ``n`` nodes. It is
meant for ``n <= 7``.

The colorings are enumerated up to a permutation of the colors, which leaves
the peak count unchanged: one restricted growth string per class, weighted by
the size of the class.

Independence: this module imports from tinto only :mod:`tinto.graphs`, and
counts the peaks with its own code (see ``tests/test_independence.py``).
"""

import itertools
from fractions import Fraction
from functools import lru_cache
from math import factorial

import numpy as np
from numba import njit

from tinto.graphs import edges_to_graph


def colorings(n, k):
    """Colorings of `n` nodes with `k` colors, up to a permutation of the colors.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    k: :class:`int`
        Number of colors.

    Yields
    ------
    :class:`tuple`
        ``(coloring, weight)``: the coloring uses the colors in order of first
        appearance (a restricted growth string), and `weight` is the number of
        colorings in ``range(k)^n`` equivalent to it, ``k (k-1) ... (k-m+1)``
        for `m` colors used.

    Examples
    --------
    >>> list(colorings(3, 2))
    [((0, 0, 0), 2), ((0, 0, 1), 2), ((0, 1, 0), 2), ((0, 1, 1), 2)]
    >>> sum(w for _, w in colorings(4, 3)) == 3**4
    True
    >>> list(colorings(0, 2))
    []
    """
    col = [0] * n

    def rec(i, m):
        if i == n:
            w = 1
            for j in range(m):
                w *= k - j
            yield tuple(col), w
            return
        for c in range(min(m + 1, k)):
            col[i] = c
            yield from rec(i + 1, max(m, c + 1))

    if n > 0:
        yield from rec(1, 1)


@njit
def _edge_list(ptr, ind):
    """Edges ``(i, j)``, ``i < j``, of an undirected CSR graph."""
    n = len(ptr) - 1
    edges = np.empty((len(ind) // 2, 2), dtype=np.int64)
    e = 0
    for i in range(n):
        for t in range(ptr[i], ptr[i + 1]):
            if i < ind[t]:
                edges[e, 0], edges[e, 1] = i, ind[t]
                e += 1
    return edges[:e]


@njit
def _peaks_histogram(n, same):
    """Histogram of the peak count over the ``n!`` labelings.

    `same` lists the edges whose ends share their color; the end with the
    smaller label is beaten. Labelings are enumerated by Heap's algorithm.
    """
    hist = np.zeros(n + 1, dtype=np.int64)
    lab = np.arange(n)
    c = np.zeros(n, dtype=np.int64)
    beaten = np.zeros(n, dtype=np.bool_)
    while True:
        beaten[:] = False
        for e in range(same.shape[0]):
            u, v = same[e, 0], same[e, 1]
            if lab[u] < lab[v]:
                beaten[u] = True
            else:
                beaten[v] = True
        hist[n - beaten.sum()] += 1
        # next permutation (Heap's algorithm, iterative)
        i = 1
        while i < n and c[i] >= i:
            c[i] = 0
            i += 1
        if i >= n:
            return hist
        if i % 2 == 0:
            lab[0], lab[i] = lab[i], lab[0]
        else:
            lab[c[i]], lab[i] = lab[i], lab[c[i]]
        c[i] += 1


@njit
def _peaks_fixed(n, same, labels):
    """Number of peaks for one labeling."""
    beaten = np.zeros(n, dtype=np.bool_)
    for e in range(same.shape[0]):
        u, v = same[e, 0], same[e, 1]
        if labels[u] < labels[v]:
            beaten[u] = True
        else:
            beaten[v] = True
    return n - beaten.sum()


def peaks_distribution(graph, k=1, split=None, labels=None):
    """Exact law of the peak count, by enumeration of labelings and colorings.

    Parameters
    ----------
    graph: :class:`~tinto.graphs.Graph`
        Undirected graph, ``n <= 7`` or so (``n!`` labelings).
    k: :class:`int`, default=1
        Number of colors.
    split: :class:`tuple` or ``None``, default=``None``
        A pair of nodes ``(u, v)``: return two histograms, for the colorings
        where `u` and `v` share their color and for the others.
    labels: array_like or ``None``, default=``None``
        A fixed labeling (pairwise distinct values): enumerate the colorings
        only. On a uniform random labeled graph, such as a uniform labeled
        tree, the labeling can be fixed to the node indices without changing
        the law.

    Returns
    -------
    :class:`~numpy.ndarray` of :class:`int64`, shape (n + 1,)
        ``N[j]`` is the number of (labeling, coloring) pairs with exactly
        `j` peaks, out of ``n! k^n`` (``k^n`` colorings with `labels`). With
        `split`, a pair of such arrays.

    Examples
    --------
    On the path with three nodes, monochrome, one peak in 4 labelings of 6:

    >>> from tinto.graphs import line
    >>> peaks_distribution(line(3)).tolist()
    [0, 4, 2, 0]

    A complete graph always has one peak per color class:

    >>> from tinto.graphs import edges_to_graph
    >>> K3 = edges_to_graph(3, np.array([[0, 1], [0, 2], [1, 2]]))
    >>> peaks_distribution(K3, 2).tolist()
    [0, 12, 36, 0]

    The path on two nodes, split by whether its ends share their color, and
    with the labeling fixed:

    >>> [h.tolist() for h in peaks_distribution(line(2), 2, split=(0, 1))]
    [[0, 4, 0], [0, 0, 4]]
    >>> peaks_distribution(line(3), 2, labels=[0, 1, 2]).tolist()
    [0, 2, 4, 2]
    >>> peaks_distribution(line(3), 2, labels=[2, 1, 0]).tolist()
    [0, 2, 4, 2]
    """
    n = graph.n
    edges = _edge_list(graph.ptr, graph.ind)
    cols, weights = _coloring_table(n, k)
    use_fixed = labels is not None
    fixed = np.asarray(labels, dtype=np.float64) if use_fixed else np.zeros(n)
    u, v = split if split is not None else (0, 0)
    hist_same, hist_dist = _histograms(n, edges, cols, weights, fixed, use_fixed, u, v)
    return hist_same if split is None else (hist_same, hist_dist)


@lru_cache(maxsize=None)
def _coloring_table(n, k):
    """The output of :func:`colorings` as arrays: one row per class, and the weights."""
    table = list(colorings(n, k))
    cols = np.array([c for c, _ in table], dtype=np.int64).reshape(len(table), n)
    weights = np.array([w for _, w in table], dtype=np.int64)
    return cols, weights


@njit
def _histograms(n, edges, cols, weights, fixed, use_fixed, u, v):
    """Peak-count histograms over the colorings of `cols`, split on whether `u`
    and `v` share their color (with ``u = v``, everything goes to the first)."""
    hist_same = np.zeros(n + 1, dtype=np.int64)
    hist_dist = np.zeros(n + 1, dtype=np.int64)
    same = np.empty_like(edges)
    for c in range(cols.shape[0]):
        col = cols[c]
        m = 0
        for e in range(edges.shape[0]):
            if col[edges[e, 0]] == col[edges[e, 1]]:
                same[m, 0], same[m, 1] = edges[e, 0], edges[e, 1]
                m += 1
        target = hist_same if col[u] == col[v] else hist_dist
        if use_fixed:
            target[_peaks_fixed(n, same[:m], fixed)] += weights[c]
        else:
            target += weights[c] * _peaks_histogram(n, same[:m])
    return hist_same, hist_dist


def factorial_moments(hist):
    """Exact :math:`E S` and :math:`E S(S-1)` from a histogram of the peak count.

    Parameters
    ----------
    hist: :class:`~numpy.ndarray` of :class:`int`
        Counts, as returned by :func:`peaks_distribution`.

    Returns
    -------
    :class:`tuple` of :class:`~fractions.Fraction`

    Examples
    --------
    >>> factorial_moments(np.array([0, 4, 2, 0]))
    (Fraction(4, 3), Fraction(2, 3))
    """
    total = int(hist.sum())
    es = sum(Fraction(j * int(c), total) for j, c in enumerate(hist))
    ess = sum(Fraction(j * (j - 1) * int(c), total) for j, c in enumerate(hist))
    return es, ess


def all_trees(n):
    """All labeled trees on `n` nodes, by Prüfer decoding.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes, ``n >= 2`` (``n^(n-2)`` trees).

    Yields
    ------
    :class:`~tinto.graphs.Graph`

    Examples
    --------
    >>> trees = list(all_trees(4))
    >>> len(trees), {len(t.ind) // 2 for t in trees}
    (16, {3})
    """
    for seq in itertools.product(range(n), repeat=n - 2):
        degree = [1] * n
        for x in seq:
            degree[x] += 1
        edges = []
        for x in seq:
            leaf = min(i for i in range(n) if degree[i] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, v = (i for i in range(n) if degree[i] == 1)
        edges.append((u, v))
        yield edges_to_graph(n, np.array(edges, dtype=np.int64))


def all_graphs(n):
    """All labeled graphs on `n` nodes.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes (``2^(n(n-1)/2)`` graphs).

    Yields
    ------
    :class:`~tinto.graphs.Graph`

    Examples
    --------
    >>> sorted(len(g.ind) // 2 for g in all_graphs(3))
    [0, 1, 1, 1, 2, 2, 2, 3]
    """
    pairs = list(itertools.combinations(range(n), 2))
    for mask in range(1 << len(pairs)):
        edges = [pairs[i] for i in range(len(pairs)) if mask >> i & 1]
        yield edges_to_graph(n, np.array(edges, dtype=np.int64).reshape(-1, 2))


def law_size(n, k=1):
    """Number of (labeling, coloring) pairs, ``n! k^n``.

    Examples
    --------
    >>> law_size(4, 3)
    1944
    """
    return factorial(n) * k**n
