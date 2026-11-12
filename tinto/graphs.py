"""Compressed sparse row (CSR) graph representation.

This module defines :class:`Graph`, a lightweight static directed graph backed
by two integer arrays in CSR format and compiled with :mod:`numba` so that it
can be manipulated inside other JIT-compiled routines. The module also exposes
two builders, :func:`line` and :func:`ring`, that return common benchmark
topologies.
"""

from numba.experimental import jitclass
from numba import int64, int32, njit
import numpy as np


@jitclass
class Graph:
    """Static directed graph stored in compressed sparse row (CSR) format.

    The out-neighbors of every node are stored consecutively in :attr:`ind`,
    and :attr:`ptr` records where each node's neighbor list begins. This is
    the layout used by :class:`scipy.sparse.csr_array`, which makes
    interoperability cheap: see :func:`adja_to_graph`.

    Parameters
    ----------
    ptr: :class:`~numpy.ndarray` of :class:`int32`, shape (n + 1,)
        Row pointer array. ``ptr[i]`` is the position in :attr:`ind` where
        the neighbors of node `i` start, and ``ptr[i + 1]`` is where they
        end. Must be non-decreasing with ``ptr[0] == 0``.
    ind: :class:`~numpy.ndarray` of :class:`int32`, shape (nnz,)
        Column indices. ``ind[ptr[i]:ptr[i + 1]]`` lists the neighbors of
        node `i`.

    Attributes
    ----------
    n: :class:`int64`
        Number of nodes, equal to ``len(ptr) - 1``.
    ptr: :class:`~numpy.ndarray` of :class:`int32`
        See the corresponding parameter.
    ind: :class:`~numpy.ndarray` of :class:`int32`
        See the corresponding parameter.
    max_degree: :class:`int64`
        Maximum out-degree of any node, computed from `ptr` and used by :class:`~tinto.configs.Config` to set the number of colors.

    Notes
    -----
    Because the class is decorated with :func:`numba.jitclass`, `ptr` and
    `ind` must be :mod:`numpy` arrays with the exact dtypes declared above;
    Python lists are not accepted.

    Examples
    --------
    Build a small directed triangle graph directly from its CSR arrays:

    >>> ptr = np.array([0, 2, 4, 6], dtype=np.int32)
    >>> ind = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
    >>> g = Graph(ptr, ind)
    >>> g.n
    3
    >>> g.neighbors(0).tolist()
    [1, 2]
    """

    n: int64  # type: ignore
    ptr: int32[:]  # type: ignore
    ind: int32[:]  # type: ignore
    max_degree: int64  # type: ignore

    def __init__(self, ptr, ind):
        self.n = len(ptr) - 1
        self.ptr = ptr
        self.ind = ind
        res = 0
        past = 0
        for i in range(self.n):
            deg = self.ptr[i + 1] - past
            res = max(res, deg)
            past += deg
        self.max_degree = res

    def to_adja(self):
        """
        Build the adjacency matrix of the graph.

        Returns
        -------
        :class:`~numpy.ndarray`
        """
        adja = np.zeros((self.n, self.n), dtype=np.bool_)
        for i in range(self.n):
            for j in self.neighbors(i):
                adja[i, j] = True
        return adja

    def neighbors(self, i):
        """Return the out-neighbors of node `i`.

        Parameters
        ----------
        i: :class:`int`
            Index of the node whose neighbors are queried, with
            ``0 <= i < n``.

        Returns
        -------
        :class:`~numpy.ndarray` of :class:`int32`
            View into :attr:`ind` containing the indices of the nodes
            reachable from `i` in a single edge.

        Examples
        --------

        >>> ptr = np.array([0, 2, 3, 4], dtype=np.int32)
        >>> ind = np.array([1, 2, 0, 0], dtype=np.int32)
        >>> g = Graph(ptr, ind)
        >>> g.neighbors(0).tolist()
        [1, 2]
        >>> g.neighbors(2).tolist()
        [0]
        """
        return self.ind[self.ptr[i] : self.ptr[i + 1]]


@njit
def adja_to_graph(adja):
    """Build a :class:`Graph` from a dense adjacency matrix.

    Non-zero entries are read in row-major order with :func:`numpy.nonzero`,
    which already groups them by row, so the column indices are exactly the
    CSR :attr:`~tinto.graphs.Graph.ind` array and :attr:`~tinto.graphs.Graph.ptr`
    is the cumulative count of entries per row. Any dtype works: a non-zero
    entry denotes an edge.

    Parameters
    ----------
    adja: :class:`~numpy.ndarray`, shape (n, n)
        Square adjacency matrix. A non-zero entry at row `i` and column
        `j` creates an edge from `i` to `j`.

    Returns
    -------
    :class:`~tinto.graphs.Graph`
        Equivalent graph in CSR form.

    Examples
    --------

    >>> adja = np.array([[0, 1, 1],
    ...                  [1, 0, 0],
    ...                  [1, 0, 0]], dtype=np.int64)
    >>> g = adja_to_graph(adja)
    >>> g.n
    3
    >>> g.neighbors(0).tolist()
    [1, 2]
    >>> g.neighbors(1).tolist()
    [0]

    A node with no out-edges, including trailing ones, yields an empty
    neighbor list:

    >>> adja = np.array([[0, 1, 0],
    ...                  [0, 0, 0],
    ...                  [0, 0, 0]], dtype=np.int64)
    >>> g = adja_to_graph(adja)
    >>> g.neighbors(0).tolist()
    [1]
    >>> g.neighbors(2).tolist()
    []
    """
    n, _ = adja.shape
    rows, cols = np.nonzero(adja)
    ptr = np.zeros(n + 1, dtype=np.int32)
    for i in rows:
        ptr[i + 1] += 1
    ptr = np.cumsum(ptr).astype(np.int32)
    ind = cols.astype(np.int32)
    return Graph(ptr, ind)


@njit
def edges_to_graph(n, edges):
    """Build an undirected :class:`Graph` from an edge list.

    Each row ``(u, v)`` of `edges` is an undirected edge, listed once; the
    graph stores it in both directions. Neighbor lists come out sorted, as
    with :func:`adja_to_graph`. Self-loops and repeated edges are not
    checked.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    edges: :class:`~numpy.ndarray` of :class:`int`, shape (m, 2)
        Edge list, with node indices in ``range(n)``.

    Returns
    -------
    :class:`~tinto.graphs.Graph`
        Graph in CSR form, with ``2 m`` neighbor entries.

    Examples
    --------
    The bull, a triangle with two pendant nodes:

    >>> g = edges_to_graph(5, np.array([[0, 1], [1, 2], [0, 2], [1, 3], [2, 4]]))
    >>> [g.neighbors(i).tolist() for i in range(5)]
    [[1, 2], [0, 2, 3], [0, 1, 4], [1], [2]]

    Isolated nodes, trailing ones included, get empty neighbor lists:

    >>> g = edges_to_graph(4, np.array([[1, 0]]))
    >>> [g.neighbors(i).tolist() for i in range(4)]
    [[1], [0], [], []]
    """
    m = edges.shape[0]
    ptr = np.zeros(n + 1, dtype=np.int32)
    for e in range(m):
        ptr[edges[e, 0] + 1] += 1
        ptr[edges[e, 1] + 1] += 1
    ptr = np.cumsum(ptr).astype(np.int32)
    ind = np.empty(2 * m, dtype=np.int32)
    fill = ptr[:-1].copy()
    for e in range(m):
        u, v = edges[e, 0], edges[e, 1]
        ind[fill[u]] = v
        fill[u] += 1
        ind[fill[v]] = u
        fill[v] += 1
    for i in range(n):
        ind[ptr[i] : ptr[i + 1]] = np.sort(ind[ptr[i] : ptr[i + 1]])
    return Graph(ptr, ind)


@njit
def _skip(lq):
    """Gap to the next success of a Bernoulli sequence, ``log(1 - p) = lq``.

    The number of failures before the next success is geometric; drawing it
    directly is what makes the sparse generators run in O(n + m).
    """
    return 1 + int(np.log(1.0 - np.random.random()) / lq)


@njit
def _grow(edges, m):
    """Double the capacity of an edge buffer holding `m` edges."""
    bigger = np.empty((2 * edges.shape[0] + 1, 2), dtype=np.int64)
    bigger[:m] = edges[:m]
    return bigger


@njit
def adja_line(n):
    """Return the adjacency matrix of the undirected line on `n` nodes.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.

    Returns
    -------
    :class:`~numpy.ndarray` of :class:`bool`, shape (n, n)
        Logical or of the two off-diagonal identity matrices at offsets
        +1 and -1. Used as a building block for :func:`line` and
        :func:`ring`.

    Examples
    --------
    >>> adja_line(3).astype(int).tolist()
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    """
    return np.eye(n, k=1, dtype=np.bool_) | np.eye(n, k=-1, dtype=np.bool_)


@njit
def line(n=100):
    """Return a line graph with `n` nodes.

    Parameters
    ----------
    n: :class:`int`, default=100
        Number of nodes in the line graph.

    Returns
    -------
    :class:`~tinto.graphs.Graph`
        Line graph with `n` nodes.

    Examples
    --------
    >>> g = line(5)
    >>> g.n
    5
    >>> g.neighbors(0).tolist()
    [1]
    >>> g.neighbors(1).tolist()
    [0, 2]
    >>> g.neighbors(4).tolist()
    [3]
    """
    adja = adja_line(n)
    return adja_to_graph(adja)


@njit
def ring(n=100):
    """Return a ring graph with `n` nodes.

    Parameters
    ----------
    n: :class:`int`, default=100
        Number of nodes in the ring graph.

    Returns
    -------
    :class:`~tinto.graphs.Graph`
        Ring graph with `n` nodes.

    Examples
    --------

    >>> g = ring(5)
    >>> g.neighbors(0).tolist()
    [1, 4]
    >>> g.neighbors(4).tolist()
    [0, 3]
    """
    adja = adja_line(n)
    adja[0, -1] = True
    adja[-1, 0] = True
    return adja_to_graph(adja)


@njit
def cayley(n=100):
    """
    Draw a random Cayley graph with `n` nodes.
    Uses the bijection described in https://arxiv.org/abs/2107.09726

    Parameters
    ----------
    n: :class:`int`, default=100
        Number of nodes of the tree.

    Returns
    -------
    :class:`~tinto.graphs.Graph`
        A Cayley tree with n nodes.

    Examples
    --------

    >>> from tinto.io import set_seed
    >>> set_seed(42)
    >>> g = cayley(5)
    >>> g.to_adja()
    array([[False, False,  True, False, False],
           [False, False, False, False,  True],
           [ True, False, False, False,  True],
           [False, False, False, False,  True],
           [False,  True,  True,  True, False]])

    The degenerate single-node tree has no edges:

    >>> cayley(1).to_adja()
    array([[False]])
    """
    return edges_to_graph(n, cayley_edges(n))


@njit
def cayley_edges(n=100):
    """Draw the edges of a uniform random labeled tree on `n` nodes.

    Same draws as :func:`cayley`, which builds its graph from this list
    (the bijection of https://arxiv.org/abs/2107.09726).

    Parameters
    ----------
    n: :class:`int`, default=100
        Number of nodes of the tree.

    Returns
    -------
    :class:`~numpy.ndarray` of :class:`int64`, shape (n - 1, 2)
        The ``n - 1`` edges of the tree.

    Examples
    --------

    >>> from tinto.io import set_seed
    >>> set_seed(42)
    >>> sorted(sorted(e) for e in cayley_edges(5).tolist())
    [[0, 2], [1, 4], [2, 4], [3, 4]]
    >>> cayley_edges(1).shape
    (0, 2)
    """
    if n == 1:
        return np.zeros((0, 2), dtype=np.int64)
    vs = np.zeros(n - 1, dtype=np.int32)
    paths = [0]
    seen = set()
    for i in range(n - 1):
        v = np.random.randint(n)
        vs[i] = v
        if v in seen:
            paths.append(i)
        else:
            seen.add(v)
    paths.append(n - 1)
    paths = np.array(paths)
    leaves = np.array([i for i in range(n) if i not in seen])

    edges = np.empty((n - 1, 2), dtype=np.int64)
    e = 0
    for p, leave in enumerate(leaves):
        for k in range(paths[p], paths[p + 1] - 1):
            edges[e, 0], edges[e, 1] = vs[k], vs[k + 1]
            e += 1
        edges[e, 0], edges[e, 1] = leave, vs[paths[p + 1] - 1]
        e += 1
    return edges


@njit
def erdos_renyi(n=100, p=0.5):
    """Draw a random undirected :math:`G(n, p)` Erdős-Rényi graph.

    Every unordered pair of distinct nodes is connected independently with
    probability `p`. The edges come from :func:`erdos_renyi_edges`, in
    O(n + m) time.

    Parameters
    ----------
    n: :class:`int`, default=100
        Number of nodes in the erdos-renyi graph.
    p: :class:`float`, default=0.5
        Probability that an edge exists between two distinct nodes.

    Returns
    -------
    :class:`~tinto.graphs.Graph`
        A G(n, p) graph.

    Examples
    --------

    >>> from tinto.io import set_seed
    >>> set_seed(42)
    >>> g = erdos_renyi(5, p=.6)
    >>> g.to_adja()
    array([[False,  True, False, False,  True],
           [ True, False, False,  True,  True],
           [False, False, False, False,  True],
           [False,  True, False, False,  True],
           [ True,  True,  True,  True, False]])
    """
    return edges_to_graph(n, erdos_renyi_edges(n, p))


@njit
def erdos_renyi_edges(n=100, p=0.5):
    """Draw the edges of a random :math:`G(n, p)` graph in O(n + m) time.

    The pairs are visited in the order ``(1, 0), (2, 0), (2, 1), (3, 0)...``
    and the gap to the next edge is drawn from its geometric law instead of
    testing every pair (Batagelj and Brandes, *Efficient generation of large
    random networks*, Phys. Rev. E 71, 2005).

    Parameters
    ----------
    n: :class:`int`, default=100
        Number of nodes.
    p: :class:`float`, default=0.5
        Probability that an edge exists between two distinct nodes.

    Returns
    -------
    :class:`~numpy.ndarray` of :class:`int64`, shape (m, 2)
        One row ``(v, w)`` per edge, with ``w < v``.

    Examples
    --------

    >>> erdos_renyi_edges(4, 0.0).shape
    (0, 2)
    >>> erdos_renyi_edges(4, 1.0).tolist()
    [[1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2]]
    """
    if p <= 0.0 or n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    if p >= 1.0:
        edges = np.empty((n * (n - 1) // 2, 2), dtype=np.int64)
        e = 0
        for v in range(1, n):
            for w in range(v):
                edges[e, 0], edges[e, 1] = v, w
                e += 1
        return edges
    lq = np.log(1.0 - p)
    mean = p * n * (n - 1) / 2
    edges = np.empty((int(mean + 5 * np.sqrt(mean)) + 16, 2), dtype=np.int64)
    m = 0
    v, w = 1, -1
    while v < n:
        w += _skip(lq)
        while w >= v and v < n:
            w -= v
            v += 1
        if v < n:
            if m == edges.shape[0]:
                edges = _grow(edges, m)
            edges[m, 0], edges[m, 1] = v, w
            m += 1
    return edges[:m].copy()


@njit
def bipartite(n1=50, n2=50, p=0.5):
    """Draw a random bipartite :math:`G(n_1, n_2, p)` graph.

    Nodes ``0, ..., n1 - 1`` form the first side and ``n1, ..., n1 + n2 - 1``
    the second; each of the ``n1 n2`` pairs across the sides is an edge
    independently with probability `p`. The edges come from
    :func:`bipartite_edges`, in O(n + m) time.

    Parameters
    ----------
    n1: :class:`int`, default=50
        Size of the first side.
    n2: :class:`int`, default=50
        Size of the second side.
    p: :class:`float`, default=0.5
        Probability that an edge exists between two nodes of different
        sides.

    Returns
    -------
    :class:`~tinto.graphs.Graph`
        A :math:`G(n_1, n_2, p)` graph on ``n1 + n2`` nodes.

    Examples
    --------

    >>> g = bipartite(2, 3, 1.0)
    >>> [g.neighbors(i).tolist() for i in range(5)]
    [[2, 3, 4], [2, 3, 4], [0, 1], [0, 1], [0, 1]]
    """
    return edges_to_graph(n1 + n2, bipartite_edges(n1, n2, p))


@njit
def bipartite_edges(n1=50, n2=50, p=0.5):
    """Draw the edges of a random bipartite :math:`G(n_1, n_2, p)` graph in O(n + m) time.

    The ``n1 n2`` pairs across the sides are visited in row-major order and
    the gap to the next edge is drawn from its geometric law, as in
    :func:`erdos_renyi_edges`.

    Parameters
    ----------
    n1: :class:`int`, default=50
        Size of the first side (nodes ``0, ..., n1 - 1``).
    n2: :class:`int`, default=50
        Size of the second side (nodes ``n1, ..., n1 + n2 - 1``).
    p: :class:`float`, default=0.5
        Probability that an edge exists between two nodes of different
        sides.

    Returns
    -------
    :class:`~numpy.ndarray` of :class:`int64`, shape (m, 2)
        One row ``(u, v)`` per edge, with ``u < n1 <= v``.

    Examples
    --------

    >>> bipartite_edges(2, 2, 1.0).tolist()
    [[0, 2], [0, 3], [1, 2], [1, 3]]
    >>> bipartite_edges(2, 2, 0.0).shape
    (0, 2)
    """
    total = n1 * n2
    if p <= 0.0 or total == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if p >= 1.0:
        edges = np.empty((total, 2), dtype=np.int64)
        for idx in range(total):
            edges[idx, 0], edges[idx, 1] = idx // n2, n1 + idx % n2
        return edges
    lq = np.log(1.0 - p)
    mean = p * total
    edges = np.empty((int(mean + 5 * np.sqrt(mean)) + 16, 2), dtype=np.int64)
    m = 0
    idx = _skip(lq) - 1
    while idx < total:
        if m == edges.shape[0]:
            edges = _grow(edges, m)
        edges[m, 0], edges[m, 1] = idx // n2, n1 + idx % n2
        m += 1
        idx += _skip(lq)
    return edges[:m].copy()
