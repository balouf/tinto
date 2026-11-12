r"""Quantities valid on every graph: the peak probability of a node (Theorem 1)
and the exact first two moments of the stabilization time (Theorems 1 and 3).

The exact functions work in rational arithmetic: with ``lo = 1 - 1/k``, every
integral below is an integral of a polynomial over ``[lo, 1]``, evaluated as
a :class:`~fractions.Fraction`.

The graph functions read a graph through ``graph.n`` and
``graph.neighbors(i)`` only (a :class:`~tinto.graphs.Graph` qualifies); this
module imports nothing from tinto. The exception is
:func:`monochrome_variance`, compiled and in floating point, which reads an
adjacency matrix: it serves loops over millions of graphs.
"""

from fractions import Fraction
from math import factorial

import numpy as np
from numba import njit


def _lo(k):
    return Fraction(k - 1, k)


def _integral(e, k):
    r""":math:`\int_{1-1/k}^1 s^e \, ds`, exact."""
    return (1 - _lo(k) ** (e + 1)) / (e + 1)


def rho(d, k=1, exact=False):
    r"""Probability that a node of degree `d` is a peak, under `k` colors.

    A node of degree `d` is a peak when its label exceeds those of its
    same-colored neighbors. With a uniform label ``s`` and each neighbor
    same-colored with probability ``1/k``, this gives (Theorem 1 of the
    exact stabilization-time paper)

    .. math::

        \rho(d, k) = k \int_{1-1/k}^1 s^d \, ds = \frac{k}{d+1}\left(1 - \left(1 - \frac1k\right)^{d+1}\right).

    Parameters
    ----------
    d: :class:`int`
        Degree of the node.
    k: :class:`int`, default=1
        Number of colors.
    exact: :class:`bool`, default=False
        Return a :class:`~fractions.Fraction` instead of a float.

    Returns
    -------
    :class:`float` or :class:`~fractions.Fraction`

    Examples
    --------
    Monochrome, a node of degree `d` is a peak iff its label is the largest
    of ``d + 1``:

    >>> rho(2)
    0.3333333333333333

    With two colors, a leaf is beaten only when its neighbor shares its color
    and has the larger label:

    >>> rho(1, 2)
    0.75
    >>> rho(2, 3, exact=True)
    Fraction(19, 27)
    """
    if exact:
        return k * _integral(d, k)
    return k / (d + 1) * (1 - (1 - 1 / k) ** (d + 1))


def pi_same(a, b, c, k=1):
    r"""Same-color term of Theorem 3, exact.

    For two non-adjacent nodes `u` and `v` with `a` private neighbors of `u`,
    `b` common neighbors and `c` private neighbors of `v`,

    .. math::

        \pi_{same}(a, b, c) = k \iint_{[1-1/k, 1]^2} s^a t^c \min(s, t)^b \, ds\, dt

    is the probability that both are peaks and share their color (the
    color is shared with probability ``1/k``, which cancels one factor `k`).
    The integral is split on ``s <= t`` and ``s >= t``.

    Parameters
    ----------
    a, b, c: :class:`int`
        Private neighbors of `u`, common neighbors, private neighbors of `v`.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`~fractions.Fraction`

    Examples
    --------
    Two isolated nodes of the same color are both peaks:

    >>> pi_same(0, 0, 0)
    Fraction(1, 1)
    """
    lo = _lo(k)
    m = a + b  # half s <= t: int t^c int_lo^t s^(a+b) ds dt
    lower = (_integral(c + m + 1, k) - lo ** (m + 1) * _integral(c, k)) / (m + 1)
    m = b + c  # half s >= t: int s^a int_lo^s t^(b+c) dt ds
    upper = (_integral(a + m + 1, k) - lo ** (m + 1) * _integral(a, k)) / (m + 1)
    return k * (lower + upper)


def pi_dist(a, b, c, k=1):
    r"""Distinct-color term of Theorem 3, exact.

    For two nodes `u` and `v` (adjacent or not) with `a` private neighbors
    of `u`, `b` common neighbors and `c` private neighbors of `v`,

    .. math::

        \pi_{dist}(a, b, c) = (k-1) k \iint_{[1-1/k, 1]^2} s^a t^c (s + t - 1)^b \, ds\, dt

    is the probability that both are peaks with distinct colors. The power
    of ``s + t - 1`` is expanded by the trinomial formula.

    Parameters
    ----------
    a, b, c: :class:`int`
        Private neighbors of `u`, common neighbors, private neighbors of `v`.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`~fractions.Fraction`
        Zero when ``k = 1``.

    Examples
    --------
    Two isolated nodes with two colors: distinct colors with probability 1/2,
    and then both are peaks:

    >>> pi_dist(0, 0, 0, 2)
    Fraction(1, 2)
    """
    if k == 1:
        return Fraction(0)
    total = Fraction(0)
    for i in range(b + 1):
        for j in range(b + 1 - i):
            rest = b - i - j
            coef = Fraction(factorial(b), factorial(i) * factorial(j) * factorial(rest))
            total += coef * (-1) ** rest * _integral(a + i, k) * _integral(c + j, k)
    return (k - 1) * k * total


def pair_parameters(graph, u, v):
    """Private and common neighbors of two nodes.

    Parameters
    ----------
    graph: :class:`~tinto.graphs.Graph`
        Undirected graph.
    u, v: :class:`int`
        Two distinct nodes.

    Returns
    -------
    :class:`tuple`
        ``(a, b, c, adjacent)``: neighbors of `u` only, common neighbors,
        neighbors of `v` only (`u` and `v` themselves excluded), and whether
        `u` and `v` are adjacent.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> pair_parameters(line(4), 0, 2)
    (0, 1, 1, False)
    """
    nu = set(graph.neighbors(u).tolist())
    nv = set(graph.neighbors(v).tolist())
    adjacent = v in nu
    nu.discard(v)
    nv.discard(u)
    return len(nu - nv), len(nu & nv), len(nv - nu), adjacent


def moments(graph, k=1):
    r"""Exact :math:`\mathbb E S` and :math:`\mathbb E S(S-1)` on a graph (Theorems 1 and 3).

    :math:`\mathbb E S = \sum_v \rho(d_v, k)`, and
    :math:`\mathbb E S(S-1)` sums over ordered pairs of distinct nodes
    :func:`pi_dist`, plus :func:`pi_same` when the pair is not an edge (two
    adjacent nodes of the same color cannot both be peaks).

    Parameters
    ----------
    graph: :class:`~tinto.graphs.Graph`
        Undirected graph.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`tuple` of :class:`~fractions.Fraction`
        ``(E S, E S(S-1))``.

    Examples
    --------
    On a monochrome triangle exactly one node is a peak:

    >>> from tinto.graphs import ring
    >>> moments(ring(3))
    (Fraction(1, 1), Fraction(0, 1))
    """
    n = graph.n
    es = sum(
        (rho(len(graph.neighbors(v)), k, exact=True) for v in range(n)), Fraction(0)
    )
    ess = Fraction(0)
    for u in range(n):
        for v in range(u + 1, n):
            a, b, c, adjacent = pair_parameters(graph, u, v)
            pair = pi_dist(a, b, c, k) + (0 if adjacent else pi_same(a, b, c, k))
            ess += 2 * pair  # both orders; the terms are symmetric in (u, v)
    return es, ess


def mean(graph, k=1):
    """Exact mean of the stabilization time ``T = n - S`` on a graph.

    Parameters
    ----------
    graph: :class:`~tinto.graphs.Graph`
        Undirected graph.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`~fractions.Fraction`

    Examples
    --------
    >>> from tinto.graphs import line
    >>> mean(line(3))
    Fraction(5, 3)
    """
    return graph.n - moments(graph, k)[0]


def variance(graph, k=1):
    r"""Exact variance of the stabilization time on a graph.

    :math:`\mathrm{Var}\, T = \mathrm{Var}\, S = \mathbb E S(S-1) + \mathbb E S - (\mathbb E S)^2`.

    Parameters
    ----------
    graph: :class:`~tinto.graphs.Graph`
        Undirected graph.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`~fractions.Fraction`

    Examples
    --------
    >>> from tinto.graphs import line
    >>> variance(line(3))
    Fraction(2, 9)
    """
    es, ess = moments(graph, k)
    return ess + es - es * es


@njit
def monochrome_variance(adj):
    r"""Monochrome variance of the stabilization time, in floating point and compiled.

    The pairwise decomposition
    :math:`\mathrm{Var}\, T = \sum_v \mathrm{Var}\, X_v + \sum_{u \ne v} \mathrm{Cov}(X_u, X_v)`,
    with :math:`X_v` the peak indicator of `v`,
    :math:`\mathrm{Var}\, X_v = d_v / (d_v + 1)^2`, and the covariances of
    the lemma "Pairwise covariance" of the exact stabilization-time paper:
    :math:`-1/((d_u + 1)(d_v + 1))` for adjacent nodes, and

    .. math::

        \mathrm{Cov}(X_u, X_v) = \frac{b}{(a + b + 1)(b + c + 1)(a + b + c + 2)}

    for non-adjacent nodes with `a` private neighbors of `u`, `b` common
    neighbors and `c` private neighbors of `v`. Same value as
    :func:`variance` at ``k = 1``, without rational arithmetic: meant for
    loops over many graphs (exhaustive enumeration, local search).

    Parameters
    ----------
    adj: :class:`~numpy.ndarray` of :class:`bool`, shape (n, n)
        Adjacency matrix: symmetric, with a zero diagonal.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> import numpy as np
    >>> from tinto.graphs import line
    >>> round(monochrome_variance(line(3).to_adja().astype(np.bool_)), 12)
    0.222222222222
    """
    n = adj.shape[0]
    deg = np.zeros(n, dtype=np.int64)
    for u in range(n):
        for v in range(n):
            if adj[u, v]:
                deg[u] += 1
    var = 0.0
    for v in range(n):
        var += deg[v] / (deg[v] + 1) ** 2
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                var -= 2.0 / ((deg[u] + 1) * (deg[v] + 1))
                continue
            b = 0
            for w in range(n):
                if adj[u, w] and adj[v, w]:
                    b += 1
            if b > 0:
                a, c = deg[u] - b, deg[v] - b
                var += 2.0 * b / ((a + b + 1) * (b + c + 1) * (a + b + c + 2))
    return var


def fixed_coloring_mean(graph, colors):
    r"""Exact mean stabilization time from a fixed initial coloring.

    From a fixed coloring :math:`\gamma`, only the labeling is random: a node
    is a peak iff its label is the largest among itself and its
    :math:`d_v(\gamma)` same-colored neighbors, so

    .. math::

        \mathbb E\, T(G, \gamma) = \sum_v \frac{d_v(\gamma)}{d_v(\gamma) + 1}.

    Parameters
    ----------
    graph: :class:`~tinto.graphs.Graph`
        Undirected graph.
    colors: array_like
        Initial color of each node.

    Returns
    -------
    :class:`~fractions.Fraction`

    Examples
    --------
    >>> from tinto.graphs import line
    >>> fixed_coloring_mean(line(3), [0, 0, 1])
    Fraction(1, 1)
    """
    total = Fraction(0)
    for v in range(graph.n):
        d = sum(1 for u in graph.neighbors(v).tolist() if colors[u] == colors[v])
        total += Fraction(d, d + 1)
    return total
