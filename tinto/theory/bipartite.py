"""Bipartite :math:`G(n_1, n_2, p)` graphs."""

from math import comb

from scipy import integrate

from tinto.theory.erdos_renyi import integrals


def mean(n1, n2, p, k=1):
    r"""Mean stabilization time on :math:`G(n_1, n_2, p)` under `k` colors.

    A node of the first side is a peak iff none of the larger nodes of the
    other side is a same-colored neighbor; averaging over the interleaving
    of the two sides gives

    .. math::

        \mathbb E S = \frac kp \left(\frac{n_1}{n_2+1}\left(1 - q^{n_2+1}\right)
        + \frac{n_2}{n_1+1}\left(1 - q^{n_1+1}\right)\right), \quad q = 1 - \frac pk.

    Parameters
    ----------
    n1: :class:`int`
        Size of the first side.
    n2: :class:`int`
        Size of the second side.
    p: :class:`float`
        Probability of an edge across the sides, ``0 < p <= 1``.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(mean(100, 100, 0.1), 6)
    180.198493
    """
    q = 1 - p / k
    es = (k / p) * (
        n1 / (n2 + 1) * (1 - q ** (n2 + 1)) + n2 / (n1 + 1) * (1 - q ** (n1 + 1))
    )
    return n1 + n2 - es


def variance(n1, n2, p, k=1):
    r"""Variance of the stabilization time on :math:`G(n_1, n_2, p)` under `k` colors.

    Pairs on the same side are never adjacent and see the other side only:
    they use :func:`~tinto.theory.erdos_renyi.integrals` with ``m`` the size
    of the other side. Pairs across the sides are adjacent with probability
    `p` (then only distinct colors can make two peaks); their private
    neighborhoods are independent binomials, which gives two more double
    integrals.

    Parameters
    ----------
    n1: :class:`int`
        Size of the first side.
    n2: :class:`int`
        Size of the second side.
    p: :class:`float`
        Probability of an edge across the sides, ``0 < p <= 1``.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(variance(100, 100, 0.1), 4)
    9.9943
    >>> round(variance(100, 100, 0.1, 3), 6)
    28.284911
    """
    es = n1 + n2 - mean(n1, n2, p, k)
    d2, s2 = integrals(n2, p, k)  # pairs in the first side, neighbors in the second
    d1, s1 = integrals(n1, p, k)  # pairs in the second side, neighbors in the first

    def l_dist(y, x):
        return (1 - p * (1 - x) / k) ** (n2 - 1) * (1 - p * (1 - y) / k) ** (n1 - 1)

    def l_same(y, x):
        a = (1 - p * (1 - x * y) / k) ** (n2 - 1) * (1 - p * (1 - y) / k) ** (n1 - 1)
        b = (1 - p * (1 - x * y) / k) ** (n1 - 1) * (1 - p * (1 - y) / k) ** (n2 - 1)
        return y * (a + b)

    cross_dist = (
        (k - 1) / k * integrate.dblquad(l_dist, 0, 1, 0, 1)[0] if k > 1 else 0.0
    )
    cross_same = 1.0 / k * integrate.dblquad(l_same, 0, 1, 0, 1)[0]
    ess = (
        n1 * (n1 - 1) * (s2 + d2)
        + n2 * (n2 - 1) * (s1 + d1)
        + 2 * n1 * n2 * (p * cross_dist + (1 - p) * (cross_same + cross_dist))
    )
    return ess + es - es * es


def law(n1, n2, p):
    r"""Exact law of the number of peaks on the monochrome :math:`G(n_1, n_2, p)`.

    Take the nodes in decreasing label order. Given the interleaving of the
    two sides in that order, a node is a peak iff it has no edge to the
    larger nodes of the other side, independently of the others: the
    conditional generating function is a product
    :math:`\prod_u (1 - q^{h_u} + q^{h_u} v)`, :math:`q = 1 - p`,
    :math:`h_u` the number of larger nodes on the other side. The law
    averages it over the :math:`\binom{n_1 + n_2}{n_1}` equally likely
    interleavings, here by dynamic programming over the prefixes.

    Parameters
    ----------
    n1, n2: :class:`int`
        Sizes of the two sides.
    p: :class:`float` or :class:`~fractions.Fraction`
        Probability of an edge across the sides; a
        :class:`~fractions.Fraction` gives an exact law.

    Returns
    -------
    :class:`list`
        :math:`P(S = j)` for ``j = 0, ..., n1 + n2``.

    Examples
    --------
    One node on each side: one peak with an edge, two without.

    >>> from fractions import Fraction
    >>> [str(x) for x in law(1, 1, Fraction(1, 3))]
    ['0', '1/3', '2/3']
    """
    q = 1 - p
    n = n1 + n2
    table = {(0, 0): [1] + [0] * n}
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            poly = table[i, j]  # every prefix (i, j) is reachable
            for di, dj, h in ((1, 0, j), (0, 1, i)):
                if i + di > n1 or j + dj > n2:
                    continue
                peak = q**h
                new = [c * (1 - peak) for c in poly]
                for d in range(n):
                    new[d + 1] += poly[d] * peak
                old = table.get((i + di, j + dj))
                table[i + di, j + dj] = (
                    new if old is None else [a + b for a, b in zip(old, new)]
                )
    total = comb(n, n1)
    return [c / total for c in table[n1, n2]]


def limit_mean(lam, p):
    r"""Limit of :math:`\mathbb E S` on the monochrome :math:`G(\lambda n, (1-\lambda) n, p)`.

    .. math::

        \lim_{n \to \infty} \mathbb E S = \frac{\lambda}{(1-\lambda) p} + \frac{1-\lambda}{\lambda p}.

    Parameters
    ----------
    lam: :class:`float`
        Fraction of the nodes on the first side, ``0 < lam < 1``.
    p: :class:`float`
        Probability of an edge across the sides.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(limit_mean(0.5, 0.25), 6)
    8.0
    """
    return lam / ((1 - lam) * p) + (1 - lam) / (lam * p)
