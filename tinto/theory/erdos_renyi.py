""":math:`G(n, p)` Erdős-Rényi graphs."""

from fractions import Fraction

import numpy as np
from scipy import integrate


def mean(n, p, k=1):
    r"""Mean stabilization time on :math:`G(n, p)` under `k` colors.

    The node of rank `r` from the top is a peak iff none of the ``r - 1``
    larger nodes is a same-colored neighbor, which has probability
    :math:`(1 - p/k)^{r-1}`; summing over the ranks,
    :math:`\mathbb E S = (k/p)(1 - (1 - p/k)^n)`.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    p: :class:`float`
        Edge probability, ``0 < p <= 1``.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(mean(200, 0.1), 6)
    190.0
    """
    return n - (k / p) * (1 - (1 - p / k) ** n)


def integrals(m, p, k=1):
    r"""The two double integrals of the :math:`G(n, p)` second factorial moment.

    For a pair of nodes, `m` other nodes can be neighbors of either; the
    integrals average the probability that both are peaks over their
    labels, for distinct colors (:math:`I_{dist}`) and for the same color
    without an edge between them (:math:`I_{same}`). They also serve the
    same-side pairs of :mod:`~tinto.theory.bipartite`.

    Parameters
    ----------
    m: :class:`int`
        Number of other nodes.
    p: :class:`float`
        Edge probability.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`tuple` of :class:`float`
        ``(I_dist, I_same)``, with ``I_dist = 0`` when ``k = 1``.
    """
    alpha, beta, delta = p * (1 - p), p * p, (1 - p) ** 2

    def b_dist(x, y):
        return (
            delta
            + (2 * alpha * (k - 1) + beta * (k - 2) + (alpha + beta) * (x + y)) / k
        )

    def b_same(x, y):
        return (alpha + beta) * (k - 1 + x * y) / k + alpha * (k - 1 + y) / k + delta

    i_dist = (
        (k - 1) / k * integrate.dblquad(lambda y, x: b_dist(x, y) ** m, 0, 1, 0, 1)[0]
        if k > 1
        else 0.0
    )
    i_same = (
        2.0 / k * integrate.dblquad(lambda y, x: y * b_same(x, y) ** m, 0, 1, 0, 1)[0]
    )
    return i_dist, i_same


def variance(n, p, k=1):
    r"""Variance of the stabilization time on :math:`G(n, p)` under `k` colors.

    :math:`\mathbb E S(S-1) = n(n-1)(I_{dist} + (1-p) I_{same})` with the
    integrals of :func:`integrals` at ``m = n - 2`` (a same-colored pair can
    only be two peaks when it is not an edge), then
    :math:`\mathrm{Var}\, S = \mathbb E S(S-1) + \mathbb E S - (\mathbb E S)^2`.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes, ``n >= 2``.
    p: :class:`float`
        Edge probability, ``0 < p <= 1``.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(variance(200, 0.1), 4)
    4.7368
    >>> round(variance(200, 0.1, 3), 6)
    14.176553
    """
    i_dist, i_same = integrals(n - 2, p, k)
    es = n - mean(n, p, k)
    return n * (n - 1) * (i_dist + (1 - p) * i_same) + es - es * es


def factorial_moments(n, p, k=1):
    r"""Exact :math:`\mathbb E S` and :math:`\mathbb E S(S-1)` on :math:`G(n, p)`.

    The closed form of the second factorial moment, with :math:`s = 1 - p/k`,
    :math:`t_1 = 1 - (2p - p^2)/k`, :math:`t_2 = 1 - 2p/k` and
    :math:`F(g) = \frac kp \left(\frac{1 - g^n}{1 - g} - \frac{s^n - g^n}{s - g}\right)`, is

    .. math::

        \mathbb E S(S-1) = 2\left(\frac{1-p}{k} F(t_1) + \frac{k-1}{k} F(t_2)\right).

    With a :class:`~fractions.Fraction` `p` the result is exact.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    p: :class:`float` or :class:`~fractions.Fraction`
        Edge probability, ``0 < p < 1``.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`tuple`
        ``(E S, E S(S-1))``.

    Examples
    --------
    >>> from fractions import Fraction
    >>> factorial_moments(3, Fraction(1, 2))
    (Fraction(7, 4), Fraction(7, 4))
    """
    s = 1 - p / k
    t1 = 1 - (2 * p - p * p) / k
    t2 = 1 - 2 * p / k

    def f(g):
        return (k / p) * ((1 - g**n) / (1 - g) - (s**n - g**n) / (s - g))

    es = (k / p) * (1 - s**n)
    ess = 2 * ((1 - p) / k * f(t1) + Fraction(k - 1, k) * f(t2))
    return es, ess


def limit_law(p, k=1, smax=60):
    r"""Limit law of the number of peaks on :math:`G(n, p)` at fixed `p`, as :math:`n \to \infty`.

    Monochrome, :math:`S_\infty = \sum_{i \ge 0} B_i` with independent
    :math:`B_i \sim \mathrm{Bernoulli}((1-p)^i)` (the node of rank `i` from
    the top is a peak iff it has no edge to the `i` larger ones). With `k`
    colors, the color classes are independent :math:`G(n/k, p)`: the law is
    the `k`-fold convolution.

    Parameters
    ----------
    p: :class:`float`
        Edge probability, ``0 < p <= 1``.
    k: :class:`int`, default=1
        Number of colors.
    smax: :class:`int`, default=60
        Largest value kept; the mass above is dropped.

    Returns
    -------
    :class:`~numpy.ndarray`, shape (smax + 1,)
        :math:`P(S_\infty = s)` for ``s = 0, ..., smax``.

    Examples
    --------
    With ``p = 1`` there is one peak per color class:

    >>> limit_law(1.0, 2, smax=3).tolist()
    [0.0, 0.0, 1.0, 0.0]
    """
    one = np.zeros(smax + 1)
    one[0] = 1.0
    i = 0
    while (1 - p) ** i > 1e-18 or i == 0:
        q = (1 - p) ** i
        nxt = one * (1 - q)
        nxt[1:] += one[:-1] * q
        one = nxt
        i += 1
    law = one
    for _ in range(k - 1):
        law = np.convolve(law, one)[: smax + 1]
    return law
