"""Uniform labeled trees (Cayley trees)."""

import math
from fractions import Fraction
from math import comb

import numpy as np
from scipy import integrate

from tinto.theory.general import rho

_OPTS = dict(epsabs=1e-12, epsrel=1e-12)


def mean(n, k=1):
    r"""Mean stabilization time on the uniform labeled tree with `n` nodes, `k` colors.

    The degree of a fixed node is ``1 + Binomial(n - 2, 1/n)``; averaging
    :func:`~tinto.theory.general.rho` over it gives :math:`\mathbb E S / n`.
    Monochrome, :math:`\mathbb E T = n - n(1 - 1/n)^{n-1}`.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(mean(200), 6)
    126.239634

    A single node never moves:

    >>> mean(1)
    0.0
    """
    if n == 1:
        return 0.0
    es = sum(
        comb(n - 2, j) * (1 / n) ** j * (1 - 1 / n) ** (n - 2 - j) * rho(j + 1, k)
        for j in range(n - 1)
    )
    return n - n * es


def variance(n, k=1):
    r"""Variance of the stabilization time on the uniform labeled tree with `n` nodes, `k` colors.

    Monochrome, the closed form :math:`n(n-1)(1-2/n)^{n-1} + m - m^2` with
    :math:`m = n(1-1/n)^{n-1}`. With ``k >= 2`` colors, the double integrals
    over the labels of a pair of nodes, split by their distance (adjacent,
    at distance two, further) through the pair-degree laws of the Cayley
    tree. The kink of ``min(s, t)`` is integrated on the two triangles
    separately.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes, ``n >= 4`` when ``k >= 2``.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(variance(200), 6)
    19.422343
    >>> round(variance(200, 3), 6)
    29.174695
    """
    if k == 1:
        m = n * (1 - 1 / n) ** (n - 1)
        return n * (n - 1) * (1 - 2 / n) ** (n - 1) + m - m * m
    lo = 1 - 1 / k

    def bracket(f):
        return k * k * integrate.dblquad(lambda t, s: f(s, t), lo, 1, lo, 1, **_OPTS)[0]

    def bracket_min(f):
        lower = integrate.dblquad(
            lambda s, t: s * f(s, t), lo, 1, lambda t: lo, lambda t: t, **_OPTS
        )[0]
        upper = integrate.dblquad(
            lambda s, t: t * f(s, t), lo, 1, lambda t: t, lambda t: 1, **_OPTS
        )[0]
        return k * k * (lower + upper)

    def phi(z, m):
        return ((z + n - 2) / n) ** m

    def g(s, t):
        return (s + t + 1) / 3 * phi(s + t, n - 4)

    a = bracket(lambda s, t: s * t * phi(s + t, n - 2))
    b = bracket(lambda s, t: (s + t) / 2 * phi(s + t, n - 3) * ((k - 1) / k - s * t))
    c = bracket_min(g) / k + bracket(
        lambda s, t: g(s, t) * ((k - 1) / k * (s + t - 1) - s * t)
    )
    ess = n * (n - 1) * (a + 2 / n * b + 3 * (n - 2) / n**2 * c)
    es = n - mean(n, k)
    return ess + es - es * es


def pair_degrees(n, i, j):
    r"""Joint law of the degrees of two distinct nodes of the uniform labeled tree.

    .. math::

        P(d_u = i, d_v = j) = \binom{n-2}{i-1}\binom{n-i-1}{j-1}
        n^{2-i-j} \left(1 - \frac2n\right)^{n-i-j}.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes, ``n >= 3``.
    i, j: :class:`int`
        Degrees, at least 1.

    Returns
    -------
    :class:`~fractions.Fraction`

    Examples
    --------
    >>> sum(pair_degrees(5, i, j) for i in range(1, 5) for j in range(1, 5))
    Fraction(1, 1)
    """
    m = n - 2
    return (
        comb(m, i - 1)
        * comb(m - i + 1, j - 1)
        * Fraction(n) ** (2 - i - j)
        * (1 - Fraction(2, n)) ** (m - i - j + 2)
    )


def pair_distances(n):
    """Probabilities that two distinct nodes of the uniform labeled tree are adjacent, or at distance two.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.

    Returns
    -------
    :class:`tuple` of :class:`~fractions.Fraction`
        ``(2/n, 3(n-2)/n^2)``.

    Examples
    --------
    >>> pair_distances(5)
    (Fraction(2, 5), Fraction(9, 25))
    """
    return Fraction(2, n), Fraction(3 * (n - 2), n * n)


def adjacent_pair_degrees(n, a, c):
    r"""Law of the other degrees of two adjacent nodes of the uniform labeled tree.

    Given that `u` and `v` are adjacent, with :math:`a = d_u - 1` and
    :math:`c = d_v - 1`,

    .. math::

        P(a, c) = \binom{n-3}{a+c-1}\binom{a+c}{a}
        \left(\frac2n\right)^{a+c-1} \left(1 - \frac2n\right)^{n-2-a-c} 2^{-(a+c)}.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes, ``n >= 3``.
    a, c: :class:`int`
        Degrees minus one.

    Returns
    -------
    :class:`~fractions.Fraction`

    Examples
    --------
    >>> sum(adjacent_pair_degrees(5, a, c) for a in range(4) for c in range(4))
    Fraction(1, 1)
    """
    if a + c == 0:
        return Fraction(0)
    return (
        comb(n - 3, a + c - 1)
        * comb(a + c, a)
        * Fraction(2, n) ** (a + c - 1)
        * (1 - Fraction(2, n)) ** (n - 2 - a - c)
        / 2 ** (a + c)
    )


def distance_two_pair_degrees(n, a, c):
    r"""Law of the other degrees of two nodes at distance two in the uniform labeled tree.

    Given that `u` and `v` have a common neighbor, with :math:`a = d_u - 1`
    and :math:`c = d_v - 1`, summing over the degree `D` of the middle node
    without `u` and `v`,

    .. math::

        P(a, c) = \sum_D \binom{n-4}{D-1} \left(\frac3n\right)^{D-1}
        \left(1 - \frac3n\right)^{n-3-D} \binom{D}{a}\binom{D-a}{c} 3^{-D}.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes, ``n >= 4``.
    a, c: :class:`int`
        Degrees minus one.

    Returns
    -------
    :class:`~fractions.Fraction`

    Examples
    --------
    >>> sum(distance_two_pair_degrees(6, a, c) for a in range(5) for c in range(5))
    Fraction(1, 1)
    """
    total = Fraction(0)
    for d in range(max(1, a + c), n - 2):
        total += (
            comb(n - 4, d - 1)
            * Fraction(3, n) ** (d - 1)
            * (1 - Fraction(3, n)) ** (n - 3 - d)
            * comb(d, a)
            * comb(d - a, c)
            / 3**d
        )
    return total


def slope(k=1):
    r"""Asymptotic slope of the variance on the uniform labeled tree.

    .. math::

        \lim_{n \to \infty} \frac{\mathrm{Var}\, T_n}{n} = e^{-1/k} - \left(1 + \frac{1}{k^2}\right) e^{-2/k}.

    Parameters
    ----------
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(slope(), 6)
    0.097209
    """
    return math.exp(-1 / k) - (1 + k**-2) * math.exp(-2 / k)


def pgf(n, k, v):
    r"""Probability generating function of the number of peaks on the uniform labeled tree, `k >= 2` colors.

    Percolating the monochrome edges and Lagrange inversion give

    .. math::

        \mathbb E\, v^{S_n} = \frac{n!}{(k-1) k^{n-1} n^n} [z^n] \exp\big((k-1) n F(z, v)\big),
        \quad F = C\big(z e^{(v-1) z}\big) + (v-1) z,

    with `C` the Cayley tree function. The coefficient is computed by power
    series arithmetic in `z`, at a numeric `v`.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    k: :class:`int`
        Number of colors, at least 2.
    v: :class:`float`
        Point of evaluation.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(pgf(10, 3, 1.0), 12)
    1.0
    """
    cayley_coefs = np.array(
        [0.0] + [m ** (m - 1) / math.factorial(m) for m in range(1, n + 1)]
    )
    w = np.array(
        [0.0] + [(v - 1) ** (j - 1) / math.factorial(j - 1) for j in range(1, n + 1)]
    )  # z e^((v-1) z)
    f = np.zeros(n + 1)
    power = np.zeros(n + 1)
    power[0] = 1.0
    for m in range(1, n + 1):
        power = np.convolve(power, w)[: n + 1]
        f += cayley_coefs[m] * power
    f[1] += v - 1
    a = (k - 1) * n
    g = np.zeros(n + 1)  # g = exp(a f), through g' = a f' g
    g[0] = 1.0
    for j in range(1, n + 1):
        g[j] = a / j * sum(i * f[i] * g[j - i] for i in range(1, j + 1))
    return math.factorial(n) / ((k - 1) * k ** (n - 1) * n**n) * g[n]
