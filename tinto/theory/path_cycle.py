r"""Path and cycle on `n` nodes: moments, generating functions and exact laws.

The laws come from the generating functions of the exact stabilization-time
paper, computed as exponential series whose normalized coefficients count
(labeling, coloring) pairs by their number of peaks (see
:mod:`tinto.theory._egf`): the monochrome path series solves the Riccati
equation :math:`\partial_z P = v + 2P + P^2`, and the `k`-colored path and
cycle series are rational in `P`.
"""

import cmath
from fractions import Fraction
from math import comb, factorial

from tinto.theory import _egf
from tinto.theory.general import rho


def _bulk(k):
    return (45 * k**3 - 90 * k**2 + 60 * k - 13) / (45 * k**4)


def mean(n, k=1, cycle=False):
    r"""Mean stabilization time on the path (or the cycle) with `n` nodes.

    Sum of the peak probabilities: two ends of degree 1 and ``n - 2`` inner
    nodes of degree 2 on the path, ``n`` nodes of degree 2 on the cycle.
    Exact for ``n >= 2`` (path) and ``n >= 3`` (cycle).

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    k: :class:`int`, default=1
        Number of colors.
    cycle: :class:`bool`, default=False
        Cycle instead of path.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(mean(60), 6)
    39.666667
    >>> round(mean(60, cycle=True), 6)
    40.0
    """
    if cycle:
        return n * (1 - rho(2, k))
    return n - ((n - 2) * rho(2, k) + 2 * rho(1, k))


def variance(n, k=1, cycle=False):
    r"""Variance of the stabilization time on the path (or the cycle) with `n` nodes.

    Linear in `n` with slope :math:`(45k^3 - 90k^2 + 60k - 13)/(45k^4)`; the
    path subtracts the boundary term
    :math:`(45k^3 - 135k^2 + 120k - 32)/(45k^4)`. Exact for ``n >= 4``
    (path) and ``n >= 5`` (cycle), where pairs of nodes at distance two no
    longer overlap.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes.
    k: :class:`int`, default=1
        Number of colors.
    cycle: :class:`bool`, default=False
        Cycle instead of path.

    Returns
    -------
    :class:`float`

    Examples
    --------
    >>> round(variance(60), 6)
    2.711111
    >>> round(variance(60, cycle=True), 6)
    2.666667
    """
    if cycle:
        return n * _bulk(k)
    return n * _bulk(k) - (45 * k**3 - 135 * k**2 + 120 * k - 32) / (45 * k**4)


def _riccati(order):
    """Monochrome path series: ``P' = v + 2P + P^2``, ``P(0) = 0``, normalized coefficients."""
    p = _egf.zero(order)
    if order >= 1:
        p[1] = [0, 1]
    for m in range(1, order):
        acc = _egf.pscale(p[m], 2)
        for i in range(1, m):
            acc = _egf.padd(acc, _egf.pscale(_egf.pmul(p[i], p[m - i]), comb(m, i)))
        p[m + 1] = acc
    return p


def path_counts(order, k=1):
    r"""Peak counts on the path, split by the colors of its two ends.

    With :math:`P` the monochrome series (:math:`\partial_z P = v + 2P + P^2`),
    the `k`-colored path series are

    .. math::

        T = \frac{kP}{1 - (k-1)P}, \quad
        T_{same} = \frac{(1 - (k-2)P)\, kP}{(1 + P)(1 - (k-1)P)}, \quad
        T_{dist} = \frac{k(k-1) P^2}{(1 + P)(1 - (k-1)P)},

    with :math:`T = T_{same} + T_{dist}`.

    Parameters
    ----------
    order: :class:`int`
        Largest number of nodes.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    :class:`tuple` of three series
        ``(total, same, dist)``: ``total[n][j]`` is the number of
        (labeling, coloring) pairs of the path on `n` nodes with `j` peaks
        (``n! k^n`` pairs in all); ``same`` and ``dist`` restrict to the
        colorings whose two ends share their color or not (a single node
        counts as *same*).

    Examples
    --------
    On three nodes, monochrome, 4 labelings of 6 have one peak:

    >>> path_counts(3)[0][3]
    [0, 4, 2]
    >>> total, same, dist = path_counts(2, k=2)
    >>> total[2], same[2], dist[2]
    ([0, 4, 4], [0, 4], [0, 0, 4])
    """
    p = _riccati(order)
    one = _egf.const(1, order)
    inv1 = _egf.inv(_egf.add(one, _egf.scale(p, [-(k - 1)])))  # 1 / (1 - (k-1) P)
    inv12 = _egf.mul(inv1, _egf.inv(_egf.add(one, p)))  # ... / (1 + P)
    kp = _egf.scale(p, [k])
    total = _egf.mul(kp, inv1)
    same = _egf.mul(_egf.mul(_egf.add(one, _egf.scale(p, [-(k - 2)])), kp), inv12)
    dist = _egf.mul(_egf.scale(_egf.mul(p, p), [k * (k - 1)]), inv12)
    return _egf.strip(total), _egf.strip(same), _egf.strip(dist)


def cycle_counts(order, k=1):
    r"""Peak counts on the cycle.

    The cycle is opened at the node of smallest label, which leaves an
    oriented path whose ends may or may not share their color:

    .. math::

        Y = kz + \int (1 + (k-1)v)\, T_{same} + \int (2 + (k-2)v)\, T_{dist},

    with the series of :func:`path_counts` (monochrome,
    :math:`Y = z + \int P`).

    Parameters
    ----------
    order: :class:`int`
        Largest number of nodes.
    k: :class:`int`, default=1
        Number of colors.

    Returns
    -------
    series
        ``cyc[n][j]`` is the number of pairs (labeling with the smallest
        label on node 0, coloring) of the cycle on `n` nodes with `j`
        peaks: ``(n - 1)! k^n`` pairs in all, for ``n >= 3``.

    Examples
    --------
    On a monochrome triangle there is always exactly one peak:

    >>> cycle_counts(3)[3]
    [0, 2]
    """
    _, same, dist = path_counts(order, k)
    cyc = _egf.add(
        _egf.scale(_egf.z(order), [k]),
        _egf.add(
            _egf.integral(_egf.scale(same, [1, k - 1])),
            _egf.integral(_egf.scale(dist, [2, k - 2])),
        ),
    )
    return _egf.strip(cyc)


def law(n, k=1, cycle=False):
    """Exact law of the number of peaks `S` on the path (or the cycle), ``T = n - S``.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes, ``n >= 3`` for the cycle.
    k: :class:`int`, default=1
        Number of colors.
    cycle: :class:`bool`, default=False
        Cycle instead of path.

    Returns
    -------
    :class:`list` of :class:`~fractions.Fraction`
        ``P(S = j)`` for ``j = 0, ..., n``.

    Examples
    --------
    >>> [str(x) for x in law(3)]
    ['0', '2/3', '1/3', '0']
    >>> [str(x) for x in law(3, cycle=True)]
    ['0', '1', '0', '0']
    """
    if cycle:
        counts, total = cycle_counts(n, k)[n], factorial(n - 1) * k**n
    else:
        counts, total = path_counts(n, k)[0][n], factorial(n) * k**n
    out = [Fraction(c, total) for c in counts]
    return out + [Fraction(0)] * (n + 1 - len(out))


def path_series(z, v):
    r"""Closed form of the monochrome path series.

    .. math::

        P(z, v) = \frac{v \tan(z r)}{r - \tan(z r)}, \quad r = \sqrt{v - 1}.

    Parameters
    ----------
    z: :class:`complex`
        Variable of the series, inside its disk of convergence.
    v: :class:`complex`
        Variable marking the peaks, ``v != 1``.

    Returns
    -------
    :class:`complex`

    Examples
    --------
    >>> round(path_series(0.1, 2.0).real, 6)
    0.223049
    """
    r = cmath.sqrt(v - 1)
    t = cmath.tan(z * r)
    return v * t / (r - t)


def cycle_series(z, v):
    r"""Closed form of the monochrome oriented cycle series.

    .. math::

        Y(z, v) = -\log\left(\cos(z r) - \frac{\sin(z r)}{r}\right), \quad r = \sqrt{v - 1}.

    Parameters
    ----------
    z: :class:`complex`
        Variable of the series, inside its disk of convergence.
    v: :class:`complex`
        Variable marking the peaks, ``v != 1``.

    Returns
    -------
    :class:`complex`

    Examples
    --------
    >>> round(cycle_series(0.1, 2.0).real, 6)
    0.110741
    """
    r = cmath.sqrt(v - 1)
    return -cmath.log(cmath.cos(z * r) - cmath.sin(z * r) / r)
