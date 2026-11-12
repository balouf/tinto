r"""Monochrome star :math:`\mathrm{St}_n`, one center and ``n - 1`` leaves: the
graph of largest variance (Theorem "Exact variance maximizer" of the exact
stabilization-time paper).

A leaf is a peak iff its label exceeds the center's, and the center is a
peak iff its label is the largest: with the center of label rank `r` among
`n`, :math:`S = n - r` for ``r < n`` and :math:`S = 1` for ``r = n``.
"""

from fractions import Fraction


def law(n):
    r"""Exact law of the number of peaks on the monochrome star with `n` nodes.

    :math:`P(S = 1) = 2/n` and :math:`P(S = s) = 1/n` for
    :math:`2 \le s \le n - 1`, that is :math:`P(T = j) = 1/n` for
    :math:`1 \le j \le n - 2` and :math:`P(T = n - 1) = 2/n`.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes, ``n >= 2``.

    Returns
    -------
    :class:`list` of :class:`~fractions.Fraction`
        :math:`P(S = s)` for ``s = 0, ..., n``.

    Examples
    --------
    >>> [str(x) for x in law(4)]
    ['0', '1/2', '1/4', '1/4', '0']
    """
    out = [Fraction(0)] + [Fraction(1, n)] * (n - 1) + [Fraction(0)]
    out[1] += Fraction(1, n)
    return out


def mean(n):
    r"""Exact mean stabilization time on the monochrome star with `n` nodes.

    .. math::

        \mathbb E\, T = \frac{(n-1)(n+2)}{2n}.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes, ``n >= 2``.

    Returns
    -------
    :class:`~fractions.Fraction`

    Examples
    --------
    >>> mean(4)
    Fraction(9, 4)
    """
    return Fraction((n - 1) * (n + 2), 2 * n)


def variance(n):
    r"""Exact variance of the stabilization time on the monochrome star with `n` nodes.

    .. math::

        \mathrm{Var}\, T = \frac{(n-1)(2n^2 + 5n - 6)}{6n} - \frac{(n-1)^2 (n+2)^2}{4n^2}
        = \frac{n^2}{12} - \frac{13}{12} + o(1).

    Parameters
    ----------
    n: :class:`int`
        Number of nodes, ``n >= 2``.

    Returns
    -------
    :class:`~fractions.Fraction`

    Examples
    --------
    >>> variance(4)
    Fraction(11, 16)
    """
    return Fraction((n - 1) * (2 * n * n + 5 * n - 6), 6 * n) - Fraction(
        (n - 1) ** 2 * (n + 2) ** 2, 4 * n * n
    )
