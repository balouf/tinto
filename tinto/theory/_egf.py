"""Exponential generating functions with polynomial coefficients, in integers.

A series is a list ``A`` of polynomials in `v`, each a list of Python ints
(coefficient of ``v^j`` at index `j`). ``A[n]`` is the *normalized*
coefficient ``n! [z^n] A(z, v)``: for the series of the papers it counts
labeled structures of size `n` by their number of peaks, so all the
arithmetic stays in exact integers. The product of two such series is the
binomial convolution; integration in `z` shifts the coefficients.
"""

from math import comb


def padd(p, q):
    """Sum of two polynomials."""
    r = [0] * max(len(p), len(q))
    for i, c in enumerate(p):
        r[i] += c
    for i, c in enumerate(q):
        r[i] += c
    return r


def pscale(p, c):
    """Polynomial times a scalar."""
    return [a * c for a in p]


def pmul(p, q):
    """Product of two polynomials."""
    if not p or not q:
        return []
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                r[i + j] += a * b
    return r


def zero(order):
    """The zero series up to `order`."""
    return [[] for _ in range(order + 1)]


def const(c, order):
    """The constant series `c`."""
    s = zero(order)
    s[0] = [c]
    return s


def z(order):
    """The series `z` (one structure of size 1, no peak counted)."""
    s = zero(order)
    s[1] = [1]
    return s


def add(a, b):
    """Sum of two series of the same order."""
    return [padd(p, q) for p, q in zip(a, b)]


def scale(a, p):
    """Series times a polynomial in `v` (constant in `z`)."""
    return [pmul(c, p) for c in a]


def mul(a, b):
    """Product of two series: binomial convolution of the normalized coefficients."""
    order = len(a) - 1
    r = zero(order)
    for i in range(order + 1):
        if not a[i]:
            continue
        for j in range(order + 1 - i):
            if b[j]:
                r[i + j] = padd(r[i + j], pscale(pmul(a[i], b[j]), comb(i + j, i)))
    return r


def inv(a):
    """Inverse of a series whose constant coefficient is the polynomial 1."""
    assert [c for c in a[0] if c] == [1] and a[0][0] == 1
    order = len(a) - 1
    r = zero(order)
    r[0] = [1]
    for n in range(1, order + 1):
        acc = []
        for i in range(1, n + 1):
            if a[i] and r[n - i]:
                acc = padd(acc, pscale(pmul(a[i], r[n - i]), comb(n, i)))
        r[n] = pscale(acc, -1)
    return r


def integral(a):
    """Integral in `z` from 0: the normalized coefficients shift by one."""
    return [[]] + [list(c) for c in a[:-1]]


def strip(a):
    """The series with the trailing zero coefficients of each polynomial removed."""
    out = []
    for c in a:
        c = list(c)
        while c and c[-1] == 0:
            c.pop()
        out.append(c)
    return out
