"""Closed forms of :mod:`tinto.theory` against the enumerations of :mod:`tinto.exact`.

Each test confronts two of the three sources of the companion checks, on
sizes small enough for the test suite (which runs without JIT). The
companion notebook repeats the comparisons on larger cases.
"""

import itertools
from fractions import Fraction
from math import factorial

import numpy as np
import pytest

from tinto import exact
from tinto.graphs import edges_to_graph, line, ring
from tinto.theory import bipartite, cayley, erdos_renyi, general, path_cycle, star


def _strip(hist):
    out = [int(x) for x in hist]
    while out and out[-1] == 0:
        out.pop()
    return out


@pytest.mark.parametrize("k", [1, 2, 3])
def test_path_and_cycle_counts_against_enumeration(k):
    total, same, dist = path_cycle.path_counts(5, k)
    cyc = path_cycle.cycle_counts(5, k)
    for n in range(1, 6):
        assert total[n] == _strip(exact.peaks_distribution(line(n), k))
        if n >= 2:
            hs, hd = exact.peaks_distribution(line(n), k, split=(0, n - 1))
            assert (same[n], dist[n]) == (_strip(hs), _strip(hd))
        if n >= 3:
            assert [n * c for c in cyc[n]] == _strip(
                exact.peaks_distribution(ring(n), k)
            )


def test_path_law_moments_match_the_closed_forms():
    law = path_cycle.law(12, 2)
    es = sum(j * x for j, x in enumerate(law))
    var = sum(j * j * x for j, x in enumerate(law)) - es**2
    assert sum(law) == 1
    assert float(12 - es) == pytest.approx(path_cycle.mean(12, 2))
    assert float(var) == pytest.approx(path_cycle.variance(12, 2))


def test_closed_forms_against_the_series():
    p = path_cycle._riccati(24)
    y = path_cycle.cycle_counts(24)
    for v, z in [(0.5, 0.1), (2.0, 0.2), (3.0, 0.05)]:
        p_series = sum(
            sum(c * v**j for j, c in enumerate(p[m])) * z**m / factorial(m)
            for m in range(25)
        )
        y_series = sum(
            sum(c * v**j for j, c in enumerate(y[m])) * z**m / factorial(m)
            for m in range(25)
        )
        assert abs(path_cycle.path_series(z, v) - p_series) < 1e-10 * abs(p_series)
        assert abs(path_cycle.cycle_series(z, v) - y_series) < 1e-10 * abs(y_series)


@pytest.mark.parametrize("k", [1, 2])
def test_erdos_renyi_moments_against_enumeration(k):
    n, p = 4, Fraction(1, 3)
    pairs = n * (n - 1) // 2
    es = ess = Fraction(0)
    for g in exact.all_graphs(n):
        m = len(g.ind) // 2
        a, b = exact.factorial_moments(exact.peaks_distribution(g, k))
        weight = p**m * (1 - p) ** (pairs - m)
        es, ess = es + weight * a, ess + weight * b
    assert erdos_renyi.factorial_moments(n, p, k) == (es, ess)


def test_erdos_renyi_variance_integral_form():
    n, p, k = 5, Fraction(1, 2), 2
    es, ess = erdos_renyi.factorial_moments(n, p, k)
    assert erdos_renyi.variance(n, float(p), k) == pytest.approx(
        float(ess + es - es * es), abs=1e-9
    )


def test_erdos_renyi_limit_law_mean():
    law = erdos_renyi.limit_law(0.3, k=2, smax=80)
    assert law.sum() == pytest.approx(1.0, abs=1e-9)
    assert (law * np.arange(81)).sum() == pytest.approx(
        2 * sum(0.7**i for i in range(200)), rel=1e-9
    )


def test_bipartite_law_against_enumeration():
    n1, n2, p = 2, 3, Fraction(1, 3)
    cross = [(u, n1 + v) for u in range(n1) for v in range(n2)]
    law = [Fraction(0)] * (n1 + n2 + 1)
    for mask in range(1 << len(cross)):
        edges = [cross[i] for i in range(len(cross)) if mask >> i & 1]
        g = edges_to_graph(n1 + n2, np.array(edges, dtype=np.int64).reshape(-1, 2))
        weight = p ** len(edges) * (1 - p) ** (len(cross) - len(edges))
        hist = exact.peaks_distribution(g, 1)
        for j, c in enumerate(hist):
            law[j] += weight * Fraction(int(c), int(hist.sum()))
    assert bipartite.law(n1, n2, p) == law
    mean = sum(j * x for j, x in enumerate(law))
    assert float(n1 + n2 - mean) == pytest.approx(bipartite.mean(n1, n2, float(p)))


def test_cayley_pair_laws_against_enumeration():
    n = 5
    trees = list(exact.all_trees(n))
    joint, adjacent, distance_two = {}, {}, {}
    for t in trees:
        nb0, nb1 = set(t.neighbors(0).tolist()), set(t.neighbors(1).tolist())
        key = (len(nb0), len(nb1))
        joint[key] = joint.get(key, 0) + 1
        if 1 in nb0:
            adjacent[key] = adjacent.get(key, 0) + 1
        elif nb0 & nb1:
            distance_two[key] = distance_two.get(key, 0) + 1
    total = len(trees)
    n_adj, n_two = sum(adjacent.values()), sum(distance_two.values())
    assert (Fraction(n_adj, total), Fraction(n_two, total)) == cayley.pair_distances(n)
    for (i, j), c in joint.items():
        assert cayley.pair_degrees(n, i, j) == Fraction(c, total)
    for (i, j), c in adjacent.items():
        assert cayley.adjacent_pair_degrees(n, i - 1, j - 1) == Fraction(c, n_adj)
    for (i, j), c in distance_two.items():
        assert cayley.distance_two_pair_degrees(n, i - 1, j - 1) == Fraction(c, n_two)


@pytest.mark.parametrize("k", [1, 2])
def test_cayley_mean_and_variance_against_enumeration(k):
    n = 5
    es = ess = Fraction(0)
    trees = list(exact.all_trees(n))
    for t in trees:
        a, b = exact.factorial_moments(exact.peaks_distribution(t, k, labels=range(n)))
        es, ess = es + a / len(trees), ess + b / len(trees)
    assert cayley.mean(n, k) == pytest.approx(float(n - es))
    assert cayley.variance(n, k) == pytest.approx(float(ess + es - es * es), abs=1e-8)


def test_cayley_pgf_against_enumeration():
    n, k = 5, 2
    hist = np.zeros(n + 1)
    for t in exact.all_trees(n):
        hist += exact.peaks_distribution(t, k, labels=range(n))
    hist /= hist.sum()
    for v in (0.3, 1.5):
        assert cayley.pgf(n, k, v) == pytest.approx(
            (hist * v ** np.arange(n + 1)).sum(), abs=1e-12
        )


def test_fixed_coloring_mean_against_enumeration():
    rng = np.random.default_rng(3)
    edges = [
        (u, v) for u, v in itertools.combinations(range(6), 2) if rng.random() < 0.5
    ]
    g = edges_to_graph(6, np.array(edges, dtype=np.int64).reshape(-1, 2))
    colors = [0, 1, 0, 0, 1, 2]
    beaten = 0
    for lab in itertools.permutations(range(6)):
        beaten += sum(
            any(
                colors[u] == colors[v] and lab[u] > lab[v]
                for u in g.neighbors(v).tolist()
            )
            for v in range(6)
        )
    assert general.fixed_coloring_mean(g, colors) == Fraction(beaten, 720)


def test_star_against_enumeration():
    for n in range(2, 7):
        st = edges_to_graph(n, np.array([(0, i) for i in range(1, n)]))
        hist = exact.peaks_distribution(st)
        es, ess = exact.factorial_moments(hist)
        assert [Fraction(int(h), factorial(n)) for h in hist] == star.law(n)
        assert (n - es, ess + es - es * es) == (star.mean(n), star.variance(n))


def test_monochrome_variance_against_enumeration():
    for g in exact.all_graphs(4):
        es, ess = exact.factorial_moments(exact.peaks_distribution(g))
        var = general.monochrome_variance(g.to_adja().astype(np.bool_))
        assert var == pytest.approx(float(ess + es - es * es), abs=1e-12)


def test_egf_inverse_with_gaps_and_order_zero():
    from tinto.theory import _egf

    a = _egf.const(1, 6)
    a[2] = [0, 3]  # 1 + 3 v z^2 / 2!, no term in z
    product = _egf.strip(_egf.mul(a, _egf.inv(a)))
    assert product[0] == [1] and all(c == [] for c in product[1:])
    assert path_cycle._riccati(0) == [[]]
