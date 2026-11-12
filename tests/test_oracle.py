"""Tests for the edge-list generators of :mod:`tinto.graphs` and for :mod:`tinto.oracle`.

They run with ``NUMBA_DISABLE_JIT=1`` (see ``conftest.py``), so sizes stay
small. The statistical tests compare sample means with exact means written
here from first principles (rank arguments), not taken from
:mod:`tinto.theory`, and allow five standard errors.
"""

import itertools

import numpy as np
import pytest

from tinto import oracle
from tinto.graphs import (
    adja_to_graph,
    bipartite,
    bipartite_edges,
    cayley_edges,
    edges_to_graph,
    erdos_renyi_edges,
    line,
)
from tinto.io import set_seed


def _close(samples, exact, z=5.0):
    se = samples.std() / np.sqrt(len(samples))
    return abs(samples.mean() - exact) <= z * se + 1e-12


def _rho(d, k):
    """Probability that a node of degree d is a peak under k colors."""
    return k / (d + 1) * (1 - (1 - 1 / k) ** (d + 1))


# ---------------------------------------------------------------- generators


def test_edges_to_graph_matches_adja_to_graph():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 12):
        a = np.triu(rng.random((n, n)) < 0.4, 1)
        a = a | a.T
        rows, cols = np.nonzero(np.triu(a, 1))
        g = edges_to_graph(n, np.stack([rows, cols], axis=1).astype(np.int64))
        h = adja_to_graph(a)
        assert g.ptr.tolist() == h.ptr.tolist()
        assert g.ind.tolist() == h.ind.tolist()


@pytest.mark.parametrize("n, p", [(2, 0.5), (15, 0.2), (15, 0.9)])
def test_erdos_renyi_edges_are_distinct_ordered_pairs(n, p):
    set_seed(1)
    for _ in range(20):
        e = erdos_renyi_edges(n, p)
        assert e.shape[1] == 2
        assert np.all((0 <= e[:, 1]) & (e[:, 1] < e[:, 0]) & (e[:, 0] < n))
        assert len({tuple(x) for x in e.tolist()}) == len(e)


def test_erdos_renyi_edges_mean_count():
    set_seed(2)
    n, p, draws = 20, 0.3, 400
    counts = np.array([len(erdos_renyi_edges(n, p)) for _ in range(draws)], dtype=float)
    assert _close(counts, p * n * (n - 1) / 2)


def test_bipartite_edges_cross_the_sides():
    set_seed(3)
    for _ in range(20):
        e = bipartite_edges(4, 6, 0.5)
        assert np.all((0 <= e[:, 0]) & (e[:, 0] < 4) & (4 <= e[:, 1]) & (e[:, 1] < 10))
        assert len({tuple(x) for x in e.tolist()}) == len(e)
    assert len(bipartite_edges(4, 6, 1.0)) == 24
    assert bipartite(3, 0, 0.5).n == 3


def test_cayley_edges_form_a_tree():
    set_seed(4)
    for n in (2, 3, 8, 25):
        e = cayley_edges(n)
        assert e.shape == (n - 1, 2)
        g = edges_to_graph(n, e)
        assert len(g.ind) == 2 * (n - 1)
        seen, stack = {0}, [0]
        while stack:
            for j in g.neighbors(stack.pop()).tolist():
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        assert len(seen) == n


def test_sparse_generators_grow_their_buffer(monkeypatch):
    """With every uniform draw at 0, each gap is 1: all pairs become edges even
    for a small p, which overflows the initial capacity of the edge buffer."""
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    e = erdos_renyi_edges(30, 0.01)
    assert len(e) == 30 * 29 // 2
    assert len({tuple(x) for x in e.tolist()}) == len(e)
    b = bipartite_edges(20, 25, 0.01)
    assert len(b) == 20 * 25
    assert len({tuple(x) for x in b.tolist()}) == len(b)


# -------------------------------------------------------------------- oracle


def test_count_peaks_against_definition():
    rng = np.random.default_rng(5)
    a = np.triu(rng.random((9, 9)) < 0.4, 1)
    a = a | a.T
    g = adja_to_graph(a)
    for _ in range(50):
        colors = rng.integers(0, 3, 9)
        labels = rng.permutation(9).astype(float)
        brute = sum(
            all(
                not (a[i, j] and colors[i] == colors[j] and labels[j] > labels[i])
                for j in range(9)
            )
            for i in range(9)
        )
        assert oracle.count_peaks(g, colors, labels) == brute


def test_sample_graph_is_the_peak_law_on_a_small_graph():
    """On the path P4 with k = 2, the law of T is the enumeration over labelings and colorings."""
    g, n, k = line(4), 4, 2
    law = np.zeros(n + 1)
    for colors in itertools.product(range(k), repeat=n):
        for labels in itertools.permutations(range(n)):
            s = oracle.count_peaks(g, np.array(colors), np.array(labels, dtype=float))
            law[n - s] += 1
    law /= law.sum()
    samples = oracle.sample_graph(g, k=k, trials=20000, seed=6)
    emp = np.bincount(samples, minlength=n + 1) / len(samples)
    assert 0.5 * np.abs(emp - law).sum() < 0.02


@pytest.mark.parametrize("k", [1, 2, 3])
def test_sample_graph_path_mean(k):
    n = 12
    samples = oracle.sample_graph(line(n), k=k, trials=4000, seed=k)
    assert _close(samples, n - (2 * _rho(1, k) + (n - 2) * _rho(2, k)))


@pytest.mark.parametrize("k", [1, 2])
def test_sample_erdos_renyi_mean(k):
    """The node of rank r (from the top) is a peak iff none of the r - 1 larger
    nodes is a same-colored neighbor: probability (1 - p/k)^(r - 1)."""
    n, p = 25, 0.2
    samples = oracle.sample_erdos_renyi(n, p, k=k, trials=3000, seed=10 + k)
    q = p / k
    assert _close(samples, n - (1 - (1 - q) ** n) / q)


def test_sample_bipartite_mean():
    """A node of one side of rank r among that side and j larger nodes on the
    other side is a peak with probability (1 - p/k)^j; averaging over the
    interleaving gives the closed form below."""
    n1, n2, p, k = 8, 12, 0.3, 2
    samples = oracle.sample_bipartite(n1, n2, p, k=k, trials=3000, seed=20)
    q = 1 - p / k
    es = (k / p) * (
        n1 / (n2 + 1) * (1 - q ** (n2 + 1)) + n2 / (n1 + 1) * (1 - q ** (n1 + 1))
    )
    assert _close(samples, n1 + n2 - es)


def test_sample_cayley_mean():
    """Monochrome: a node of degree d is a peak with probability 1/(d+1), and the
    degree is 1 + Binomial(n - 2, 1/n); the expectation sums to (1 - 1/n)^(n - 1)."""
    n = 20
    samples = oracle.sample_cayley(n, trials=3000, seed=30)
    assert _close(samples, n - n * (1 - 1 / n) ** (n - 1))


def test_samples_depend_on_the_seed_only():
    a = oracle.sample_erdos_renyi(15, 0.3, k=2, trials=500, seed=40)
    b = oracle.sample_erdos_renyi(15, 0.3, k=2, trials=500, seed=40)
    assert (a == b).all()
    set_seed(41)
    c = oracle.sample_cayley(10, trials=300), oracle.sample_cayley(10, trials=300)
    set_seed(41)
    d = oracle.sample_cayley(10, trials=300), oracle.sample_cayley(10, trials=300)
    assert all((x == y).all() for x, y in zip(c, d))
    assert (c[0] != c[1]).any()
