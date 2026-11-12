"""Tests for :mod:`tinto.exact`, and for the exact moments of :mod:`tinto.theory.general`.

The last test confronts two of the three sources of the companion checks:
Theorems 1 and 3 (closed forms, rational arithmetic) against the enumeration
of every labeling and coloring, on every graph with four nodes.
"""

import itertools
from math import comb, factorial

import numpy as np
import pytest

from tinto import exact
from tinto.graphs import edges_to_graph
from tinto.theory import general


def _brute_histogram(n, edges, k):
    hist = np.zeros(n + 1, dtype=np.int64)
    for col in itertools.product(range(k), repeat=n):
        for lab in itertools.permutations(range(n)):
            beaten = set()
            for u, v in edges:
                if col[u] == col[v]:
                    beaten.add(u if lab[u] < lab[v] else v)
            hist[n - len(beaten)] += 1
    return hist


def _stirling2(n, m):
    return sum(
        (-1) ** i * comb(m, i) * (m - i) ** n for i in range(m + 1)
    ) // factorial(m)


@pytest.mark.parametrize("n, k", [(1, 1), (3, 3), (4, 2), (5, 3)])
def test_colorings_cover_all_colorings_once(n, k):
    found = list(exact.colorings(n, k))
    assert len(found) == sum(_stirling2(n, m) for m in range(1, min(n, k) + 1))
    assert sum(w for _, w in found) == k**n


@pytest.mark.parametrize("seed", range(6))
def test_peaks_distribution_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(2, 6)), int(rng.integers(1, 4))
    edges = [
        (u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < 0.5
    ]
    g = edges_to_graph(n, np.array(edges, dtype=np.int64).reshape(-1, 2))
    hist = exact.peaks_distribution(g, k)
    assert hist.sum() == exact.law_size(n, k)
    assert hist.tolist() == _brute_histogram(n, edges, k).tolist()


def test_all_trees_and_all_graphs():
    for n in (2, 3, 5):
        trees = list(exact.all_trees(n))
        assert len(trees) == n ** (n - 2)
        shapes = {
            tuple(
                sorted(
                    tuple(sorted((i, j)))
                    for i in range(n)
                    for j in t.neighbors(i).tolist()
                )
            )
            for t in trees
        }
        assert len(shapes) == len(trees)
    assert sum(1 for _ in exact.all_graphs(4)) == 2**6


@pytest.mark.parametrize("k", [1, 2, 3])
def test_theorems_1_and_3_against_enumeration_on_all_graphs_with_four_nodes(k):
    for g in exact.all_graphs(4):
        assert general.moments(g, k) == exact.factorial_moments(
            exact.peaks_distribution(g, k)
        )
