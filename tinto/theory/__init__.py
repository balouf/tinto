"""Closed forms of the exact stabilization time of self-stabilizing coloring.

This subpackage follows the results of the papers: it evaluates closed forms
proved there, it does not explore. Every function returns a moment of the
stabilization time ``T = n - S`` of the conflict-guard rule under the random
central daemon, started from a uniform random coloring with `k` colors and a
palette of at least ``Delta + 1`` colors; `S` is the number of peaks of a
uniform random labeling (see :mod:`tinto.oracle`). As ``T = n - S``, both
have the same variance.

Submodules, one per graph family:

- :mod:`~tinto.theory.general`: per-node quantities shared by the families;
- :mod:`~tinto.theory.path_cycle`: path and cycle;
- :mod:`~tinto.theory.erdos_renyi`: :math:`G(n, p)`;
- :mod:`~tinto.theory.bipartite`: :math:`G(n_1, n_2, p)`;
- :mod:`~tinto.theory.cayley`: uniform labeled tree;
- :mod:`~tinto.theory.star`: star, the graph of largest variance.

Independence: :mod:`tinto.theory` imports at most :mod:`tinto.graphs`, never
the simulation side of tinto (:mod:`tinto.oracle`, :mod:`tinto.runners`...)
nor :mod:`tinto.exact`. The companion checks rely on it: closed forms, exact
enumeration and Monte Carlo are three independent sources.
"""

from tinto.theory import bipartite, cayley, erdos_renyi, general, path_cycle, star

__all__ = ["bipartite", "cayley", "erdos_renyi", "general", "path_cycle", "star"]
