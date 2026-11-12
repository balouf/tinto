# History

## 0.1.1 (2026-11-12): README fix

- The logo of the README shows on PyPI (absolute URL).
- Software Heritage badge.

## 0.1.0 (2026-11-11): First release

- First release on PyPI.

### Coloring dynamics

- `graphs`: a Numba-compiled CSR graph, the `line` and `ring` builders,
  and three random generators, each with its edge-list variant: uniform
  labeled tree (`cayley`), Erdős-Rényi `G(n, p)` drawn by geometric
  skipping in O(n + m) (`erdos_renyi`), and bipartite `G(n1, n2, p)`
  (`bipartite`). Graphs are also built from an edge list
  (`edges_to_graph`) or an adjacency matrix (`adja_to_graph`).
- `guards`: four enabling rules, from the default `color_clash` (a node
  shares its color with a neighbor) to `color_clash_tie`, `color_min`
  and `color_min_tie`.
- `configs.make`: the coloring configuration, which keeps the set of
  enabled nodes in sync with the colors. Initial colorings may use fewer
  colors than the palette (`Config.color_randomize(k_init)`); without an
  explicit palette, it has `max(k_init, Delta + 1)` colors.
- `starters`: initial configurations, either on a fixed graph
  (`from_graph`, uniform or random coloring) or on a fresh random graph
  per draw (`from_cayley`, `from_erdos_renyi`, `from_bipartite`); every
  starter takes a palette size `k` and a number of initial colors
  `k_init`.
- `schedulers`: `shuffled`, `ordered`, `bernoulli` and `independent`.
- `solvers.make`, with five colorers: `colorer_lowest`, `colorer_next`
  and `colorer_random` pick a free color (smallest, next, uniform);
  `colorer_random_blind` and `colorer_next_blind` skip the neighborhood
  scan and may leave clashes behind, modelling agents that detect a
  conflict locally without seeing which colors are free.
- `runners`: `simple` runs one trajectory and returns its numbers of
  steps and activations (recolorings); `mc` runs replicas in parallel
  and returns the mean, variance and third central moment of both.
- The factories are memoized: two calls with the same arguments return
  the same compiled object, so parameter sweeps compile each variant
  once.
- `io.set_seed` seeds every random generator, Numba's included.

### Experiments and estimation

- `tinto.Experiment`, `tinto.Iterator` and `tinto.ProductIterator`:
  parameter sweeps (cartesian products with `*`, campaigns merged with
  `+`), with `.run()`, `.plot()`, `.fit()`, `.fit_fractions()`,
  `.report()`, `.residuals()`, `.to_dict()` and `.to_csv()`.
- `estimate`: weighted least squares on a linear basis (`wls`, `fit`),
  a likelihood-ratio test for a hidden term (`test_extra_term`), and the
  recovery of exact fractions from fitted coefficients (`simplest_in`,
  `recover`, `fit_fractions`, `fractions_table`).

### Exact stabilization time

Three independent sources for the stabilization time of the
conflict-guard rule under the random central daemon:

- `oracle`, the peak oracle: by the reduction lemma, `T = n - S` with `S`
  the number of peaks of a uniform labeling, so `T` is sampled without
  running the dynamics. `count_peaks`, and parallel samplers on a fixed
  graph (`sample_graph`) or on a fresh random graph per trial
  (`sample_erdos_renyi`, `sample_bipartite`, `sample_cayley`); the
  samples depend on the seed only, not on the number of threads.
- `exact`: the exact law of the peak count on small graphs, by
  enumeration of all labelings and colorings (`peaks_distribution`,
  colorings up to a permutation of the colors, optional split on whether
  two nodes share their color, optional fixed labeling), its factorial
  moments, and the enumeration of all labeled trees and all graphs on
  `n` nodes.
- `theory`: closed forms, one submodule per family:
  - `general`: exact mean and variance on any graph (Theorems 1 and 3)
    in rational arithmetic, the mean from a fixed initial coloring, and
    a compiled floating-point monochrome variance for loops over
    millions of graphs;
  - `path_cycle`: generating functions in exact integers, exact laws and
    closed forms;
  - `erdos_renyi`: exact moments and fixed-`p` limit law;
  - `bipartite`: monochrome law and limit mean;
  - `cayley`: pair-degree laws, asymptotic variance slope and the
    generating function of the law on the uniform labeled tree;
  - `star`: law, mean and variance of the star, the graph of largest
    variance.

  `theory` imports nothing from the simulation side, nor from `exact`;
  `tests/test_independence.py` enforces it.

### Documentation

- Numpydoc docstrings with doctests on every public module, class and
  function; 100% branch coverage under `NUMBA_DISABLE_JIT=1`.
- Sphinx reference pages for every module.
- Two tutorials: `introduction`, a tour of the modules with a
  `verbose_run` helper that walks one step at a time through the
  dynamics, and `basics`, a first set of experiments.
- Companion notebooks, starting with
  `docs/companion/exact_stabilization_time.ipynb`: the numerical
  statements of the exact stabilization-time report, each checked by
  confronting closed forms, exact enumeration and Monte Carlo (sympy is
  in the `dev` group for its symbolic sections). Notebooks that assert
  checks are stored without outputs; the docs workflow executes them
  before building, so a failing check fails the build.
