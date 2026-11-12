# <img src="https://balouf.github.io/tinto/_static/logo.svg" alt="Tinto logo" width="48" valign="middle" /> Tinto


[![PyPI Status](https://img.shields.io/pypi/v/tinto.svg)](https://pypi.python.org/pypi/tinto)
[![Build Status](https://github.com/balouf/tinto/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/balouf/tinto/actions?query=workflow%3Abuild)
[![Documentation Status](https://github.com/balouf/tinto/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/balouf/tinto/actions?query=workflow%3Adocs)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://codecov.io/gh/balouf/tinto/branch/main/graphs/badge.svg)](https://codecov.io/gh/balouf/tinto/tree/main)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/balouf/tinto/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/balouf/tinto)

How long does a self-stabilizing graph coloring take to settle? Tinto measures it with a Numba-compiled Monte Carlo over pluggable graphs, guards, schedulers and recoloring rules, down to the variance and skewness of the stabilization time. For the conflict-guard rule under the random central daemon, it also provides exact results.


- Free software: MIT
- Documentation: <https://balouf.github.io/tinto/>.
- Github: <https://github.com/balouf/tinto>


## Features

- Numba-compiled dynamics: CSR graphs (line, ring, uniform labeled tree,
  Erdős-Rényi `G(n, p)`, bipartite `G(n1, n2, p)`), pluggable guards,
  four schedulers (shuffled, ordered, Bernoulli, independent) and five
  colorers (lowest, next or uniform free color, plus two blind variants
  that may leave clashes behind).
- Monte-Carlo driver aggregating mean, variance and third central moment
  of both the number of steps and the number of activations before
  stabilization, on a fixed graph or on a fresh random graph per replica.
- `Experiment` high-level API with cartesian-product sweeps, campaign
  addition, light matplotlib plotting, CSV export, and fits of the
  results (weighted least squares, test for a hidden term, recovery of
  exact fractions).
- Exact stabilization time of the conflict-guard rule under the random
  central daemon, from three independent sources: the peak oracle, which
  samples it without running the dynamics (`tinto.oracle`), exact
  enumeration on small graphs (`tinto.exact`) and closed forms
  (`tinto.theory`).
- Companion notebooks that check the numerical statements of the papers.
- 100% branch-covered by doctest-based examples.

## Quickstart

Install Tinto:

```console
$ pip install tinto
```

### Low-level API

Drive one trajectory on a line graph:

```pycon
>>> from tinto import graphs, configs, schedulers, solvers, runners
>>> from tinto.io import set_seed
>>> set_seed(0)
>>> Config = configs.make()
>>> c = Config(graphs.line(20))
>>> sched = schedulers.shuffled()(c)
>>> solver = solvers.make()
>>> steps, activations = runners.simple(c, sched, solver)
>>> steps > 0
True
>>> c.total
0
```

Or Monte-Carlo the moments of both metrics over many replicas:

```pycon
>>> from tinto import starters
>>> set_seed(0)
>>> starter = starters.from_graph(Config)(graphs.line(20))
>>> res = runners.mc(starter, schedulers.shuffled(), solver, n_trials=20)
>>> mean, var, m3 = res["steps"]
>>> mean > 0
True
```

### High-level API

The `Experiment` class bundles a sweep + the Monte-Carlo runner + plot
and export helpers. Define an `Iterator` over one parameter, wrap it in
an `Experiment`, run:

```pycon
>>> from tinto import Experiment, Iterator
>>> def build_starter(n):
...     return starters.from_graph(Config)(graphs.line(n))
>>> n_iter = Iterator("starter", [10, 20, 50], name="n", process=build_starter)
>>> exp = Experiment(
...     "line-sweep",
...     iterator=n_iter,
...     scheduler=schedulers.shuffled(),
...     solver=solver,
...     n_trials=50,
... )
>>> _ = exp.run(progress=False)
>>> exp.results["line-sweep"]["axes"]
('starter',)
```

Compose cartesian products with `*` and combine multiple campaigns with
`+`:

```pycon
>>> kind_iter = Iterator("starter", ["random", "mono"], name="kind")
>>> # plot or export via exp.plot(x="n", hue="kind") / exp.to_csv(...)
```

## Credits

This package was created with [Cookiecutter][CC] and the [Package Helper 3][PH3] project template.

[CC]: <https://github.com/audreyr/cookiecutter>
[PH3]: <https://balouf.github.io/package-helper-3/>
