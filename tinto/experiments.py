"""High-level experiment API built on top of :mod:`tinto.runners`.

This module consolidates the notebook boilerplate of parameter sweeps +
Monte-Carlo aggregation + plotting into a single :class:`Experiment`
object. The design is deliberately thin: the heavy lifting stays in
:mod:`tinto.runners` and the Config/Graph/scheduler/solver stack, while
:class:`Experiment` adds three amenities:

- *Sweep composition*: a 1-D :class:`Iterator` for one varying parameter,
  and a cartesian :class:`ProductIterator` obtained via ``*`` for joint
  sweeps.
- *Campaign aggregation*: ``Experiment + Experiment`` (and therefore
  :func:`sum`) merges campaigns into a single object.
- *Result handling*: :meth:`Experiment.run` fills
  :attr:`Experiment.results`, a structured nested dict. From there,
  :meth:`Experiment.plot`, :meth:`Experiment.to_dict` and
  :meth:`Experiment.to_csv` expose the common consumption paths.

Matplotlib and :mod:`tqdm` are imported lazily; neither is a hard
dependency.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator as _PyIterator


class Iterator:
    """One-dimensional parameter sweep.

    Yields, at each step, a triplet ``(kwargs, display_names,
    display_values)`` where `kwargs` is a single-entry dict ready to be
    passed as keyword arguments to the runner, and the two tuples carry
    the display metadata used by :meth:`Experiment.plot` and
    :meth:`Experiment.to_csv`. For a 1-D iterator both tuples have
    length one; for a :class:`ProductIterator` obtained via ``*`` they
    have one entry per dimension.

    Parameters
    ----------
    parameter: :class:`str`
        Keyword argument name the runner expects.
    values: iterable
        Raw values to sweep. Stored as a list.
    name: :class:`str`, optional
        Human-readable axis label for plots. Defaults to `parameter`.
    process: callable, optional
        Maps each raw value to the object actually passed to the runner
        (e.g. build a ``Starter`` from an integer `n`). Defaults to
        identity.

    Examples
    --------
    >>> it = Iterator("n", [1, 2, 3])
    >>> len(it)
    3
    >>> list(it) == [
    ...     ({"n": 1}, ("n",), (1,)),
    ...     ({"n": 2}, ("n",), (2,)),
    ...     ({"n": 3}, ("n",), (3,)),
    ... ]
    True

    A custom ``name`` and ``process`` let the display label decouple
    from the runner keyword:

    >>> it2 = Iterator("n", [2, 4], name="$n$", process=lambda v: v * 10)
    >>> [kwargs for (kwargs, _, _) in it2]
    [{'n': 20}, {'n': 40}]
    >>> [names for (_, names, _) in it2]
    [('$n$',), ('$n$',)]
    """

    def __init__(
        self,
        parameter: str,
        values: Iterable[Any],
        name: str | None = None,
        process: Callable[[Any], Any] | None = None,
    ):
        self.parameter = parameter
        self.values = list(values)
        self.name = parameter if name is None else name
        self.process = (lambda x: x) if process is None else process

    def __iter__(self) -> _PyIterator[tuple[dict, tuple, tuple]]:
        for v in self.values:
            yield {self.parameter: self.process(v)}, (self.name,), (v,)

    def __len__(self) -> int:
        return len(self.values)

    def __mul__(self, other: "Iterator | ProductIterator") -> "ProductIterator":
        """Combine with another iterator into a cartesian product.

        Examples
        --------
        >>> (Iterator("n", [1, 2]) * Iterator("p", [0.1, 0.5])).__class__.__name__
        'ProductIterator'

        Right-associativity flattens nested :class:`ProductIterator`:

        >>> prod = Iterator("a", [1]) * (Iterator("b", [2]) * Iterator("c", [3]))
        >>> [it.parameter for it in prod.iterators]
        ['a', 'b', 'c']

        Multiplying with a non-iterator raises :class:`TypeError`:

        >>> Iterator("n", [1]) * 5
        Traceback (most recent call last):
            ...
        TypeError: unsupported operand type(s) for *: 'Iterator' and 'int'
        """
        if isinstance(other, ProductIterator):
            return ProductIterator([self, *other.iterators])
        if isinstance(other, Iterator):
            return ProductIterator([self, other])
        return NotImplemented


class ProductIterator:
    """Cartesian product of :class:`Iterator` (and/or :class:`ProductIterator`) instances.

    Yields the same triplet as :class:`Iterator` but with one entry per
    dimension, in left-to-right order. The rightmost dimension varies
    fastest (standard :func:`itertools.product` behavior).

    Parameters
    ----------
    iterators: list of :class:`Iterator` or :class:`ProductIterator`
        Dimensions of the product. Stored flat: passing a
        :class:`ProductIterator` in the list merges its inner iterators.

    Examples
    --------
    >>> it_n = Iterator("n", [1, 2])
    >>> it_p = Iterator("p", [0.1, 0.5])
    >>> prod = it_n * it_p
    >>> len(prod)
    4
    >>> [values for (_, _, values) in prod]
    [(1, 0.1), (1, 0.5), (2, 0.1), (2, 0.5)]

    Multiplication chains without nesting:

    >>> prod3 = it_n * it_p * Iterator("k", [10])
    >>> [it.parameter for it in prod3.iterators]
    ['n', 'p', 'k']
    >>> len(prod3)
    4

    Direct construction with a nested :class:`ProductIterator` in the
    list flattens it transparently:

    >>> nested = ProductIterator([prod, Iterator("w", [0])])
    >>> [it.parameter for it in nested.iterators]
    ['n', 'p', 'w']
    """

    def __init__(self, iterators: Iterable):
        flat: list = []
        for it in iterators:
            if isinstance(it, ProductIterator):
                flat.extend(it.iterators)
            else:
                flat.append(it)
        self.iterators = flat

    def __iter__(self) -> _PyIterator[tuple[dict, tuple, tuple]]:
        for combo in itertools.product(*[list(it) for it in self.iterators]):
            merged: dict = {}
            names: list = []
            values: list = []
            for kw, ns, vs in combo:
                merged.update(kw)
                names.extend(ns)
                values.extend(vs)
            yield merged, tuple(names), tuple(values)

    def __len__(self) -> int:
        total = 1
        for it in self.iterators:
            total *= len(it)
        return total

    def __mul__(self, other):
        """Extend the product by another iterator or product.

        Examples
        --------
        >>> prod = Iterator("a", [1]) * Iterator("b", [2])
        >>> joined = prod * (Iterator("c", [3]) * Iterator("d", [4]))
        >>> [it.parameter for it in joined.iterators]
        ['a', 'b', 'c', 'd']
        >>> prod * 0
        Traceback (most recent call last):
            ...
        TypeError: unsupported operand type(s) for *: 'ProductIterator' and 'int'
        """
        if isinstance(other, ProductIterator):
            return ProductIterator([*self.iterators, *other.iterators])
        if isinstance(other, Iterator):
            return ProductIterator([*self.iterators, other])
        return NotImplemented


@dataclass
class _Campaign:
    """Internal record holding one campaign's inputs."""

    name: str | None
    iterator: Iterator | ProductIterator | None
    params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return 1 if self.iterator is None else len(self.iterator)


class Experiment:
    """High-level entry point binding sweeps to a Monte-Carlo runner.

    An :class:`Experiment` bundles one or more *campaigns*. Each campaign
    pairs a name, an optional :class:`Iterator` (or
    :class:`ProductIterator`) sweeping one or several parameters, and a
    dict of fixed keyword arguments forwarded to the runner. Combining
    experiments with ``+`` (or :func:`sum`) concatenates their
    campaigns; :meth:`run` executes every campaign sequentially and
    stores the per-replica moments under :attr:`results`.

    Parameters
    ----------
    name: :class:`str`, optional
        Campaign label, used as key in :attr:`results` and as plot
        legend.
    iterator: :class:`Iterator` or :class:`ProductIterator`, optional
        Sweep definition. If ``None``, the campaign is a single point
        (useful only when combined with others).
    **params: keyword arguments
        Runner arguments that stay fixed across the sweep (typically
        ``starter``, ``scheduler``, ``solver``, ``n_trials``).

    Examples
    --------
    A minimal 1-D sweep evaluated with an injected dummy runner:

    >>> def dummy(n):
    ...     return {"steps": (float(n), 0.0, 0.0), "activations": (0.0, 0.0, 0.0)}
    >>> exp = Experiment("demo", iterator=Iterator("n", [1, 2, 3]))
    >>> _ = exp.run(runner=dummy, progress=False)
    >>> exp.results["demo"]["axes"]
    ('n',)
    >>> exp.results["demo"]["data"][(2,)]["steps"]
    (2.0, 0.0, 0.0)

    Two campaigns combined with ``+`` keep distinct entries in
    :attr:`results`:

    >>> a = Experiment("A", iterator=Iterator("n", [1]))
    >>> b = Experiment("B", iterator=Iterator("n", [2]))
    >>> sorted((a + b).to_dict_keys())
    ['A', 'B']
    """

    def __init__(
        self,
        name: str | None = None,
        iterator: Iterator | ProductIterator | None = None,
        **params: Any,
    ):
        self._campaigns: list[_Campaign] = [_Campaign(name, iterator, params)]
        self._results: dict | None = None

    def to_dict_keys(self):
        return [c.name for c in self._campaigns]

    def __add__(self, other: "Experiment | int") -> "Experiment":
        """Combine two :class:`Experiment` instances.

        Supports ``sum([exp1, exp2, ...])`` by accepting an integer
        ``0`` from :func:`sum`'s initial accumulator.

        Examples
        --------
        >>> a = Experiment("A", iterator=Iterator("n", [1, 2]))
        >>> b = Experiment("B", iterator=Iterator("n", [3]))
        >>> combined = a + b
        >>> len(combined)
        3
        >>> sum([a, b]).to_dict_keys()
        ['A', 'B']

        Adding anything other than an :class:`Experiment` or ``0`` is a
        :class:`TypeError`:

        >>> a + "foo"
        Traceback (most recent call last):
            ...
        TypeError: unsupported operand type(s) for +: 'Experiment' and 'str'
        """
        if other == 0:
            return self
        if not isinstance(other, Experiment):
            return NotImplemented
        merged = Experiment.__new__(Experiment)
        merged._campaigns = self._campaigns + other._campaigns
        merged._results = None
        return merged

    def __radd__(self, other: "Experiment | int") -> "Experiment":
        return self + other

    def __len__(self) -> int:
        return sum(len(c) for c in self._campaigns)

    def __iter__(self):
        """Yield ``(campaign_name, display_names, display_values, params)`` tuples.

        Examples
        --------
        >>> exp = Experiment("demo", iterator=Iterator("n", [1, 2]), k=3)
        >>> for name, names, values, params in exp:
        ...     print(name, names, values, params)
        demo ('n',) (1,) {'k': 3, 'n': 1}
        demo ('n',) (2,) {'k': 3, 'n': 2}

        A campaign without iterator yields a single tuple with empty
        display slots:

        >>> list(Experiment("bare", k=5))
        [('bare', (), (), {'k': 5})]
        """
        for c in self._campaigns:
            if c.iterator is None:
                yield c.name, (), (), dict(c.params)
            else:
                for kwargs, names, values in c.iterator:
                    yield c.name, names, values, {**c.params, **kwargs}

    def run(
        self,
        runner: Callable | None = None,
        progress: bool = True,
    ) -> "Experiment":
        """Execute every campaign sequentially and fill :attr:`results`.

        Parameters
        ----------
        runner: callable, optional
            Function called with the per-point parameters and returning
            a dict ``{metric: (mean, var, m3)}``. Defaults to
            :func:`tinto.runners.mc`.
        progress: :class:`bool`, default=True
            Show a :mod:`tqdm` progress bar when :mod:`tqdm` is
            installed. Silently ignored otherwise.

        Returns
        -------
        :class:`Experiment`
            Self, for chaining.

        Examples
        --------
        With the default runner (:func:`tinto.runners.mc`) on a small
        line-graph campaign with no sweep:

        >>> from tinto.graphs import line
        >>> from tinto.configs import make as make_config
        >>> from tinto.starters import from_graph
        >>> from tinto.schedulers import shuffled
        >>> from tinto.solvers import make as make_solver
        >>> from tinto.io import set_seed
        >>> set_seed(0)
        >>> Config = make_config()
        >>> starter = from_graph(Config, color=0)(line(5))
        >>> exp = Experiment(
        ...     "bare",
        ...     starter=starter,
        ...     scheduler=shuffled(),
        ...     solver=make_solver(),
        ...     n_trials=5,
        ... )
        >>> _ = exp.run()
        >>> sorted(exp.results["bare"]["data"][()].keys())
        ['activations', 'steps']

        :attr:`results` raises if accessed before :meth:`run`:

        >>> Experiment("untouched").results
        Traceback (most recent call last):
            ...
        RuntimeError: Experiment.run() must be called before accessing results.
        """
        if runner is None:
            from tinto import runners

            runner = runners.mc

        if progress:
            try:
                from tqdm.auto import tqdm as _bar
            except ImportError:  # pragma: no cover
                _bar = lambda x, **kw: x  # noqa: E731
        else:
            _bar = lambda x, **kw: x  # noqa: E731

        self._results = {}
        for c in self._campaigns:
            if c.iterator is None:
                metrics = runner(**c.params)
                self._results[c.name] = {
                    "axes": (),
                    "display_names": (),
                    "n_trials": c.params.get("n_trials"),
                    "data": {(): metrics},
                }
                continue

            first = next(iter(c.iterator))
            axes = tuple(first[0].keys())
            display_names = first[1]
            data: dict[tuple, dict] = {}
            for kwargs, _names, values in _bar(
                c.iterator, total=len(c.iterator), desc=c.name or ""
            ):
                data[values] = runner(**{**c.params, **kwargs})
            self._results[c.name] = {
                "axes": axes,
                "display_names": display_names,
                "n_trials": c.params.get("n_trials"),
                "data": data,
            }
        return self

    @property
    def results(self) -> dict:
        """Structured per-campaign result dict filled by :meth:`run`."""
        if self._results is None:
            raise RuntimeError(
                "Experiment.run() must be called before accessing results."
            )
        return self._results

    def _select(self, campaign):
        """Return an ordered ``{name: camp}`` dict filtered by `campaign`.

        `campaign` is ``None`` (all), a string (one), or an iterable of
        strings (a subset). Unknown names are dropped silently.
        """
        results = self.results
        if campaign is None:
            return dict(results)
        if isinstance(campaign, str):
            return {campaign: results[campaign]} if campaign in results else {}
        return {n: results[n] for n in campaign if n in results}

    def _infer_x(self, x, hue, selected):
        """Resolve an omitted `x` when the selected campaigns leave no ambiguity.

        Collects every axis appearing in the selected campaigns, drops
        `hue` if present, and returns the unique remaining axis if any.
        Otherwise leaves `x` untouched.
        """
        if x is not None:
            return x
        candidates = {a for camp in selected.values() for a in camp["axes"]}
        candidates.discard(hue)
        if len(candidates) == 1:
            return next(iter(candidates))
        return x

    def series(
        self,
        metric: str = "steps",
        stat: str = "mean",
        x: str | None = None,
        hue: str | None = None,
        campaign: str | Iterable[str] | None = None,
    ) -> dict[str, tuple[list, list]]:
        """Extract ``{label: (xs, ys)}`` without any plotting.

        Same selection logic as :meth:`plot`: campaigns missing `x`, a
        `hue` not in the axes, or a 2-D campaign without a `hue` are
        skipped. Labels are ``campaign_name`` for 1-D campaigns,
        ``"{campaign_name} / {hue}={hue_value}"`` for 2-D campaigns
        split by `hue`.

        When `x` is omitted, it is inferred from the selected campaigns:
        if exactly one axis (excluding `hue`) appears across them, it is
        used. If zero or more than one, the call behaves as if `x` was
        unknown (empty result).

        Parameters
        ----------
        metric, stat, x, hue: see :meth:`plot`.
        campaign: :class:`str`, iterable of strings, or ``None``,
            default=``None``
            Filter on campaign name(s). ``None`` keeps all, a string
            keeps one, an iterable keeps a subset. Unknown names are
            dropped silently.

        Returns
        -------
        dict
            Insertion-ordered mapping ``{label: (xs, ys)}``. Both `xs`
            and `ys` are plain Python lists, sorted by `x`.

        Examples
        --------
        >>> def dummy(n):
        ...     return {"steps": (float(n), 0.0, 0.0), "activations": (float(n * 2), 0.0, 0.0)}
        >>> a = Experiment("A", iterator=Iterator("n", [1, 2, 3]))
        >>> b = Experiment("B", iterator=Iterator("n", [1, 2, 3]))
        >>> exp = (a + b).run(runner=dummy, progress=False)
        >>> exp.series(x="n", metric="steps")
        {'A': ([1, 2, 3], [1.0, 2.0, 3.0]), 'B': ([1, 2, 3], [1.0, 2.0, 3.0])}

        Selecting a single campaign:

        >>> exp.series(x="n", metric="activations", campaign="A")
        {'A': ([1, 2, 3], [2.0, 4.0, 6.0])}

        Or a subset. Unknown names silently drop:

        >>> exp.series(x="n", campaign=["B", "ghost"])
        {'B': ([1, 2, 3], [1.0, 2.0, 3.0])}

        With `hue` on a cartesian-product campaign, labels combine
        campaign name and hue value:

        >>> def dummy2(n, kind):
        ...     return {"steps": (float(n) + {"a": 0.0, "b": 1.0}[kind], 0.0, 0.0),
        ...             "activations": (0.0, 0.0, 0.0)}
        >>> prod = Experiment(
        ...     "P", iterator=Iterator("n", [1, 2]) * Iterator("kind", ["a", "b"])
        ... )
        >>> _ = prod.run(runner=dummy2, progress=False)
        >>> prod.series(x="n", hue="kind")
        {'P / kind=a': ([1, 2], [1.0, 2.0]), 'P / kind=b': ([1, 2], [2.0, 3.0])}

        `x` is inferred when unambiguous, so passing it is optional on
        simple 1-D campaigns:

        >>> exp.series() == exp.series(x="n")
        True
        """
        stat_idx = {"mean": 0, "var": 1, "m3": 2}[stat]
        selected = self._select(campaign)
        x = self._infer_x(x, hue, selected)
        out: dict[str, tuple[list, list]] = {}
        for camp_name, camp in selected.items():
            axes = camp["axes"]
            data = camp["data"]
            if x is None or x not in axes:
                continue
            ix = axes.index(x)

            if hue is not None:
                if hue not in axes:
                    continue
                ih = axes.index(hue)
                groups: dict = {}
                for key, m in data.items():
                    groups.setdefault(key[ih], []).append((key[ix], m))
                for hval, pairs in sorted(groups.items()):
                    pairs.sort(key=lambda t: t[0])
                    xs = [p[0] for p in pairs]
                    ys = [p[1][metric][stat_idx] for p in pairs]
                    prefix = f"{camp_name} / " if camp_name else ""
                    out[f"{prefix}{hue}={hval}"] = (xs, ys)
            else:
                if len(axes) > 1:
                    continue
                pairs = sorted((key[ix], m) for key, m in data.items())
                xs = [p[0] for p in pairs]
                ys = [p[1][metric][stat_idx] for p in pairs]
                out[camp_name] = (xs, ys)
        return out

    def plot(
        self,
        metric: str = "steps",
        stat: str = "mean",
        x: str | None = None,
        hue: str | None = None,
        campaign: str | Iterable[str] | None = None,
        scale: str = "linear",
        ax=None,
        **plot_kwargs,
    ):
        """Light matplotlib rendering of a metric across campaigns.

        Thin wrapper around :meth:`series`: pulls the ``{label: (xs,
        ys)}`` dict and renders one line per entry on `ax`.

        Parameters
        ----------
        metric: :class:`str`, default='steps'
            Key into the runner's output dict (``"steps"`` or
            ``"activations"`` when using :func:`tinto.runners.mc`).
        stat: :class:`str`, default='mean'
            One of ``"mean"``, ``"var"``, ``"m3"``; indexes the
            3-tuple returned by the runner.
        x: :class:`str`, optional
            Name of the `parameter` to place on the x-axis. When
            omitted, inferred from the selected campaigns: if exactly
            one axis (excluding `hue`) appears, it is used. Campaigns
            missing the resolved axis are skipped.
        hue: :class:`str`, optional
            Name of a second `parameter` to use as series. If ``None``,
            each campaign contributes a single line.
        campaign: :class:`str`, iterable of strings, or ``None``,
            default=``None``
            Restrict to a subset of campaigns. See :meth:`series`.
        scale: :class:`str`, default='linear'
            One of ``"linear"``, ``"logx"``, ``"logy"``, ``"loglog"``.
            Applied via :meth:`~matplotlib.axes.Axes.set_xscale` /
            :meth:`~matplotlib.axes.Axes.set_yscale` after the lines
            are drawn.
        ax: :class:`matplotlib.axes.Axes`, optional
            Existing axes. Created via :func:`matplotlib.pyplot.gca`
            when ``None``.
        **plot_kwargs: keyword arguments
            Forwarded to :meth:`matplotlib.axes.Axes.plot`. User kwargs
            take precedence over the ones set internally (notably
            ``label``), so passing ``label="foo"`` overrides the
            auto-generated labels — shared across every series, which
            produces a duplicated legend when more than one line is
            drawn. For per-series customization, prefer :meth:`series`
            + a manual matplotlib loop.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            The axes drawn on.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> def dummy(n):
        ...     return {"steps": (float(n), 0.0, 0.0), "activations": (0.0, 0.0, 0.0)}
        >>> exp = Experiment("demo", iterator=Iterator("n", [1, 2, 3]))
        >>> _ = exp.run(runner=dummy, progress=False)
        >>> fig, ax = plt.subplots()
        >>> _ = exp.plot(x="n", ax=ax)
        >>> ax.get_xlabel(), ax.get_ylabel()
        ('n', 'steps (mean)')
        >>> len(ax.lines)
        1
        >>> plt.close(fig)

        Cartesian-product campaigns light up multiple series via `hue`:

        >>> def dummy2(n, kind):
        ...     return {"steps": (float(n) + {"a": 0.0, "b": 1.0}[kind], 0.0, 0.0),
        ...             "activations": (0.0, 0.0, 0.0)}
        >>> exp2 = Experiment(
        ...     "multi",
        ...     iterator=Iterator("n", [1, 2]) * Iterator("kind", ["a", "b"]),
        ... )
        >>> _ = exp2.run(runner=dummy2, progress=False)
        >>> fig, ax = plt.subplots()
        >>> _ = exp2.plot(x="n", hue="kind", ax=ax)
        >>> len(ax.lines)
        2
        >>> plt.close(fig)

        When `x` is omitted it is inferred from the campaign axes.
        Passing `hue` alone is enough to disambiguate a 2-D campaign:

        >>> fig, ax = plt.subplots()
        >>> _ = exp.plot(ax=ax)
        >>> _ = exp2.plot(hue="kind", ax=ax)
        >>> [line.get_label() for line in ax.lines]
        ['demo', 'multi / kind=a', 'multi / kind=b']
        >>> plt.close(fig)

        Ambiguous axes (two candidates, no `hue`), explicit `x` not in
        axes, `hue` not in axes, or a 2-D campaign with `x` but no
        `hue`, all skip silently:

        >>> fig, ax = plt.subplots()
        >>> _ = exp2.plot(ax=ax)
        >>> _ = exp.plot(x="unknown", ax=ax)
        >>> _ = exp.plot(x="n", hue="unknown", ax=ax)
        >>> _ = exp2.plot(x="n", ax=ax)
        >>> len(ax.lines)
        0
        >>> plt.close(fig)

        Filter to one campaign and pick a log-log scale. When `ax` is
        omitted, :func:`matplotlib.pyplot.gca` is used:

        >>> bis = Experiment("bis", iterator=Iterator("n", [1, 2, 3, 4]))
        >>> both = (exp + bis).run(runner=dummy, progress=False)
        >>> ax = both.plot(x="n", campaign="bis", scale="loglog")
        >>> ax.get_xscale(), ax.get_yscale()
        ('log', 'log')
        >>> len(ax.lines)
        1
        >>> plt.close("all")

        Plot kwargs take precedence over the auto-generated ``label``:

        >>> fig, ax = plt.subplots()
        >>> _ = exp.plot(x="n", label="override", color="red", ax=ax)
        >>> ax.lines[0].get_label()
        'override'
        >>> plt.close(fig)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        xlog = scale in ("logx", "loglog")
        ylog = scale in ("logy", "loglog")

        selected = self._select(campaign)
        x = self._infer_x(x, hue, selected)

        for label, (xs, ys) in self.series(
            metric=metric, stat=stat, x=x, hue=hue, campaign=campaign
        ).items():
            ax.plot(xs, ys, **{"label": label, **plot_kwargs})

        # Set xlabel from the first selected campaign that exposes `x`.
        if x is not None:
            for camp in selected.values():
                if x in camp["axes"]:
                    ax.set_xlabel(camp["display_names"][camp["axes"].index(x)])
                    break
        ax.set_ylabel(f"{metric} ({stat})")

        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        if ax.lines:
            ax.legend()
        return ax

    def fit(
        self,
        metric: str = "steps",
        basis=None,
        x: str | None = None,
        hue: str | None = None,
        campaign: str | Iterable[str] | None = None,
    ) -> dict:
        """Fit a law to a metric's mean across campaigns, with error bars.

        Convenience bridge to :mod:`tinto.estimate`. For each series (same
        ``{label: ...}`` selection as :meth:`series`/:meth:`plot`), it
        pulls the per-point mean and variance, derives the standard error
        of the mean ``sigma_i = sqrt(var_i / (n_trials - 1))`` (the
        ``n_trials / (n_trials - 1)`` term turns the population variance
        returned by :func:`tinto.runners.mc` into the unbiased estimate),
        and runs :func:`tinto.estimate.fit`.

        The default `basis` is affine (``a*x + b``); pass a custom basis
        such as :func:`tinto.estimate.power_basis` for inverse-power laws
        ``f(k) = a/k + b/k**2``. The fitted coefficients can then be turned
        into exact fractions with :func:`tinto.estimate.recover`.

        Parameters
        ----------
        metric: :class:`str`, default='steps'
            ``"steps"`` or ``"activations"``.
        basis: list of callables, optional
            Linear basis forwarded to :func:`tinto.estimate.fit`. Defaults
            to the affine basis.
        x, hue, campaign: see :meth:`series`.

        Returns
        -------
        dict
            ``{label: fit_dict}`` with the same labels as :meth:`series`,
            each value being the dict returned by
            :func:`tinto.estimate.fit`.

        Raises
        ------
        ValueError
            If a selected campaign has no recorded ``n_trials`` (e.g. it
            was run with a custom runner that does not take it).

        Examples
        --------
        A dummy runner with exactly affine means and unit variance: the
        fit recovers the slope and intercept.

        >>> import numpy as np
        >>> def dummy(n, n_trials):
        ...     return {"steps": (2.0 * n + 1, 1.0, 0.0),
        ...             "activations": (0.0, 1.0, 0.0)}
        >>> exp = Experiment("demo", iterator=Iterator("n", [1, 2, 3]), n_trials=100)
        >>> _ = exp.run(runner=dummy, progress=False)
        >>> r = exp.fit(metric="steps")
        >>> np.round(r["demo"]["beta"], 6).tolist()
        [2.0, 1.0]
        >>> r["demo"]["dof"]
        1
        """
        from tinto import estimate

        return {
            label: estimate.fit(xs, ys, sigma, basis)
            for label, xs, ys, sigma in self._xy_sigma(metric, x, hue, campaign)
        }

    def fit_fractions(
        self,
        metric: str = "steps",
        basis=None,
        x: str | None = None,
        hue: str | None = None,
        campaign: str | Iterable[str] | None = None,
        **kwargs,
    ) -> dict:
        """Recover exact fractions for a metric's law, coefficient by coefficient.

        Same ``(x, y, sigma)`` assembly as :meth:`fit`, but forwarded to
        :func:`tinto.estimate.fit_fractions`, which jointly fits, anchors
        the surest coefficient as a fraction, and refits the rest (see
        that function for the anchor-and-refit rationale). Extra keyword
        arguments (``k``, ``max_denominator``, ``resolution``, ...) pass
        straight through.

        Parameters
        ----------
        metric, basis, x, hue, campaign: see :meth:`fit`.
        **kwargs:
            Forwarded to :func:`tinto.estimate.fit_fractions`.

        Returns
        -------
        dict
            ``{label: result}`` with the same labels as :meth:`series`,
            each value the dict returned by
            :func:`tinto.estimate.fit_fractions`.

        Examples
        --------
        >>> def dummy(n, n_trials):
        ...     return {"steps": ((2 / 3) * n - 1 / 3, 1.0, 0.0),
        ...             "activations": (0.0, 1.0, 0.0)}
        >>> exp = Experiment("demo", iterator=Iterator("n", range(10, 60)), n_trials=10**6)
        >>> _ = exp.run(runner=dummy, progress=False)
        >>> exp.fit_fractions(metric="steps")["demo"]["fractions"]
        [Fraction(2, 3), Fraction(-1, 3)]
        """
        from tinto import estimate

        return {
            label: estimate.fit_fractions(xs, ys, sigma, basis, **kwargs)
            for label, xs, ys, sigma in self._xy_sigma(metric, x, hue, campaign)
        }

    def report(
        self,
        metric: str = "steps",
        basis=None,
        names=None,
        x: str | None = None,
        hue: str | None = None,
        campaign: str | Iterable[str] | None = None,
        **kwargs,
    ):
        """Tabulate recovered fractions across campaigns, ready to display.

        Runs :meth:`fit_fractions` and hands the result to
        :func:`tinto.estimate.fractions_table`, which returns a table that
        prints as aligned ASCII in a terminal and renders as HTML in a
        notebook (Jupyter calls its ``_repr_html_`` automatically — the
        same ``.auto`` spirit as :mod:`tqdm`, with no environment
        sniffing). Each row shows, per coefficient, either ``value ->
        fraction`` (anchored) or ``value +/- error`` (left free), plus
        ``chi2_red``, ``pvalue`` and a one-word fit verdict; a ``high``
        verdict means the fractions are not to be trusted.

        Coefficient names default to :func:`tinto.estimate.basis_names`
        applied to `basis` — ``["1", "x"]`` for the affine default,
        ``["x^-1", "x^-2"]`` for ``power_basis([-1, -2])``, positional
        ``c0, c1, ...`` for an anonymous custom basis. Override with
        `names`.

        Parameters
        ----------
        metric, basis, x, hue, campaign: see :meth:`fit_fractions`.
        names: list of :class:`str`, optional
            Column names for the coefficients. Defaults to
            :func:`tinto.estimate.basis_names` of `basis`.
        **kwargs:
            Forwarded to :meth:`fit_fractions` (``k``, ``max_denominator``,
            ``resolution``, ...).

        Returns
        -------
        :class:`tinto.estimate._FractionTable`
            Printable / notebook-displayable table.

        Examples
        --------
        >>> def dummy(n, n_trials):
        ...     return {"steps": ((2 / 3) * n - 1 / 3, 1.0, 0.0),
        ...             "activations": (0.0, 1.0, 0.0)}
        >>> exp = Experiment("demo", iterator=Iterator("n", range(10, 60)), n_trials=10**6)
        >>> _ = exp.run(runner=dummy, progress=False)
        >>> print(exp.report())
        campaign | n             | 1               | chi2_red | pvalue | fit
        ---------+---------------+-----------------+----------+--------+----
        demo     | 0.6667 -> 2/3 | -0.3333 -> -1/3 | 0.00     | 1      | low
        """
        from tinto import estimate

        results = self.fit_fractions(
            metric=metric, basis=basis, x=x, hue=hue, campaign=campaign, **kwargs
        )
        if names is None:
            names = estimate.basis_names(basis)
        return estimate.fractions_table(results, names=names)

    def residuals(
        self,
        metric: str = "steps",
        basis=None,
        normalize: bool = True,
        x: str | None = None,
        hue: str | None = None,
        campaign: str | Iterable[str] | None = None,
    ) -> dict:
        """Per-series fit residuals, for eyeballing noise vs. leftover structure.

        Same ``(x, y, sigma)`` assembly as :meth:`fit`, then returns the
        residuals of the linear `basis` fit (affine by default). The whole
        point is to *look*: under a correct model the normalized residuals
        are iid ``N(0, 1)`` scatter, whereas a **trend, curvature, or a run
        of same-sign points across** `x` betrays a missing term that the
        scalar ``chi2_red`` only hints at.

        Residuals follow the usual convention ``y - model`` (observed
        minus predicted), so a point *above* the fit reads *positive* and
        the plotted curvature matches the data's: a concave law shows a
        concave (∩) residual, not an inverted one.

        Normalized residuals ``(y - model) / sigma`` are the right thing to
        plot: they are dimensionless and weighted by each point's error
        bar, so heteroscedastic ``sigma_i`` do not distort the picture.
        Pass ``normalize=False`` for raw residuals ``y - model`` in the
        metric's own units. Note both the sign convention *and* the
        weighted fit (small-``sigma`` points dominate the line) shape what
        you see — with a strongly ``n``-dependent ``sigma`` the fitted line
        tilts toward the low-``n`` end.

        .. caution::
           Residuals are an *exploratory* tool. Choosing a new basis term
           because you *saw* it in the residuals and then testing that same
           term on the same data is double-dipping: the resulting
           ``pvalue``/``t`` are optimistic. Confirm a term either on fresh
           data or by checking that its coefficient lands on a clean,
           stable fraction — not by the in-sample fit improving (it always
           does).

        Parameters
        ----------
        metric, basis, x, hue, campaign: see :meth:`fit`.
        normalize: :class:`bool`, default=True
            Return ``(y - model) / sigma`` when ``True``, else ``y - model``.

        Returns
        -------
        dict
            ``{label: (xs, resid)}`` with the same labels as :meth:`series`;
            `xs` and `resid` are plain Python lists, sorted by `x`. Plot one
            with ``plt.plot(*exp.residuals()[label], "o")``.

        Examples
        --------
        On exactly affine data the residuals vanish (pure-noise limit):

        >>> import numpy as np
        >>> def dummy(n, n_trials):
        ...     return {"steps": (2.0 * n + 1, 1.0, 0.0),
        ...             "activations": (0.0, 1.0, 0.0)}
        >>> exp = Experiment("demo", iterator=Iterator("n", [1, 2, 3]), n_trials=100)
        >>> _ = exp.run(runner=dummy, progress=False)
        >>> xs, res = exp.residuals()["demo"]
        >>> xs
        [1, 2, 3]
        >>> bool(np.allclose(res, 0.0))
        True
        """
        from tinto import estimate

        out: dict[str, tuple[list, list]] = {}
        for label, xs, ys, sigma in self._xy_sigma(metric, x, hue, campaign):
            r = estimate.fit(xs, ys, sigma, basis)
            resid = r["resid"]
            if not normalize:
                import numpy as np

                resid = resid * np.asarray(sigma, float)
            out[label] = (list(xs), [float(v) for v in resid])
        return out

    def _xy_sigma(self, metric, x, hue, campaign):
        """Yield ``(label, xs, ys, sigma)`` for each selected series.

        Pulls the per-point mean and variance via :meth:`series` and
        derives the standard error of the mean ``sigma_i = sqrt(var_i /
        (n_trials - 1))``. Shared by :meth:`fit` and :meth:`fit_fractions`.
        """
        selected = self._select(campaign)
        for camp_name, camp in selected.items():
            n_trials = camp.get("n_trials")
            if n_trials is None:
                raise ValueError(
                    f"campaign {camp_name!r} has no recorded n_trials; "
                    "cannot derive error bars for a fit."
                )
            means = self.series(metric, "mean", x=x, hue=hue, campaign=camp_name)
            variances = self.series(metric, "var", x=x, hue=hue, campaign=camp_name)
            for label, (xs, ys) in means.items():
                _, vs = variances[label]
                sigma = [(v / (n_trials - 1)) ** 0.5 for v in vs]
                yield label, xs, ys, sigma

    def to_dict(self) -> dict:
        """Return a deep copy of the internal result dict.

        Examples
        --------
        >>> def dummy(n):
        ...     return {"steps": (float(n), 0.0, 0.0), "activations": (0.0, 0.0, 0.0)}
        >>> exp = Experiment("demo", iterator=Iterator("n", [1, 2]))
        >>> d = exp.run(runner=dummy, progress=False).to_dict()
        >>> d["demo"]["data"][(1,)]["steps"]
        (1.0, 0.0, 0.0)
        """
        import copy

        return copy.deepcopy(self.results)

    def to_csv(self, target) -> None:
        """Write flat CSV rows (one per data point) to `target`.

        Columns: ``campaign``, then every axis seen in any campaign
        (empty cells when absent), then ``{metric}_{stat}`` for
        ``steps`` and ``activations`` × ``mean``, ``var``, ``m3``.

        Parameters
        ----------
        target: path-like or text stream
            Opened file path (string / :class:`~pathlib.Path`) or any
            object exposing a ``.write()`` method
            (:class:`io.StringIO`, an opened file, ...).

        Examples
        --------
        >>> import io
        >>> def dummy(n):
        ...     return {"steps": (float(n), 0.0, 0.0), "activations": (float(n+1), 0.0, 0.0)}
        >>> exp = Experiment("demo", iterator=Iterator("n", [1, 2]))
        >>> _ = exp.run(runner=dummy, progress=False)
        >>> buf = io.StringIO()
        >>> exp.to_csv(buf)
        >>> header, *rows = buf.getvalue().splitlines()
        >>> header
        'campaign,n,steps_mean,steps_var,steps_m3,activations_mean,activations_var,activations_m3'
        >>> len(rows)
        2
        >>> rows[0]
        'demo,1,1.0,0.0,0.0,2.0,0.0,0.0'

        Two campaigns sharing the same axis keep a single column for
        it:

        >>> combined = exp + Experiment("bis", iterator=Iterator("n", [3]))
        >>> _ = combined.run(runner=dummy, progress=False)
        >>> buf = io.StringIO()
        >>> combined.to_csv(buf)
        >>> buf.getvalue().splitlines()[0].split(",").count("n")
        1

        A filesystem path is also accepted:

        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as td:
        ...     path = Path(td) / "out.csv"
        ...     exp.to_csv(path)
        ...     "steps_mean" in path.read_text()
        True
        """
        results = self.results

        all_axes: list[str] = []
        for camp in results.values():
            for a in camp["axes"]:
                if a not in all_axes:
                    all_axes.append(a)

        metric_keys = ("steps", "activations")
        stat_names = ("mean", "var", "m3")
        header = (
            ["campaign"]
            + list(all_axes)
            + [f"{m}_{s}" for m in metric_keys for s in stat_names]
        )

        def _write(stream) -> None:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            for camp_name, camp in results.items():
                axes = camp["axes"]
                for key, metrics in camp["data"].items():
                    row: list = [camp_name]
                    for a in all_axes:
                        row.append(key[axes.index(a)] if a in axes else "")
                    for m in metric_keys:
                        for i in range(3):
                            row.append(metrics[m][i])
                    writer.writerow(row)

        if hasattr(target, "write"):
            _write(target)
        else:
            with open(target, "w", newline="") as f:
                _write(f)
