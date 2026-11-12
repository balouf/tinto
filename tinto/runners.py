"""Drivers that iterate the coloring dynamics until stabilization.

:func:`simple` runs one trajectory of the scheduler/solver pair on a given
:class:`~tinto.configs.Config` until no node remains enabled, returning the
number of productive steps. :func:`mc` runs many such trajectories in
parallel via :func:`numba.prange` and aggregates the first three raw moments
of the step count.

Both entry points dispatch to a specialized :func:`numba.njit` function
compiled on demand for a given shuffle strategy. The shuffle decision is
taken at call time by OR-ing the ``uses_ids`` flag of the
:class:`~tinto.configs.Config` (inherited from its guard) with that of the
solver (inherited from its colorer). When ``uses_ids`` is already set on
the config — meaning ``Config.__init__`` has already drawn a fresh
permutation — the runner skips the extra shuffle to avoid a redundant
allocation.
"""

from functools import lru_cache

from numba import njit, prange


@lru_cache(maxsize=None)
def _make_simple(shuffle):
    @njit(fastmath=True)
    def simple(config, scheduler, solver):
        if shuffle:
            config.shuffle_ids()
        steps = 0
        activations = 0
        while config.total > 0:
            decisions = scheduler.draw(config)
            solver(config, decisions)
            if len(decisions) > 0:
                steps += 1
                activations += len(decisions)
        return steps, activations

    return simple


def simple(config, scheduler, solver):
    """Run the dynamics until :attr:`config.total` reaches zero.

    At each iteration, the `scheduler` is asked for a batch of decisions
    (nodes to recolor), the `solver` applies one color update per
    decision, and :meth:`~tinto.configs.Config.check` is implicitly
    refreshed through the solver's side effects. A step is counted only
    when the scheduler returned a non-empty batch ; idle draws are not
    counted but do not stop the loop.

    Before the loop starts, :meth:`~tinto.configs.Config.shuffle_ids` is
    called iff the solver's colorer consumes ids and the config's guard
    did not already populate them at construction time. The decision is
    taken once at call time from the ``uses_ids`` flags exposed on the
    config class and on the solver.

    Parameters
    ----------
    config: :class:`~tinto.configs.Config`
        Mutable coloring configuration. On return, ``config.total == 0``.
    scheduler: jitclass instance
        Must expose ``draw(config) -> ndarray of int64`` returning the
        indices of nodes to recolor at the current step.
    solver: callable
        Signature ``solver(config, decisions) -> None``. Typically a
        function produced by :func:`tinto.solvers.make`.

    Returns
    -------
    steps: :class:`int`
        Number of productive steps taken to reach a stable coloring.
    activations: :class:`int`
        Total number of recolorings performed by the solver across all
        steps.

    Examples
    --------
    >>> from tinto.io import set_seed
    >>> from tinto.graphs import line
    >>> from tinto.configs import make as make_config
    >>> from tinto.schedulers import shuffled, bernoulli, independent
    >>> from tinto.solvers import make as make_solver
    >>> set_seed(0)
    >>> Config = make_config()
    >>> c = Config(line(5))
    >>> c.total
    5
    >>> sched = shuffled()(c)
    >>> solver = make_solver()
    >>> int(simple(c, sched, solver)[0]) > 0
    True
    >>> c.total
    0
    >>> c = Config(line(10))
    >>> sched = bernoulli()(c)
    >>> simple(c, sched, solver)
    (5, 9)
    >>> c.color_uniform()
    >>> sched = independent()(c)
    >>> simple(c, sched, solver)
    (4, 6)
    >>> c.color_uniform()
    >>> sched = bernoulli(p=.95)(c)
    >>> simple(c, sched, solver)
    (8, 49)
    >>> c.color_uniform()
    >>> sched = bernoulli(p=.05)(c)
    >>> simple(c, sched, solver)
    (6, 6)

    When the solver's colorer consumes ids but the config's guard does
    not, :func:`simple` calls :meth:`~tinto.configs.Config.shuffle_ids`
    before the loop so that the colorer sees a valid permutation. We
    illustrate this by tagging a throwaway colorer with
    ``uses_ids=True``:

    >>> from numba import njit
    >>> from tinto.solvers import colorer_lowest
    >>> @njit
    ... def id_aware_colorer(config, i):
    ...     _ = config.ids[i]
    ...     return colorer_lowest(config, i)
    >>> id_aware_colorer.uses_ids = True
    >>> id_solver = make_solver(id_aware_colorer)
    >>> c = Config(line(6))
    >>> c.ids.shape  # empty: default guard has uses_ids=False
    (0,)
    >>> sched = shuffled()(c)
    >>> steps, _ = simple(c, sched, id_solver)
    >>> int(steps) > 0
    True
    >>> c.ids.shape  # now populated by the runner's pre-run shuffle
    (6,)
    """
    shuffle = _resolve_shuffle(config, solver)
    return _make_simple(shuffle)(config, scheduler, solver)


@lru_cache(maxsize=None)
def _make_mc(shuffle):
    _simple = _make_simple(shuffle)

    @njit(parallel=True)
    def mc(starter, scheduler, solver, n_trials=1000):
        s1, a1 = 0.0, 0.0
        s2, a2 = 0.0, 0.0
        s3, a3 = 0.0, 0.0
        for i in prange(n_trials):
            config = starter.draw()
            steps, activations = _simple(config, scheduler(config), solver)
            s1 += steps
            s2 += steps * steps
            s3 += steps * steps * steps
            a1 += activations
            a2 += activations * activations
            a3 += activations * activations * activations
        mean_s = s1 / n_trials
        var_s = s2 / n_trials - mean_s**2
        m3_s = s3 / n_trials - 3 * mean_s * (s2 / n_trials) + 2 * mean_s**3
        mean_a = a1 / n_trials
        var_a = a2 / n_trials - mean_a**2
        m3_a = a3 / n_trials - 3 * mean_a * (a2 / n_trials) + 2 * mean_a**3
        return {"steps": (mean_s, var_s, m3_s), "activations": (mean_a, var_a, m3_a)}

    return mc


def mc(starter, scheduler, solver, n_trials=1000):
    """Monte-Carlo sample the step count of :func:`simple` and return its first three moments.

    For each of `n_trials` replicas, a fresh :class:`~tinto.configs.Config`
    is drawn from `starter`, a fresh scheduler instance is built from
    `scheduler(config)`, and :func:`simple` runs it to completion. The
    step counts are aggregated into mean, variance and third central
    moment.

    The same shuffle decision as :func:`simple` is taken once at call
    time from the ``uses_ids`` flags of the starter and the solver.

    Parameters
    ----------
    starter: jitclass instance
        Must expose ``draw() -> Config`` returning a freshly initialized
        :class:`~tinto.configs.Config`. Typically an instance of one of
        the starter classes produced by :mod:`tinto.starters`.
    scheduler: callable
        Factory ``scheduler(config) -> jitclass instance`` as produced
        by e.g. :func:`tinto.schedulers.shuffled`. Called once per
        replica on the freshly drawn config.
    solver: callable
        Same contract as in :func:`simple`.
    n_trials: :class:`int`, default=1000
        Number of Monte-Carlo replicas.

    Returns
    -------
    :class:`dict`
        Maps ``"steps"`` and ``"activations"`` to a tuple
        ``(mean, var, m3)`` where ``mean`` is the empirical mean,
        ``var`` is the empirical variance ``E[X^2] - E[X]^2``, and
        ``m3`` is the third central moment ``E[(X - E[X])^3]``.

    Examples
    --------
    >>> from tinto.io import set_seed
    >>> from tinto.graphs import line
    >>> from tinto.configs import make as make_config
    >>> from tinto.starters import from_graph
    >>> from tinto.schedulers import shuffled
    >>> from tinto.solvers import make as make_solver
    >>> set_seed(0)
    >>> Config = make_config()
    >>> starter = from_graph(Config)(line(5))
    >>> res = mc(starter, shuffled(), make_solver(), n_trials=100)
    >>> mean, var, m3 = res["steps"]
    >>> float(mean) > 0
    True
    >>> isinstance(float(var), float) and isinstance(float(m3), float)
    True
    """
    shuffle = _resolve_shuffle(starter, solver)
    return _make_mc(shuffle)(starter, scheduler, solver, n_trials)


def _resolve_shuffle(config_or_starter, solver):
    """OR the ``uses_ids`` flags and subtract the already-done shuffle.

    A ``Config`` whose guard consumes ids has already drawn a fresh
    permutation inside ``__init__`` (see :func:`tinto.configs.make`) and
    does not need a second shuffle. Likewise a ``Starter`` whose
    underlying Config does the same. The runner therefore needs to
    shuffle only when the solver's colorer consumes ids *and* the config
    side does not.
    """
    config_uses_ids = getattr(config_or_starter, "uses_ids", True)
    solver_uses_ids = getattr(solver, "uses_ids", False)
    return solver_uses_ids and not config_uses_ids
