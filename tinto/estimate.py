"""Parameter estimation on Monte-Carlo sweeps: laws, hidden terms, fractions.

Given results ``y_i = f(x_i)`` produced by :func:`tinto.runners.mc` over a
swept parameter (graph size ``n``, palette size ``k``, scheduler probability
``p``, ...), this module answers two questions:

1. *Does* ``f`` *follow a given law, or does it hide an extra term?* —
   :func:`wls` fits an arbitrary **linear basis** by weighted least squares
   (affine ``a*x + b`` is just the default basis), and :func:`test_extra_term`
   runs a likelihood-ratio test against the same model augmented by a suspect
   term ``c*g(x)`` (typically ``g(x) = x*log x``, or ``x**-3`` for a hidden
   ``1/k**3``).
2. *If the law holds, what are the exact coefficients?* —
   :func:`simplest_in` returns the simplest fraction inside an interval
   (Stern-Brocot), and :func:`recover` turns a fitted coefficient and its
   error bar into the simplest compatible fraction.

Everything hinges on **one input**: a correct per-point error bar ``sigma_i``.
For a mean estimated over ``n_trials`` replicas this is the standard error of
the mean, ``sigma_i = sqrt(var_i / n_trials)`` — see
:meth:`tinto.experiments.Experiment.fit`, which assembles it from the
``var`` already returned by :func:`tinto.runners.mc`. The fit is run with
``absolute_sigma=True`` semantics: the ``sigma_i`` are trusted as given and
the covariance is **not** rescaled, so the goodness-of-fit (``chi2``,
``pvalue``) stays meaningful.

The basis is deliberately generic. ``f(n)`` affine laws use the default
:func:`affine_basis`; an inverse-power law such as
``f(k) = n*(1/k + 1/(3*k**2))`` uses :func:`power_basis` with exponents
``[-1, -2]``. The fraction machinery then applies to each coefficient
unchanged.

Reading the outputs (consolidated from the reference recipe):

- ``chi2_red`` ~ 1 (within ``sqrt(2/dof)``) when the model is right and the
  ``sigma_i`` are correct; ``>> 1`` means a hidden term or under-estimated
  ``sigma_i``; ``<< 1`` means over-estimated ``sigma_i``.
- ``pvalue`` of :func:`fit` is the global goodness-of-fit; tiny means
  something is hiding.
- ``t = c/sc`` of :func:`test_extra_term` is the signal-to-noise of the
  suspect term: ``|t| <~ 2`` no evidence, ``|t| >~ 3`` the term is real.
"""

import math
from fractions import Fraction

import numpy as np
from scipy import stats


def affine_basis(var="n"):
    """Return the affine basis ``[x, 1]`` — dominant term first.

    The linear term comes first (the leading behaviour of the laws this
    module hunts), the constant second, so the fitted coefficients read
    ``[slope, intercept]``. Each column carries a ``__name__`` (`var` and
    ``"1"``) so that :func:`basis_names` — and therefore
    :meth:`tinto.experiments.Experiment.report` — labels the coefficients
    automatically.

    Parameters
    ----------
    var: :class:`str`, default='n'
        Symbol used to name the linear column. ``"n"`` matches the usual
        graph-size sweep.

    Examples
    --------
    >>> import numpy as np
    >>> b = affine_basis()
    >>> len(b)
    2
    >>> [f.__name__ for f in b]
    ['n', '1']
    >>> x = np.array([0.0, 1.0, 2.0])
    >>> b[0](x).tolist()
    [0.0, 1.0, 2.0]
    >>> b[1](x).tolist()
    [1.0, 1.0, 1.0]
    >>> [f.__name__ for f in affine_basis(var="k")]
    ['k', '1']
    """
    x = lambda v: v  # noqa: E731
    x.__name__ = var
    one = lambda v: np.ones_like(v)  # noqa: E731
    one.__name__ = "1"
    return [x, one]


def power_basis(exponents, var="n"):
    """Return a basis of pure powers ``v ** e``, one column per exponent.

    Useful for inverse-power laws: ``power_basis([-1, -2])`` spans
    ``{1/n, 1/n**2}``, the natural basis for ``f(k) = a/k + b/k**2``.

    Each column carries a ``__name__`` of the form ``"<var>^<e>"`` (read
    by :func:`basis_names`) so the fitted coefficients label themselves.

    Parameters
    ----------
    exponents: iterable of numbers
        Real exponents; one basis column ``v -> v ** e`` is produced per
        entry, in order.
    var: :class:`str`, default='n'
        Symbol used to name the columns.

    Examples
    --------
    >>> import numpy as np
    >>> b = power_basis([-1, -2])
    >>> [f.__name__ for f in b]
    ['n^-1', 'n^-2']
    >>> [f.__name__ for f in power_basis([-1, -2], var="k")]
    ['k^-1', 'k^-2']
    >>> x = np.array([1.0, 2.0, 4.0])
    >>> b[0](x).tolist()
    [1.0, 0.5, 0.25]
    >>> b[1](x).tolist()
    [1.0, 0.25, 0.0625]
    """
    cols = []
    for e in exponents:
        f = lambda v, e=float(e): np.asarray(v, float) ** e  # noqa: E731
        f.__name__ = f"{var}^{float(e):g}"
        cols.append(f)
    return cols


def basis_names(basis=None, var="n"):
    """Best-effort coefficient names for a linear `basis`.

    Reads the ``__name__`` attached to each basis column (the built-in
    :func:`affine_basis` and :func:`power_basis` set meaningful ones).
    Anonymous columns — a bare ``lambda`` whose ``__name__`` is
    ``"<lambda>"``, or anything without one — fall back to a positional
    ``"c<i>"``.

    Parameters
    ----------
    basis: list of callables, optional
        The basis to name. ``None`` is treated as the affine default and
        returns ``[var, "1"]`` (dominant term first, matching
        :func:`affine_basis`).
    var: :class:`str`, default='n'
        Variable symbol used for the affine default's linear term.

    Returns
    -------
    list of :class:`str`
        One name per basis column, aligned with the fitted coefficients.

    Examples
    --------
    >>> basis_names()
    ['n', '1']
    >>> basis_names(power_basis([-1, -2]))
    ['n^-1', 'n^-2']
    >>> basis_names(affine_basis(var="k"))
    ['k', '1']

    A custom basis of anonymous lambdas gets positional names:

    >>> basis_names([lambda v: v, lambda v: v * v])
    ['c0', 'c1']
    """
    if basis is None:
        return [var, "1"]
    names = []
    for i, f in enumerate(basis):
        nm = getattr(f, "__name__", None)
        names.append(nm if nm and nm != "<lambda>" else f"c{i}")
    return names


def wls(x, y, sigma, basis):
    """Weighted least squares of ``y`` on a linear `basis`.

    Minimizes ``sum(((X @ beta - y) / sigma) ** 2)`` where ``X`` is the
    design matrix whose columns are ``f(x)`` for each ``f`` in `basis`.
    The covariance is ``inv(A.T @ A)`` with ``A = X / sigma`` — i.e.
    ``absolute_sigma=True`` semantics: the `sigma` are trusted and the
    covariance is **not** rescaled by the reduced chi-square.

    Parameters
    ----------
    x, y, sigma: array_like
        Abscissae, measured values, and per-point error bars (same
        length). ``sigma`` must be strictly positive.
    basis: list of callables
        Each ``f(x) -> column`` builds one column of the design matrix.

    Returns
    -------
    beta: :class:`numpy.ndarray`
        Fitted coefficients, one per basis column.
    cov: :class:`numpy.ndarray`
        Covariance matrix of `beta`.
    chi2: :class:`float`
        Weighted sum of squared residuals.
    dof: :class:`int`
        Degrees of freedom, ``len(x) - len(basis)``.
    resid: :class:`numpy.ndarray`
        Normalized residuals in the usual convention ``(y - X @ beta) /
        sigma`` (observed minus predicted): a point *above* the fit gets a
        *positive* residual. The sign is irrelevant to `chi2` (squared)
        but matters when the residuals are inspected or plotted.

    Examples
    --------
    On exactly affine data the slope and intercept are recovered and the
    residual chi-square is zero:

    >>> import numpy as np
    >>> x = np.array([0.0, 1.0, 2.0, 3.0])
    >>> y = 2 * x + 1
    >>> sigma = np.ones_like(x)
    >>> beta, cov, chi2, dof, resid = wls(x, y, sigma, [np.ones_like, lambda v: v])
    >>> np.round(beta, 6).tolist()
    [1.0, 2.0]
    >>> bool(chi2 < 1e-20)
    True
    >>> dof
    2
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    X = np.column_stack([f(x) for f in basis])
    A = X / sigma[:, None]
    b = y / sigma
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    resid = (y - X @ beta) / sigma
    chi2 = float(resid @ resid)
    dof = len(x) - X.shape[1]
    return beta, cov, chi2, dof, resid


def fit(x, y, sigma, basis=None):
    """Fit ``y`` on a linear `basis` and report goodness-of-fit.

    Thin wrapper around :func:`wls` that defaults to the affine basis and
    packages the per-coefficient error bars and the chi-square statistics
    into a single dict.

    Parameters
    ----------
    x, y, sigma: array_like
        See :func:`wls`.
    basis: list of callables, optional
        Defaults to :func:`affine_basis` (``a*x + b``).

    Returns
    -------
    :class:`dict`
        ``beta`` (coefficients), ``cov`` (covariance), ``sigma_beta``
        (``sqrt(diag(cov))``), ``chi2``, ``dof``, ``chi2_red``
        (``chi2 / dof``), ``pvalue`` (``stats.chi2.sf(chi2, dof)``), and
        ``resid`` (normalized residuals).

    Examples
    --------
    Affine fit (the default basis), exact data:

    >>> import numpy as np
    >>> x = np.array([0.0, 1.0, 2.0, 3.0])
    >>> y = 2 * x + 1
    >>> r = fit(x, y, np.ones_like(x))
    >>> np.round(r["beta"], 6).tolist()
    [2.0, 1.0]
    >>> r["dof"]
    2
    >>> bool(r["chi2_red"] < 1e-20)
    True
    >>> round(r["pvalue"], 6)
    1.0

    Inverse-power fit ``f(k) = n*(1/k + 1/(3*k**2))`` with ``n = 10``;
    dividing the coefficients by ``n`` recovers ``1`` and ``1/3``:

    >>> k = np.array([3.0, 4.0, 5.0, 6.0])
    >>> n = 10
    >>> y = n * (1 / k) + n / (3 * k**2)
    >>> r = fit(k, y, np.ones_like(k), power_basis([-1, -2]))
    >>> [round(float(b) / n, 6) for b in r["beta"]]
    [1.0, 0.333333]
    """
    if basis is None:
        basis = affine_basis()
    beta, cov, chi2, dof, resid = wls(x, y, sigma, basis)
    return dict(
        beta=beta,
        cov=cov,
        sigma_beta=np.sqrt(np.diag(cov)),
        chi2=chi2,
        dof=dof,
        chi2_red=chi2 / dof if dof else float("inf"),
        pvalue=float(stats.chi2.sf(chi2, dof)) if dof else float("nan"),
        resid=resid,
    )


def test_extra_term(x, y, sigma, g, name="g", basis=None):
    """Likelihood-ratio test for a suspect extra term ``c*g(x)``.

    Fits the `basis` model (default affine) and the same model augmented
    by one column ``g(x)``, then compares them. Under the null hypothesis
    ``c = 0`` the drop in chi-square ``dchi2 = chi2_0 - chi2_1`` follows a
    chi-square law with one degree of freedom (``dchi2 ~= t**2``).

    Parameters
    ----------
    x, y, sigma: array_like
        See :func:`wls`.
    g: callable
        The suspect term ``g(x) -> column``. E.g. ``lambda v: v*np.log(v)``
        for a hidden ``x*log x``, or ``lambda v: v**-3`` for a ``1/k**3``.
    name: :class:`str`, default='g'
        Label echoed back in the result, for readability.
    basis: list of callables, optional
        The null model basis. Defaults to :func:`affine_basis`.

    Returns
    -------
    :class:`dict`
        ``name``, ``c`` (fitted coefficient of `g`), ``sc`` (its error
        bar), ``t`` (``c / sc``, the signal-to-noise), ``dchi2``, and
        ``pvalue`` (``stats.chi2.sf(dchi2, 1)``).

    Examples
    --------
    On exactly affine data the suspect ``x*log x`` term is not detected
    (``t`` ~ 0):

    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y = 2 * x + 1
    >>> sigma = np.ones_like(x)
    >>> res = test_extra_term(x, y, sigma, lambda v: v * np.log(v), "xlogx")
    >>> res["name"]
    'xlogx'
    >>> bool(abs(res["t"]) < 1e-6)
    True

    Inject a real term and it is recovered:

    >>> y2 = 2 * x + 1 + 0.5 * (x * np.log(x))
    >>> res2 = test_extra_term(x, y2, sigma, lambda v: v * np.log(v))
    >>> round(res2["c"], 6)
    0.5
    """
    if basis is None:
        basis = affine_basis()
    _, _, chi2_0, _, _ = wls(x, y, sigma, basis)
    beta, cov, chi2_1, _, _ = wls(x, y, sigma, [*basis, g])
    c = float(beta[-1])
    sc = math.sqrt(cov[-1, -1])
    dchi2 = chi2_0 - chi2_1
    return dict(
        name=name,
        c=c,
        sc=sc,
        t=c / sc,
        dchi2=dchi2,
        pvalue=float(stats.chi2.sf(dchi2, 1)),
    )


def simplest_in(lo, hi):
    """Return the simplest fraction inside ``[lo, hi]`` (Stern-Brocot).

    "Simplest" = smallest denominator, then smallest numerator. Handles
    the sign cases that a naive continued-fraction search gets wrong: an
    interval straddling zero returns ``0``, a fully negative interval is
    handled by symmetry.

    Parameters
    ----------
    lo, hi: :class:`float`
        Interval bounds, in any order.

    Returns
    -------
    :class:`fractions.Fraction`

    Examples
    --------
    >>> simplest_in(0.66, 0.67)
    Fraction(2, 3)
    >>> simplest_in(0.2, 0.4)
    Fraction(1, 3)
    >>> simplest_in(1.5, 2.5)
    Fraction(2, 1)

    Zero is the simplest fraction of all, so any interval containing it
    returns it; negative intervals work by symmetry:

    >>> simplest_in(-0.1, 0.1)
    Fraction(0, 1)
    >>> simplest_in(-0.67, -0.66)
    Fraction(-2, 3)
    """
    if lo > hi:
        lo, hi = hi, lo
    if lo <= 0 <= hi:
        return Fraction(0)
    if hi < 0:
        return -simplest_in(-hi, -lo)
    if math.ceil(lo) <= math.floor(hi):
        return Fraction(math.ceil(lo))
    n = math.floor(lo)
    return n + 1 / simplest_in(1 / (hi - n), 1 / (lo - n))


def recover(value, sigma_abs, chi2_red=1.0, k=2, max_chi2_red=5.0, rescale_small=False):
    """Simplest fraction compatible with a fitted coefficient.

    Returns the simplest fraction inside ``value +/- k*sigma_abs``. The
    ``chi2_red`` of the parent fit acts as a guard: a model that fits
    badly (``chi2_red`` above `max_chi2_red`) is rejected outright, since
    a hidden term makes fraction recovery meaningless.

    The reference recipe additionally *shrinks* the interval by
    ``sqrt(chi2_red)`` when ``chi2_red < 1`` (the symptom of
    over-estimated ``sigma_i``). That is **off by default** here: with few
    sweep points ``chi2_red`` fluctuates widely (its spread is
    ``sqrt(2/dof)``), so a value below 1 is usually sampling noise, not
    inflated error bars — and shrinking then wrongly excludes the true
    simple fraction. Enable `rescale_small` only when you *know* the
    ``sigma_i`` are inflated (e.g. a forgotten ``/n_trials``); the right
    fix is to correct them at the source.

    Parameters
    ----------
    value: :class:`float`
        Fitted coefficient.
    sigma_abs: :class:`float`
        Its (absolute) error bar, e.g. ``sigma_beta[i]`` from :func:`fit`.
    chi2_red: :class:`float`, default=1.0
        Reduced chi-square of the parent fit, used by the `max_chi2_red`
        guard and (if `rescale_small`) the shrink factor.
    k: :class:`float`, default=2
        Half-width of the confidence interval, in error bars.
    max_chi2_red: :class:`float`, default=5.0
        Above this, the model is deemed wrong and the call raises.
    rescale_small: :class:`bool`, default=False
        When ``True`` and ``chi2_red < 1``, shrink the interval by
        ``sqrt(chi2_red)``.

    Returns
    -------
    :class:`fractions.Fraction`

    Raises
    ------
    ValueError
        If ``chi2_red`` exceeds `max_chi2_red`.

    Examples
    --------
    >>> recover(0.6667, 0.001)
    Fraction(2, 3)
    >>> recover(0.296269, 0.000077)
    Fraction(8, 27)

    A badly fitting model is rejected rather than mined for a fraction:

    >>> recover(0.5, 0.001, 100.0)
    Traceback (most recent call last):
        ...
    ValueError: chi2_red exceeds max_chi2_red: hidden structure, model rejected

    Opting into the shrink reproduces the reference behaviour:

    >>> recover(0.5, 0.1, chi2_red=0.25, rescale_small=True)
    Fraction(1, 2)
    """
    if chi2_red > max_chi2_red:
        raise ValueError(
            "chi2_red exceeds max_chi2_red: hidden structure, model rejected"
        )
    scale = math.sqrt(chi2_red) if (rescale_small and chi2_red < 1) else 1.0
    return simplest_in(value - k * sigma_abs * scale, value + k * sigma_abs * scale)


def fit_fractions(
    x, y, sigma, basis=None, k=2, max_chi2_red=5.0, max_denominator=64, resolution=1.0
):
    """Recover exact fractions coefficient by coefficient (anchor & refit).

    A single joint fit gives unbiased coefficients but leaves the
    sub-leading ones noisy and correlated with the leading ones, so
    :func:`recover` often collapses a busy fraction (say ``-7/27``) onto a
    simpler neighbour (``-1/4``). This routine peels the coefficients off
    one at a time instead:

    1. **jointly** fit every still-free coefficient (never one in
       isolation — fitting a term alone biases it by the terms you drop),
    2. **anchor** the one most confidently a simple fraction (smallest
       denominator, then smallest relative error), fixing it to that
       exact value,
    3. **refit** the remaining free coefficients jointly — the anchored
       term's contribution is subtracted from ``y`` and its correlated
       degree of freedom is gone, so the others tighten honestly,
    4. repeat until no free coefficient lands cleanly on a fraction.

    A coefficient is anchored only if its precision actually **resolves**
    its denominator: fractions with denominator at most ``q`` are spaced
    by about ``1/q**2``, so the confidence interval must be narrower than
    that grid, ``k*sigma*q**2 < resolution``. This is what tells a
    well-determined busy fraction (a slope of ``8/27``, tiny relative
    error) apart from an under-determined one of the *same* denominator (a
    sub-leading ``-7/27`` whose error bar spans several competing
    fractions): the former is anchored, the latter is left free. The check
    is deliberately conservative — anchoring a *wrong* fraction would bias
    every later refit — so when the data cannot resolve a coefficient it
    stays ``None`` and its float estimate is returned instead.

    Parameters
    ----------
    x, y, sigma: array_like
        See :func:`wls`.
    basis: list of callables, optional
        Defaults to :func:`affine_basis`.
    k: :class:`float`, default=2
        Confidence half-width in error bars, forwarded to :func:`recover`.
    max_chi2_red: :class:`float`, default=5.0
        If a refit's reduced chi-square exceeds this, peeling stops (the
        model is judged incomplete on the remaining terms).
    max_denominator: :class:`int`, default=64
        A candidate fraction with a larger denominator is never anchored.
    resolution: :class:`float`, default=1.0
        Resolution budget: a denominator-``q`` fraction is anchored only
        if ``k*sigma*q**2 < resolution``. Smaller is stricter.

    Returns
    -------
    :class:`dict`
        ``fractions`` (list aligned with `basis`: a
        :class:`~fractions.Fraction` for anchored coefficients, ``None``
        for those left free), ``beta`` (full coefficient vector, anchored
        entries as exact floats), ``sigma_beta`` (``0`` for anchored,
        fitted error for free), ``order`` (basis indices in the order they
        were anchored), the final-model ``chi2``, ``dof``, ``chi2_red``,
        ``pvalue``, and ``resid`` (normalized residuals ``(y - model) /
        sigma`` of the final anchored model, observed minus predicted —
        inspect them, e.g. via
        :meth:`tinto.experiments.Experiment.residuals`, to tell noise from
        leftover structure).

    Examples
    --------
    On a clean affine law both coefficients are recovered exactly:

    >>> import numpy as np
    >>> x = np.arange(10.0, 60.0)
    >>> y = (2 / 3) * x - 1 / 3
    >>> r = fit_fractions(x, y, np.full_like(x, 1e-3))
    >>> r["fractions"]
    [Fraction(2, 3), Fraction(-1, 3)]
    """
    if basis is None:
        basis = affine_basis()
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    n = len(basis)
    cols = [np.asarray(f(x), float) for f in basis]

    anchored: dict = {}
    order: list = []
    while len(anchored) < n:
        free = [i for i in range(n) if i not in anchored]
        y_adj = y - sum((float(anchored[j]) * cols[j] for j in anchored), np.zeros_like(y))
        r = fit(x, y_adj, sigma, [basis[i] for i in free])
        if r["chi2_red"] > max_chi2_red:
            break
        best = None
        for pos, i in enumerate(free):
            val, sig = float(r["beta"][pos]), float(r["sigma_beta"][pos])
            frac = recover(val, sig, k=k, max_chi2_red=float("inf"))
            q = frac.denominator
            if q > max_denominator:
                continue
            # Resolution test: the interval must be narrower than the ~1/q**2
            # spacing of the denominator-<=q grid, else the fraction is not
            # actually pinned down (several competitors fit) and anchoring it
            # would bias every later refit.
            if k * sig * q * q >= resolution:
                continue
            # Among resolved candidates, anchor the most relatively-precise
            # (i.e. leading) one first; the denominator breaks ties.
            rank = (abs(sig / val) if val else float("inf"), q)
            if best is None or rank < best[0]:
                best = (rank, i, frac)
        if best is None:
            break
        anchored[best[1]] = best[2]
        order.append(best[1])

    beta = np.zeros(n)
    sigma_beta = np.zeros(n)
    for i, frac in anchored.items():
        beta[i] = float(frac)
    free = [i for i in range(n) if i not in anchored]
    if free:
        y_adj = y - sum((float(anchored[j]) * cols[j] for j in anchored), np.zeros_like(y))
        rf = fit(x, y_adj, sigma, [basis[i] for i in free])
        for pos, i in enumerate(free):
            beta[i] = float(rf["beta"][pos])
            sigma_beta[i] = float(rf["sigma_beta"][pos])

    model = sum((beta[i] * cols[i] for i in range(n)), np.zeros_like(y))
    resid = (y - model) / sigma
    chi2 = float(resid @ resid)
    dof = len(x) - len(free)
    return dict(
        fractions=[anchored.get(i) for i in range(n)],
        beta=beta,
        sigma_beta=sigma_beta,
        order=order,
        chi2=chi2,
        dof=dof,
        chi2_red=chi2 / dof if dof else float("inf"),
        pvalue=float(stats.chi2.sf(chi2, dof)) if dof else float("nan"),
        resid=resid,
    )


def _health(pvalue):
    """One-word verdict on a fit, read straight from its chi-square p-value.

    ``pvalue = stats.chi2.sf(chi2, dof)`` is the upper-tail probability:
    under a correct model with correct ``sigma_i`` it is uniform on
    ``[0, 1]``, so reading the tails is dof-free (no normal approximation,
    valid even at small dof):

    - ``"high"`` (``pvalue < 0.01``): ``chi2`` far too large — a hidden
      term or under-estimated ``sigma_i``. Recovered fractions are *not*
      trustworthy.
    - ``"low"`` (``pvalue > 0.99``): ``chi2`` far too small —
      over-estimated ``sigma_i`` (e.g. a forgotten ``/n_trials``).
    - ``"check"`` (``pvalue < 0.05`` or ``> 0.95``): marginal either way,
      worth a second look (typically a weak sub-leading term).
    - ``"ok"`` otherwise.
    - ``"?"`` when `pvalue` is not finite (e.g. ``dof <= 0``).

    Examples
    --------
    >>> [_health(p) for p in (0.5, 0.001, 0.999, 0.02, 0.98)]
    ['ok', 'high', 'low', 'check', 'check']
    >>> _health(float("nan"))
    '?'
    """
    if not math.isfinite(pvalue):
        return "?"
    if pvalue < 0.01:
        return "high"
    if pvalue > 0.99:
        return "low"
    if pvalue < 0.05 or pvalue > 0.95:
        return "check"
    return "ok"


class _FractionTable:
    """A table of recovered fractions with environment-aware rendering.

    Holds ``headers`` and string ``rows``; renders as an aligned ASCII
    grid via :meth:`__str__`/:meth:`__repr__` (terminals, REPL, captured
    test output) and as an HTML ``<table>`` via :meth:`_repr_html_`,
    which Jupyter picks up automatically — same ``.auto`` spirit as
    :mod:`tqdm`, but driven by the consumer rather than by sniffing the
    environment. Column widths are computed from the content, so labels,
    wide fractions, and negative numbers all stay aligned.

    The ASCII view stays plain (terminal-safe on every code page); the
    HTML view, unconstrained by the console encoding, upgrades to proper
    typography — a real ``χ²``, ``→``/``±`` instead of ``->``/``+/-``, and
    a coloured glyph per ``fit`` verdict.
    """

    #: ``fit`` verdict -> (glyph, CSS colour) for the HTML rendering.
    _FIT_GLYPH = {
        "ok": ("✓", "#2e7d32"),
        "check": ("⚠", "#f9a825"),
        "high": ("✗", "#c62828"),
        "low": ("↓", "#1565c0"),
        "?": ("?", "#757575"),
    }

    def __init__(self, headers, rows):
        self.headers = [str(h) for h in headers]
        self.rows = [[str(c) for c in row] for row in rows]

    def __str__(self):
        grid = [self.headers, *self.rows]
        widths = [max(len(row[i]) for row in grid) for i in range(len(self.headers))]
        fmt = lambda row: " | ".join(f"{c:<{w}}" for c, w in zip(row, widths))  # noqa: E731
        sep = "-+-".join("-" * w for w in widths)
        return "\n".join([fmt(self.headers), sep, *(fmt(r) for r in self.rows)])

    __repr__ = __str__

    @staticmethod
    def _esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _header_html(self, h):
        h = self._esc(h).replace("chi2_red", "χ²<sub>red</sub>").replace("pvalue", "p")
        return f"<th>{h}</th>"

    def _cell_html(self, cell, is_fit):
        if is_fit and cell in self._FIT_GLYPH:
            glyph, color = self._FIT_GLYPH[cell]
            return f'<td style="color:{color}">{glyph} {self._esc(cell)}</td>'
        # esc first, then prettify the now-escaped ASCII operators.
        pretty = self._esc(cell).replace(" -&gt; ", " → ").replace(" +/- ", " ± ")
        return f"<td>{pretty}</td>"

    def _repr_html_(self):
        last = len(self.headers) - 1
        th = "".join(self._header_html(h) for h in self.headers)
        body = "".join(
            "<tr>"
            + "".join(self._cell_html(c, i == last) for i, c in enumerate(row))
            + "</tr>"
            for row in self.rows
        )
        return f"<table>\n<thead><tr>{th}</tr></thead>\n<tbody>{body}</tbody>\n</table>"


def fractions_table(results, names=None):
    """Render a ``{label: fit_fractions result}`` mapping as a table.

    Pure formatter for the output of :func:`fit_fractions` (one entry per
    series, as produced by :meth:`tinto.experiments.Experiment.fit_fractions`).
    Knows nothing about :class:`~tinto.experiments.Experiment`: it consumes
    only the result dicts, so it stays trivially testable and reusable.

    One row per label; one column per coefficient, named by `names`. An
    **anchored** coefficient shows ``value -> fraction``; a **free** one
    shows ``value +/- error``. Two trailing columns report ``chi2_red``
    and ``pvalue``, and a final ``fit`` column carries the
    :func:`_health` verdict (``ok`` / ``check`` / ``high`` / ``low``) —
    the whole point being that a fraction recovered under a ``high`` fit
    is not to be trusted, and a ``check`` deserves a second look.

    Parameters
    ----------
    results: mapping
        ``{label: result}`` where each `result` is a dict as returned by
        :func:`fit_fractions` (keys ``fractions``, ``beta``,
        ``sigma_beta``, ``chi2_red``, ``pvalue``, ``dof``).
    names: list of :class:`str`, optional
        Coefficient names, aligned with the basis. Defaults to positional
        ``c0, c1, ...`` inferred from the first result's length. Use
        :func:`basis_names` to derive them from a basis.

    Returns
    -------
    :class:`_FractionTable`
        Printable (aligned ASCII) and notebook-displayable (HTML) table.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(10.0, 60.0)
    >>> y = (2 / 3) * x - 1 / 3
    >>> results = {"clean": fit_fractions(x, y, np.full_like(x, 1e-3))}
    >>> table = fractions_table(results, names=basis_names())
    >>> table.headers
    ['campaign', 'n', '1', 'chi2_red', 'pvalue', 'fit']
    >>> print(table)
    campaign | n             | 1               | chi2_red | pvalue | fit
    ---------+---------------+-----------------+----------+--------+----
    clean    | 0.6667 -> 2/3 | -0.3333 -> -1/3 | 0.00     | 1      | low

    The HTML rendering Jupyter would show upgrades the typography (a real
    ``χ²``, ``→``, and a coloured ``fit`` glyph):

    >>> html = table._repr_html_()
    >>> "<table>" in html and "χ²<sub>red</sub>" in html and "→" in html
    True
    """
    results = dict(results)
    if names is None:
        first = next(iter(results.values()), None)
        n = len(first["fractions"]) if first else 0
        names = [f"c{i}" for i in range(n)]
    headers = ["campaign", *names, "chi2_red", "pvalue", "fit"]
    rows = []
    for label, r in results.items():
        cells = [str(label)]
        for frac, beta, sig in zip(r["fractions"], r["beta"], r["sigma_beta"]):
            if frac is None:
                cells.append(f"{float(beta):.4f} +/- {float(sig):.4f}")
            else:
                cells.append(f"{float(frac):.4f} -> {frac}")
        cells.append(f"{r['chi2_red']:.2f}")
        cells.append(f"{r['pvalue']:.3g}")
        cells.append(_health(r["pvalue"]))
        rows.append(cells)
    return _FractionTable(headers, rows)
