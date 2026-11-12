#!/usr/bin/env python

"""Tests for `tinto` package.

These are property-based regression tests for :mod:`tinto.graphs`. They run
under the test suite's default ``NUMBA_DISABLE_JIT=1`` (see ``conftest.py``),
which is enough to catch the bugs they guard against: a corrupted CSR for
graphs with isolated trailing nodes, and an uninitialised loop variable in
:func:`~tinto.graphs.cayley` that crashed on some seeds.
"""

from fractions import Fraction

import numpy as np
import pytest

from tinto import estimate
from tinto.configs import make as make_config
from tinto.experiments import Experiment, Iterator
from tinto.graphs import adja_to_graph, cayley, erdos_renyi, line, ring
from tinto.io import set_seed
from tinto.starters import from_cayley, from_erdos_renyi


def _is_valid_csr(g):
    """``ptr`` starts at 0, is non-decreasing, and ends at ``len(ind)``."""
    ptr = g.ptr
    return ptr[0] == 0 and ptr[-1] == len(g.ind) and bool(np.all(np.diff(ptr) >= 0))


def _is_simple_undirected(adja):
    """Symmetric matrix with a zero diagonal."""
    return np.array_equal(adja, adja.T) and not np.any(np.diag(adja))


def _is_tree(adja):
    """Simple, undirected, exactly ``n - 1`` edges, and connected."""
    n = adja.shape[0]
    if not _is_simple_undirected(adja):
        return False
    if int(adja.sum()) // 2 != n - 1:
        return False
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in range(n):
            if adja[u, v] and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def test_adja_to_graph_trailing_empty_rows():
    """Trailing nodes with no out-edges must keep an empty neighbor list.

    Regression: a previous version only fixed the very last row pointer, so
    the pointers of trailing empty rows stayed at zero and corrupted the CSR.
    """
    adja = np.zeros((4, 4), dtype=np.int64)
    adja[0, 1] = 1
    g = adja_to_graph(adja)
    assert _is_valid_csr(g)
    assert g.neighbors(0).tolist() == [1]
    assert g.neighbors(1).tolist() == []
    assert g.neighbors(3).tolist() == []


def test_adja_to_graph_round_trip():
    """``adja_to_graph`` and ``Graph.to_adja`` are inverse on random graphs."""
    rng = np.random.RandomState(0)
    for _ in range(500):
        n = rng.randint(1, 9)
        adja = rng.rand(n, n) < rng.uniform(0, 0.6)
        np.fill_diagonal(adja, False)
        g = adja_to_graph(adja.astype(np.int64))
        assert _is_valid_csr(g)
        assert np.array_equal(g.to_adja(), adja)


def test_cayley_is_always_a_tree():
    """``cayley`` produces a valid tree for every seed.

    Regression: an uninitialised loop variable made some seeds crash (JIT
    off) or emit self-loops and the wrong edge count (JIT on).
    """
    for n in (2, 3, 6, 12):
        for seed in range(150):
            set_seed(seed)
            g = cayley(n)
            assert _is_valid_csr(g)
            assert _is_tree(g.to_adja()), f"cayley({n}) seed={seed} not a tree"


def test_cayley_single_node():
    """The degenerate single-node tree has no edges."""
    g = cayley(1)
    assert g.n == 1
    assert g.ptr.tolist() == [0, 0]
    assert g.ind.tolist() == []


def test_cayley_covers_all_labeled_trees():
    """Over enough seeds, ``cayley`` hits each of the ``n ** (n - 2)`` trees."""
    n = 4
    seen = set()
    for seed in range(2000):
        set_seed(seed)
        adja = cayley(n).to_adja()
        edges = tuple((i, j) for i in range(n) for j in range(i + 1, n) if adja[i, j])
        seen.add(edges)
    assert len(seen) == n ** (n - 2)


def test_erdos_renyi_is_simple_undirected():
    """``erdos_renyi`` yields a valid simple undirected graph for every seed.

    A low ``p`` makes isolated trailing nodes likely, which is exactly the
    case that broke the CSR construction before.
    """
    for seed in range(200):
        set_seed(seed)
        g = erdos_renyi(8, p=0.15)
        assert _is_valid_csr(g)
        assert _is_simple_undirected(g.to_adja())


def test_erdos_renyi_density_extremes():
    """``p = 0`` gives the empty graph, ``p = 1`` the complete one."""
    set_seed(0)
    assert not erdos_renyi(10, p=0.0).to_adja().any()
    set_seed(0)
    assert erdos_renyi(10, p=1.0).to_adja().sum() == 10 * 9


def test_line_and_ring_structure():
    """``line`` and ``ring`` have the expected neighborhoods."""
    g = line(5)
    assert g.n == 5
    assert g.neighbors(0).tolist() == [1]
    assert g.neighbors(2).tolist() == [1, 3]
    assert g.neighbors(4).tolist() == [3]
    r = ring(5)
    assert r.neighbors(0).tolist() == [1, 4]
    assert r.neighbors(4).tolist() == [0, 3]


def _is_consistent_config(c):
    """Colors stay in ``range(k)`` and the enabled cache matches ``total``."""
    return (
        bool(((c.colors >= 0) & (c.colors < c.k)).all())
        and c.total == int(c.enabled.sum())
        and c.colors.shape[0] == c.graph.n
    )


def test_from_cayley_draws_valid_config_over_tree():
    """``from_cayley`` seeds a consistent config over a genuine Cayley tree."""
    Config = make_config()
    starter = from_cayley(Config)(8)
    for seed in range(50):
        set_seed(seed)
        c = starter.draw()
        assert c.graph.n == 8
        assert _is_tree(c.graph.to_adja())
        assert _is_consistent_config(c)


def test_from_erdos_renyi_draws_valid_config_over_simple_graph():
    """``from_erdos_renyi`` seeds a consistent config over a simple undirected graph."""
    Config = make_config()
    starter = from_erdos_renyi(Config)(8, p=0.4)
    for seed in range(50):
        set_seed(seed)
        c = starter.draw()
        assert c.graph.n == 8
        assert _is_simple_undirected(c.graph.to_adja())
        assert _is_consistent_config(c)


def test_generating_starters_draw_a_fresh_graph_each_call():
    """Successive ``draw`` calls regenerate the random graph rather than reuse it."""
    Config = make_config()
    set_seed(0)
    cayley_adjas = [from_cayley(Config)(12).draw().graph.to_adja() for _ in range(5)]
    er_adjas = [from_erdos_renyi(Config)(12, p=0.5).draw().graph.to_adja() for _ in range(5)]
    # Over five draws on 12 nodes, at least one pair must differ.
    assert any(not np.array_equal(a, cayley_adjas[0]) for a in cayley_adjas[1:])
    assert any(not np.array_equal(a, er_adjas[0]) for a in er_adjas[1:])


def test_generating_starters_uniform_color_regimes():
    """``color=0`` keeps the all-zeros default; a positive color forces a uniform fill."""
    Config = make_config()
    set_seed(0)
    assert from_cayley(Config, color=0)(4).draw().colors.tolist() == [0, 0, 0, 0]
    assert from_cayley(Config, color=1)(4).draw().colors.tolist() == [1, 1, 1, 1]
    assert from_erdos_renyi(Config, color=0)(4, p=0.5).draw().colors.tolist() == [0, 0, 0, 0]
    # p=1.0 is the complete graph, guaranteeing k > 1 so color 1 is valid.
    assert from_erdos_renyi(Config, color=1)(4, p=1.0).draw().colors.tolist() == [1, 1, 1, 1]


def test_generating_starters_are_cached_per_config_and_color():
    """Factories memoize on ``(Config, color)`` and propagate ``uses_ids``."""
    Config = make_config()
    assert from_cayley(Config) is from_cayley(Config, color=None)
    assert from_cayley(Config, color=0) is not from_cayley(Config, color=1)
    assert from_erdos_renyi(Config) is from_erdos_renyi(Config, color=None)
    assert from_erdos_renyi(Config, color=0) is not from_erdos_renyi(Config, color=1)
    # The starter instance mirrors the uses_ids flag of the Config it builds,
    # so that the runner's shuffle resolution behaves as with from_graph.
    assert from_cayley(Config)(8).uses_ids == Config.uses_ids
    assert from_erdos_renyi(Config)(8, p=0.5).uses_ids == Config.uses_ids


# --- estimate: weighted least squares, hidden terms, fractions -------------


def test_wls_and_fit_recover_affine_exactly():
    """On noiseless affine data the coefficients are exact and chi2 is zero."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 3.0 * x - 2.0
    sigma = np.ones_like(x)
    beta, cov, chi2, dof, resid = estimate.wls(x, y, sigma, estimate.affine_basis())
    # Affine basis is [slope, intercept] (dominant term first).
    assert beta == pytest.approx([3.0, -2.0])
    assert chi2 == pytest.approx(0.0, abs=1e-18)
    assert dof == 3
    # fit() agrees and reports a perfect goodness-of-fit.
    r = estimate.fit(x, y, sigma)
    assert r["beta"] == pytest.approx([3.0, -2.0])
    assert r["chi2_red"] == pytest.approx(0.0, abs=1e-18)
    assert r["pvalue"] == pytest.approx(1.0)


def test_residuals_follow_observed_minus_model_convention():
    """Residuals are y - model: a point lifted ABOVE the fit reads positive.

    Guards the sign convention across wls/fit/fit_fractions — a flip would
    silently invert the curvature of every residual plot (a concave law
    would masquerade as convex).
    """
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0
    y[4] += 3.0  # lift the top point clearly above the trend
    sigma = np.ones_like(x)
    _, _, _, _, resid = estimate.wls(x, y, sigma, estimate.affine_basis())
    assert resid[-1] > 0  # observed above predicted -> positive residual
    assert estimate.fit(x, y, sigma)["resid"][-1] > 0
    assert estimate.fit_fractions(x, y, sigma)["resid"][-1] > 0


def test_fit_inverse_power_basis_recovers_k_law():
    """``f(k) = n*(1/k + 1/(3*k**2))`` is linear in ``{1/k, 1/k**2}``."""
    k = np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    n = 50
    y = n * (1.0 / k) + n / (3.0 * k**2)
    r = estimate.fit(k, y, np.ones_like(k), estimate.power_basis([-1, -2]))
    # Coefficients are n and n/3; dividing by n recovers 1 and 1/3.
    assert (r["beta"] / n) == pytest.approx([1.0, 1.0 / 3.0])


def test_simplest_in_tabulated():
    """Stern-Brocot search, including the sign edge cases."""
    assert estimate.simplest_in(0.66, 0.67) == Fraction(2, 3)
    assert estimate.simplest_in(0.67, 0.66) == Fraction(2, 3)  # order-agnostic
    assert estimate.simplest_in(0.2, 0.4) == Fraction(1, 3)
    assert estimate.simplest_in(1.5, 2.5) == Fraction(2)
    assert estimate.simplest_in(-0.1, 0.1) == Fraction(0)  # straddles zero
    assert estimate.simplest_in(-0.67, -0.66) == Fraction(-2, 3)  # negative


def test_recover_fractions_and_guard():
    """recover() picks the simplest compatible fraction and rejects bad fits."""
    assert estimate.recover(0.666694, 0.000116) == Fraction(2, 3)
    assert estimate.recover(0.296269, 0.000077) == Fraction(8, 27)
    # By default chi2_red < 1 does NOT shrink the interval (sampling noise at
    # low dof would otherwise wrongly exclude the true fraction).
    assert estimate.recover(0.6660, 0.0010, 0.25) == Fraction(2, 3)
    # The reference shrink is available on demand.
    assert estimate.recover(0.5, 0.1, chi2_red=0.25, rescale_small=True) == Fraction(1, 2)
    # A badly fitting model is rejected outright.
    with pytest.raises(ValueError, match="hidden structure"):
        estimate.recover(0.5, 0.001, 100.0)


def test_test_extra_term_detects_injected_term():
    """No signal on affine data; strong signal when an extra term is present."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g = lambda v: v * np.log(v)  # noqa: E731

    clean = estimate.test_extra_term(x, 2.0 * x + 1.0, np.ones_like(x), g, "xlogx")
    assert clean["name"] == "xlogx"
    assert abs(clean["t"]) < 1e-6
    assert clean["pvalue"] == pytest.approx(1.0, abs=1e-6)

    spiked = estimate.test_extra_term(x, 2.0 * x + 1.0 + 0.5 * g(x), 0.01 * np.ones_like(x), g)
    assert spiked["c"] == pytest.approx(0.5)
    assert abs(spiked["t"]) > 5.0


def test_fit_fractions_peels_leading_then_subleading():
    """Clean affine law: both coefficients recover, slope (leading) anchored first."""
    x = np.arange(10.0, 60.0)
    y = (2 / 3) * x - 1 / 3
    r = estimate.fit_fractions(x, y, np.full_like(x, 1e-3))
    # Affine basis is [slope, intercept]; the leading slope is anchored first.
    assert r["fractions"] == [Fraction(2, 3), Fraction(-1, 3)]
    assert r["order"] == [0, 1]


def test_fit_fractions_resolution_gate():
    """A busy fraction is anchored only when the precision resolves its denominator."""
    x = np.arange(10.0, 60.0)
    y = np.full_like(x, 8 / 27)
    # Tight error bars resolve the 1/27 grid -> 8/27 is anchored.
    tight = estimate.fit_fractions(x, y, np.full_like(x, 1e-4), basis=[np.ones_like])
    assert tight["fractions"] == [Fraction(8, 27)]
    # Loose error bars cannot resolve it -> left free rather than forced onto a
    # spurious nearby simple fraction.
    loose = estimate.fit_fractions(x, y, np.full_like(x, 0.5), basis=[np.ones_like])
    assert loose["fractions"] == [None]


def test_basis_names_from_builtins_and_fallback():
    """Names come from the basis columns; anonymous lambdas get positional names."""
    assert estimate.basis_names() == ["n", "1"]  # dominant term first
    assert estimate.basis_names(estimate.affine_basis()) == ["n", "1"]
    assert estimate.basis_names(estimate.affine_basis(var="k")) == ["k", "1"]
    assert estimate.basis_names(estimate.power_basis([-1, -2])) == ["n^-1", "n^-2"]
    assert estimate.basis_names([lambda v: v, lambda v: v * v]) == ["c0", "c1"]


def test_health_verdict_from_pvalue():
    """The fit verdict reads straight from the chi-square p-value tails."""
    assert estimate._health(0.5) == "ok"
    assert estimate._health(0.001) == "high"  # chi2 too big: hidden structure
    assert estimate._health(0.999) == "low"  # chi2 too small: inflated sigma
    assert estimate._health(0.02) == "check"  # marginal high
    assert estimate._health(0.98) == "check"  # marginal low
    assert estimate._health(float("nan")) == "?"  # no degrees of freedom


def test_fractions_table_anchored_free_and_html():
    """Anchored coefficients show fractions; free ones show value +/- error."""
    x = np.arange(10.0, 60.0)
    # Perfect affine fit: both coefficients anchored, p ~ 1 -> 'low'.
    anchored = estimate.fit_fractions(x, (2 / 3) * x - 1 / 3, np.full_like(x, 1e-3))
    t1 = estimate.fractions_table({"demo": anchored}, names=["n", "1"])
    assert t1.headers == ["campaign", "n", "1", "chi2_red", "pvalue", "fit"]
    text1 = str(t1)
    assert "-> 2/3" in text1 and "-> -1/3" in text1
    assert text1.strip().endswith("low")  # over-conservative sigma flagged

    # Loose error bars cannot resolve the constant -> shown as value +/- error.
    free = estimate.fit_fractions(
        x, np.full_like(x, 8 / 27), np.full_like(x, 0.5), basis=[np.ones_like]
    )
    t2 = estimate.fractions_table({"demo": free}, names=["1"])
    assert "+/-" in str(t2)
    # HTML upgrades typography: real chi-square, arrow, coloured fit glyph.
    html = t2._repr_html_()
    assert html.startswith("<table>") and "χ²<sub>red</sub>" in html


def test_fractions_table_default_names_are_positional():
    """Without names, columns fall back to c0, c1, ... inferred from the result."""
    x = np.arange(10.0, 60.0)
    r = estimate.fit_fractions(x, (2 / 3) * x - 1 / 3, np.full_like(x, 1e-3))
    table = estimate.fractions_table({"demo": r})
    assert table.headers == ["campaign", "c0", "c1", "chi2_red", "pvalue", "fit"]


def test_experiment_report_tabulates_fractions():
    """Experiment.report builds a named, displayable table from fit_fractions."""

    def dummy(n, n_trials):
        return {"steps": ((2 / 3) * n - 1 / 3, 1.0, 0.0), "activations": (0.0, 1.0, 0.0)}

    exp = Experiment("demo", iterator=Iterator("n", range(10, 60)), n_trials=10**6)
    exp.run(runner=dummy, progress=False)
    table = exp.report()
    # Coefficient columns are named from the (default affine) basis: [n, 1].
    assert table.headers == ["campaign", "n", "1", "chi2_red", "pvalue", "fit"]
    text = str(table)
    assert "demo" in text and "-> 2/3" in text and "-> -1/3" in text
    assert "<table>" in table._repr_html_()


def test_experiment_residuals_and_fit_fractions_expose_resid():
    """Residuals vanish on affine data; fit_fractions now returns them too."""

    def dummy(n, n_trials):
        return {"steps": (2.0 * n + 1.0, 4.0, 0.0), "activations": (0.0, 4.0, 0.0)}

    exp = Experiment("demo", iterator=Iterator("n", [1, 2, 3, 4]), n_trials=101)
    exp.run(runner=dummy, progress=False)
    xs, res = exp.residuals()["demo"]
    assert xs == [1, 2, 3, 4]
    assert res == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-9)
    # Raw residuals exercise the un-normalized branch (also ~0 here).
    _, raw = exp.residuals(normalize=False)["demo"]
    assert raw == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-9)
    # fit_fractions surfaces the normalized residuals of the anchored model.
    r = exp.fit_fractions(metric="steps")["demo"]
    assert len(r["resid"]) == 4


def test_experiment_fit_propagates_n_trials_and_sigma():
    """Experiment.fit derives sigma from var + n_trials and fits per label."""

    def dummy(n, n_trials):
        return {"steps": (2.0 * n + 1.0, 4.0, 0.0), "activations": (0.0, 4.0, 0.0)}

    exp = Experiment("demo", iterator=Iterator("n", [1, 2, 3, 4]), n_trials=101)
    exp.run(runner=dummy, progress=False)
    # n_trials is surfaced in the stored results.
    assert exp.results["demo"]["n_trials"] == 101
    r = exp.fit(metric="steps")
    assert r["demo"]["beta"] == pytest.approx([2.0, 1.0])  # [slope, intercept]
    # sigma_i = sqrt(var / (n_trials - 1)) = sqrt(4 / 100) = 0.2.
    assert r["demo"]["resid"] == pytest.approx(np.zeros(4), abs=1e-9)


def test_experiment_fit_requires_n_trials():
    """A campaign run without n_trials cannot produce error bars."""

    def dummy(n):
        return {"steps": (2.0 * n, 4.0, 0.0), "activations": (0.0, 4.0, 0.0)}

    exp = Experiment("demo", iterator=Iterator("n", [1, 2]))
    exp.run(runner=dummy, progress=False)
    with pytest.raises(ValueError, match="n_trials"):
        exp.fit(metric="steps")


def test_test_extra_term_accepts_an_explicit_basis():
    """Against a constant-only basis, the linear part is the extra term."""
    x = np.arange(1.0, 6.0)
    sigma = 0.01 * np.ones_like(x)
    r = estimate.test_extra_term(x, 3.0 + 2.0 * x, sigma, lambda v: v, basis=[np.ones_like])
    assert r["c"] == pytest.approx(2.0)
    assert abs(r["t"]) > 5.0


def test_fit_fractions_stops_on_an_incomplete_model():
    """A quadratic law fitted on the affine basis: nothing is anchored."""
    x = np.arange(10.0, 60.0)
    r = estimate.fit_fractions(x, x**2 / 7, np.full_like(x, 1e-3))
    assert r["fractions"] == [None, None]
    assert r["order"] == []


def test_fit_fractions_skips_denominators_above_the_cap():
    """A slope of 1/97 is never anchored with max_denominator=64; the intercept is."""
    x = np.arange(10.0, 60.0)
    y = x / 97 + 1 / 2
    r = estimate.fit_fractions(x, y, np.full_like(x, 1e-9), max_denominator=64)
    assert r["fractions"] == [None, Fraction(1, 2)]


def test_experiment_report_accepts_names():
    def dummy(n, n_trials):
        steps = ((2 / 3) * n - 1 / 3, 1.0, 0.0)
        return {"steps": steps, "activations": (0.0, 1.0, 0.0)}

    exp = Experiment("demo", iterator=Iterator("n", range(10, 60)), n_trials=10**6)
    exp.run(runner=dummy, progress=False)
    table = exp.report(names=["slope", "intercept"])
    assert table.headers[1:3] == ["slope", "intercept"]
