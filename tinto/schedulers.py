"""Schedulers deciding which nodes get a chance to be recolored at each step.

A *scheduler* is a :mod:`numba` jitclass exposing
``draw(config) -> ndarray of int64`` that returns the indices of nodes to
recolor at the current step. Returning an empty array is allowed and does
not count as a productive step in :func:`tinto.runners.simple`.

Four schedulers are shipped:

- :class:`Shuffled`: walks a random permutation of the node set and returns
  the current node iff it is enabled.
- :class:`Ordered`: same, but walks the node set in natural order.
- :class:`Bernoulli`: each enabled node is independently selected with
  probability `p`.
- :class:`Independent`: sequential greedy variant of :class:`Bernoulli`
  that, visiting the nodes in random order, refuses to add a node whose
  neighbor has already been accepted, so that the returned batch is an
  independent set.

The lowercase factory functions :func:`shuffled`, :func:`ordered`,
:func:`bernoulli` and :func:`independent` return a compiled
``factory(config) -> scheduler`` callable, which is the form expected by
:func:`tinto.runners.mc`.
"""

from functools import lru_cache

from numba.experimental import jitclass
from numba import int64, float64, bool, njit
import numpy as np


@jitclass
class Shuffled:
    """Round-robin scheduler over a random permutation of the nodes.

    At each call, the cursor :attr:`i` advances one position in
    :attr:`perm`; the pointed node is returned iff it is currently
    enabled.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes in the graph.

    Attributes
    ----------
    n: :class:`int64`
        See the corresponding parameter.
    i: :class:`int64`
        Current position in :attr:`perm`, initialized to ``n - 1`` so
        that the first :meth:`draw` call reads ``perm[0]``.
    perm: :class:`~numpy.ndarray` of :class:`int64`, shape (n,)
        Random permutation of ``range(n)`` fixed at construction time.
    empty: :class:`~numpy.ndarray` of :class:`int64`, shape (0,)
        Cached empty return value, reused to avoid per-call allocation.
    one: :class:`~numpy.ndarray` of :class:`int64`, shape (1,)
        Cached single-element return buffer, overwritten in place.

    Examples
    --------
    >>> from tinto.io import set_seed
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> set_seed(0)
    >>> Config = make()
    >>> c = Config(line(4))
    >>> s = Shuffled(c.graph.n)

    On an all-enabled configuration, draw returns exactly one node:

    >>> out = s.draw(c)
    >>> len(out) == 1 and bool(c.enabled[out[0]])
    True

    Once a proper coloring disables every node, every draw is empty:

    >>> c.colors[:] = [0, 1, 0, 1]
    >>> c.check_all()
    >>> all(len(s.draw(c)) == 0 for _ in range(10))
    True
    """

    n: int64  # type: ignore
    i: int64  # type: ignore
    perm: int64[:]  # type: ignore
    empty: int64[:]  # type: ignore
    one: int64[:]  # type: ignore

    def __init__(self, n):
        self.n = n
        self.perm = np.random.permutation(self.n)
        self.i = self.n - 1
        self.empty = np.zeros(0, dtype=np.int64)
        self.one = np.zeros(1, dtype=np.int64)

    def draw(self, config):
        """Advance the cursor and return the pointed node iff enabled.

        Parameters
        ----------
        config: :class:`~tinto.configs.Config`
            Current coloring configuration, read-only.

        Returns
        -------
        :class:`~numpy.ndarray` of :class:`int64`
            Either ``[perm[i]]`` if that node is enabled, or an empty
            array otherwise.
        """
        self.i = (self.i + 1) % self.n
        target = self.perm[self.i]
        if config.enabled[target]:
            self.one[0] = target
            return self.one
        return self.empty


@lru_cache(maxsize=None)
def shuffled():
    """Build a :class:`Shuffled` factory suitable for :func:`tinto.runners.mc`.

    The returned callable takes a :class:`~tinto.configs.Config` and
    returns a fresh :class:`Shuffled` instance sized for its graph.

    Returns
    -------
    callable
        A :func:`numba.njit`-compiled ``factory(config) -> Shuffled``.

    Notes
    -----
    Memoized: repeated calls return the same compiled factory.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> Config = make()
    >>> c = Config(line(4))
    >>> factory = shuffled()
    >>> s = factory(c)
    >>> s.n
    4
    >>> shuffled() is shuffled()
    True
    """

    def _shuffled(config):
        return Shuffled(config.graph.n)

    return njit(_shuffled, fastmath=True)


@jitclass
class Ordered:
    """Round-robin scheduler over the nodes in natural order.

    Identical contract as :class:`Shuffled` but visits the nodes in
    ``0, 1, ..., n-1`` order instead of a random permutation.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes in the graph.

    Attributes
    ----------
    n: :class:`int64`
        See the corresponding parameter.
    i: :class:`int64`
        Current cursor, initialized to ``n - 1`` so that the first
        :meth:`draw` call reads node ``0``.
    empty: :class:`~numpy.ndarray` of :class:`int64`, shape (0,)
        Cached empty return value.
    one: :class:`~numpy.ndarray` of :class:`int64`, shape (1,)
        Cached single-element return buffer.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> Config = make()
    >>> c = Config(line(4))
    >>> o = Ordered(c.graph.n)
    >>> o.draw(c).tolist()
    [0]
    >>> o.draw(c).tolist()
    [1]

    Disabling every node makes the next draw empty:

    >>> c.colors[:] = [0, 1, 0, 1]
    >>> c.check_all()
    >>> o.draw(c).tolist()
    []
    """

    n: int64  # type: ignore
    i: int64  # type: ignore
    empty: int64[:]  # type: ignore
    one: int64[:]  # type: ignore

    def __init__(self, n):
        self.n = n
        self.i = self.n - 1
        self.empty = np.zeros(0, dtype=np.int64)
        self.one = np.zeros(1, dtype=np.int64)

    def draw(self, config):
        """Advance the cursor and return the pointed node iff enabled.

        Parameters
        ----------
        config: :class:`~tinto.configs.Config`
            Current coloring configuration, read-only.

        Returns
        -------
        :class:`~numpy.ndarray` of :class:`int64`
            Either ``[i]`` if node `i` is enabled, or an empty array
            otherwise.
        """
        self.i = (self.i + 1) % self.n
        if config.enabled[self.i]:
            self.one[0] = self.i
            return self.one
        return self.empty


@lru_cache(maxsize=None)
def ordered():
    """Build an :class:`Ordered` factory suitable for :func:`tinto.runners.mc`.

    Returns
    -------
    callable
        A :func:`numba.njit`-compiled ``factory(config) -> Ordered``.

    Notes
    -----
    Memoized: repeated calls return the same compiled factory.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> Config = make()
    >>> c = Config(line(4))
    >>> factory = ordered()
    >>> o = factory(c)
    >>> o.n
    4
    >>> ordered() is ordered()
    True
    """

    def _ordered(config):
        return Ordered(config.graph.n)

    return njit(_ordered, fastmath=True)


@jitclass
class Bernoulli:
    """Each enabled node is independently drawn with probability `p`.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes in the graph.
    p: :class:`float`
        Independent selection probability, in ``[0, 1]``.

    Attributes
    ----------
    n: :class:`int64`
        See the corresponding parameter.
    p: :class:`float64`
        See the corresponding parameter.
    choices: :class:`~numpy.ndarray` of :class:`int64`, shape (n,)
        Cached ``arange(n)`` used to slice out the drawn indices.
    decisions: :class:`~numpy.ndarray` of :class:`bool`, shape (n,)
        Scratch mask refilled at each :meth:`draw` call. Reusing this
        buffer saves one allocation per call.

    Examples
    --------
    With ``p = 0.0``, no node is ever selected:

    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> Config = make()
    >>> c = Config(line(4))
    >>> Bernoulli(c.graph.n, 0.0).draw(c).tolist()
    []

    With ``p = 1.0``, every enabled node is selected:

    >>> Bernoulli(c.graph.n, 1.0).draw(c).tolist()
    [0, 1, 2, 3]
    """

    n: int64  # type: ignore
    p: float64  # type: ignore
    choices: int64[:]  # type: ignore
    decisions: bool[:]  # type: ignore

    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.choices = np.arange(n)
        self.decisions = np.zeros(n, dtype=np.bool)

    def draw(self, config):
        """Return a random subset of the enabled nodes.

        Parameters
        ----------
        config: :class:`~tinto.configs.Config`
            Current coloring configuration, read-only.

        Returns
        -------
        :class:`~numpy.ndarray` of :class:`int64`
            Indices of nodes drawn with probability `p` among those
            currently enabled.
        """
        self.decisions.fill(False)
        for i in range(self.n):
            if config.enabled[i] and np.random.rand() < self.p:
                self.decisions[i] = True
        return self.choices[self.decisions]


@lru_cache(maxsize=None)
def bernoulli(p=1 / 3):
    """Build a :class:`Bernoulli` factory suitable for :func:`tinto.runners.mc`.

    Parameters
    ----------
    p: :class:`float`, default=1/3
        Independent selection probability passed to
        :class:`Bernoulli`.

    Returns
    -------
    callable
        A :func:`numba.njit`-compiled ``factory(config) -> Bernoulli``.

    Notes
    -----
    Memoized on `p`: two calls with the same float value hit the
    cache.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> Config = make()
    >>> c = Config(line(4))
    >>> b = bernoulli(p=0.25)(c)
    >>> b.p
    0.25
    >>> bernoulli(0.25) is bernoulli(0.25)
    True
    >>> bernoulli(0.25) is bernoulli(0.5)
    False
    """

    def _bernoulli(config):
        return Bernoulli(config.graph.n, p)

    return njit(_bernoulli, fastmath=True)


@jitclass
class Independent:
    """Sequential greedy scheduler yielding an independent set of enabled nodes.

    The nodes are visited in a fresh random order at every
    :meth:`draw` call; at each visit, the current node goes through a
    Bernoulli trial (probability `p`) and is accepted only if none of
    its neighbors has already been accepted during this call. The
    returned batch is therefore an independent set of the graph, which
    lets the solver recolor every chosen node in parallel without
    conflict.

    Compared to a batch Bernoulli draw followed by a conflict filter,
    this sequential scheme accepts more nodes (it never rejects a pair
    of adjacent draws outright) at the cost of an extra permutation
    per call.

    Parameters
    ----------
    n: :class:`int`
        Number of nodes in the graph.
    p: :class:`float`
        Per-visit Bernoulli probability, in ``[0, 1]``.

    Attributes
    ----------
    n: :class:`int64`
        See the corresponding parameter.
    p: :class:`float64`
        See the corresponding parameter.
    choices: :class:`~numpy.ndarray` of :class:`int64`, shape (n,)
        Cached ``arange(n)`` used to slice out the accepted indices.
    decisions: :class:`~numpy.ndarray` of :class:`bool`, shape (n,)
        Scratch mask flipped to ``True`` for accepted nodes. Zeroed at
        the start of each :meth:`draw` call so reusing the buffer saves
        one allocation per call.

    Examples
    --------
    With ``p = 1.0`` on a 5-node line, every enabled node is considered
    and the sequential filter keeps exactly the even-indexed nodes once
    the random order is fixed by the seed:

    >>> from tinto.io import set_seed
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> Config = make()
    >>> set_seed(0)
    >>> c = Config(line(5))
    >>> Independent(c.graph.n, 1.0).draw(c).tolist()
    [0, 2, 4]

    With ``p = 0.0`` no node passes the Bernoulli trial:

    >>> Independent(c.graph.n, 0.0).draw(c).tolist()
    []

    Disabling odd-indexed nodes on the same line makes the three even
    nodes mutually non-adjacent, so all of them survive regardless of
    the perm order:

    >>> c.enabled[1::2] = False
    >>> Independent(c.graph.n, 1.0).draw(c).tolist()
    [0, 2, 4]
    """

    n: int64  # type: ignore
    p: float64  # type: ignore
    choices: int64[:]  # type: ignore
    decisions: bool[:]  # type: ignore

    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.choices = np.arange(n)
        self.decisions = np.zeros(n, dtype=np.bool)

    def draw(self, config):
        """Return an independent-set batch of enabled nodes.

        Parameters
        ----------
        config: :class:`~tinto.configs.Config`
            Current coloring configuration, read-only.

        Returns
        -------
        :class:`~numpy.ndarray` of :class:`int64`
            Indices of the nodes accepted during the sequential pass,
            sorted in ascending order.
        """
        self.decisions.fill(False)
        perm = np.random.permutation(self.n)
        for i in perm:
            if config.enabled[i] and np.random.rand() < self.p:
                for j in config.graph.neighbors(i):
                    if self.decisions[j]:
                        break
                else:
                    self.decisions[i] = True
        return self.choices[self.decisions]


@lru_cache(maxsize=None)
def independent(p=1 / 3):
    """Build an :class:`Independent` factory suitable for :func:`tinto.runners.mc`.

    Parameters
    ----------
    p: :class:`float`, default=1/3
        Independent selection probability passed to
        :class:`Independent`.

    Returns
    -------
    callable
        A :func:`numba.njit`-compiled ``factory(config) -> Independent``.

    Notes
    -----
    Memoized on `p`: two calls with the same float value hit the
    cache.

    Examples
    --------
    >>> from tinto.graphs import line
    >>> from tinto.configs import make
    >>> Config = make()
    >>> c = Config(line(4))
    >>> ind = independent(p=0.5)(c)
    >>> ind.p
    0.5
    >>> independent(0.5) is independent(0.5)
    True
    """

    def _independent(config):
        return Independent(config.graph.n, p)

    return njit(_independent, fastmath=True)
