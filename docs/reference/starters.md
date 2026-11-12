# Starters

```{eval-rst}
.. automodule:: tinto.starters
   :members:
   :undoc-members:
   :private-members:
   :show-inheritance:
```

```{note}
The `FromGraphStarter`, `FromCayleyStarter` and `FromErdosRenyiStarter`
jitclasses are built dynamically by {func}`tinto.starters.from_graph`,
{func}`tinto.starters.from_cayley` and
{func}`tinto.starters.from_erdos_renyi` respectively, and cannot be
reached by `automodule`. Their constructors and `draw()` methods are
described in the docstrings of the factory functions above.
```
