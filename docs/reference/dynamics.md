# Dynamics

A run of the dynamics combines a graph, a coloring configuration whose guard says which nodes are enabled, a scheduler that picks the nodes to recolor, and a solver that recolors them. A starter draws the initial configuration and a runner iterates until no node is enabled.

- {doc}`graphs`: compiled graphs, the path and the cycle, random generators;
- {doc}`guards`: the enabling rules;
- {doc}`configs`: the coloring configuration and its set of enabled nodes;
- {doc}`starters`: initial configurations, on a fixed or a random graph;
- {doc}`schedulers`: which enabled nodes move at each step;
- {doc}`solvers`: how a node picks its new color;
- {doc}`runners`: one trajectory, or Monte Carlo over many.

:::{toctree}
:hidden:

graphs
guards
configs
starters
schedulers
solvers
runners
:::
