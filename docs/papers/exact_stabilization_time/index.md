# Exact stabilization time of self-stabilizing coloring

*The Exact Stabilization Time of Self-Stabilizing Coloring: Mean, Variance, and Distribution*, by Élie de Panafieu, Fabien Mathieu, Maria Potop-Butucaru and Sébastien Tixeuil. Research report, 2026.

% To add with the HAL deposit: the link to the PDF of the report.

- {doc}`Research report <report>`, web edition: the full text, appendices included, with links between statements, equations and references.
- Software: the report cites [tinto 0.1.0](https://github.com/balouf/tinto/tree/v0.1.0), archived by Software Heritage as [`swh:1:rev:79d72a3026ff486616ef0ba83b70e192de339cee`](https://archive.softwareheritage.org/swh:1:rev:79d72a3026ff486616ef0ba83b70e192de339cee;origin=https://github.com/balouf/tinto;visit=swh:1:snp:b56b180e61fa4780760e32a04e9dd7c4fac3fec1).
- {doc}`Companion notebook <companion>`: every numerical check of the report (its Appendix G), with the current version of tinto.

## Abstract

The worst-case stabilization time of a self-stabilizing algorithm usually says little about its typical behavior beyond the order of magnitude.
This paper revisits the classical self-stabilizing $(\Delta+1)$-coloring algorithm with vertices activated at random, one at a time (the random central daemon).
Considering the number of color changes as a random variable, we study its mean, variance and, on several families of graphs, its exact distribution.
The analysis rests on a reduction: a vertex changes color, at most once, exactly when it is not a local maximum of a uniformly random labeling of the vertices, maxima being taken among the neighbors that share its initial color (the monochrome initial configuration is the slowest case).
What can be predicted depends on what is known of the graph. The degree distribution (the view at distance one) gives the mean. The variance needs the view at distance two, in particular the triangles, and the full law needs the whole graph.
We give closed-form results, from monochrome and from uniformly random initial configurations, on paths, cycles, random trees, Erdős-Rényi graphs and bipartite random graphs.
From a monochrome start, the mean stays within a factor two of the worst case, whereas the variance ranges from zero (disjoint cliques) to order $n^2$ (the star on $n$ vertices).

:::{toctree}
:hidden:

Research report <report>
Companion notebook <companion>
:::
