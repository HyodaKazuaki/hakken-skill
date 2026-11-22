"""hakkenlib - Hakken-style knowledge prediction toolkit (Python standard library only).

Implements the workflow of Besold et al., "Hakken: Predicting future discoveries to
fill the gaps in today's knowledge" (arXiv:2609.04494) without model training on GPUs:

* graph branch  -> temporal network features + per-run logistic regression
                   (fitted in seconds on the graph's own history) or a fixed heuristic
* language branch -> performed by the calling agent (judge queue / record / fuse)
* explanations  -> PHELInE-style sufficiency / necessity over evidence paths
* selection     -> novelty, top-k per entity, recency, path length
"""

__version__ = "2.0.0"
WORKSPACE_SCHEMA = 1
