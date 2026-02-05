from typing import Callable

import igraph as ig
import pandas as pd

from tidygraph.activate import ActiveType
from tidygraph.exceptions import TidygraphValueError


def centrality_degree(
    active: ActiveType,
    g: ig.Graph,
    **kwargs: str | bool | Callable[[pd.DataFrame], pd.Series],
) -> float | list[float]:
    """Centrality wrapper for `igraph.degree` and `igraph.strength`.

    References:
        - https://github.com/thomasp85/tidygraph/blob/9a3385fcecc89b6f210c51c5bc9936797b100be4/R/centrality.R#L150
    """
    if active != ActiveType.NODES:
        raise TidygraphValueError("`centrality_degree` can only be applied on Nodes context.")

    valid_keys = set(["weights", "mode", "loops", "normalized"])
    keys = set(kwargs.keys())
    if not keys.issubset(valid_keys):
        diff = keys - valid_keys
        raise TidygraphValueError(f"`centrality_degree` received unexpected keyword arguments: {diff}")

    weights = pd.Series()
    if "weights" in kwargs:
        edges_df = g.get_edge_dataframe()
        weights_func: Callable[[pd.DataFrame], pd.Series] = kwargs.pop("weights")
        weights = weights_func(edges_df)

    if weights.empty:
        return g.degree(**kwargs)
    else:
        return g.strength(weights=weights, **kwargs)
