from typing import Callable

import igraph as ig
import pandas as pd

from tidygraph.activate import ActiveType
from tidygraph.exceptions import TidygraphValueError


def centrality_degree(
    active: ActiveType,
    g: ig.Graph,
    **kwargs: object,
) -> float | list[float]:
    """Centrality wrapper for `igraph.degree` and `igraph.strength` [2].

    The degree centrality measures the fraction of vertices is connected to a vertex [1].

    Args:
        weights (Callable[[pd.DataFrame], pd.Series], Optional): A function that when given the edges \
            DataFrame, returns a pandas series representing custom edge weights.
        mode (str, Optional): `in`, `out`, or `all` representing the type of degree to be returned.
        loops (bool, Optional): Whether to count self-loops.

    Returns:
        float or list of float representing calculated centrality.

    References:
        [1] https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.degree_centrality.html#networkx.algorithms.centrality.degree_centrality
        [2] https://github.com/thomasp85/tidygraph/blob/9a3385fcecc89b6f210c51c5bc9936797b100be4/R/centrality.R#L150
        [3] https://python.igraph.org/en/1.0.0/api/igraph.GraphBase.html#strength
        [4] https://python.igraph.org/en/1.0.0/api/igraph.GraphBase.html#degree
    """
    if active != ActiveType.NODES:
        raise TidygraphValueError("`centrality_degree` can only be applied on Nodes context.")

    valid_keys = set(["weights", "mode", "loops"])
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


def centrality_harmonic(
    active: ActiveType,
    g: ig.Graph,
    **kwargs: object,
) -> list[float]:
    """Computes the harmonic centralities of vertices in a graph.

    The harmonic centrality of a vertex measures how easily other vertices can be reached from it. It is defined as the
    mean inverse distance to all other vertices [1].

    Args:
        vertices (int | list[int], Optional): The vertices for which result should be returned.
        weights (Callable[[pd.DataFrame]. pd.Series], Optional): A function that when given the edges \
            DataFrame, returns a pandas series representing custom edge weights.
        mode (str, Optional): `in`, `out`, or `all`. `in` is the length of incoming paths, `out` is the length \
            of outgoing paths, and `all` means both should be calculated.
        cutoff (float, Optional): When not `None`, only paths less than or equal to this length are considered.
        normalized (bool, Optional): Whether to normalize the result. If True, the result is the mean inverse path \
            length to other vertices. If False, the result is the sum of inverse path lengths to other vertices.

    Returns:
        The calculated harmonic centralities in a list.

    References:
        [1] https://python.igraph.org/en/1.0.0/api/igraph.GraphBase.html#harmonic_centrality
    """
    if active != ActiveType.NODES:
        raise TidygraphValueError("`centrality_harmonic` can only be applied on Nodes context.")

    valid_keys = set(["vertices", "weights", "mode", "cutoff", "normalized"])
    keys = set(kwargs.keys())
    if not keys.issubset(valid_keys):
        diff = keys = valid_keys
        raise TidygraphValueError(f"`centrality_harmonic` received unexpected keyword arguments: {diff}")

    weights = pd.Series()
    if "weights" in kwargs:
        edges_df = g.get_edge_dataframe()
        weights_func: Callable[[pd.DataFrame], pd.Series] = kwargs.pop("weights")
        weights = weights_func(edges_df)

    if weights.empty:
        return g.harmonic_centrality(**kwargs)
    else:
        return g.harmonic_centrality(weights=weights, **kwargs)


def centrality_betweenness(
    active: ActiveType,
    g: ig.Graph,
    **kwargs: object,
) -> list[float]:
    """Computes or estimates the betweenness of vertices in a graph.

    The betweenness measures the extent to which a vertex lies on the shortest paths between
    other vertices in the network [1].

    Args:
        vertices (int | list[int], Optional): The vertices for which betweenness should be returned.
        directed (bool, Optional): Whether to consider directed paths. Defaults to True.
        cutoff (int, Optional): If given, only paths less than or equal to this length are considered, \
            effectively resulting in an estimation of the betweenness for the given vertices. If `None`, the \
            exact betweenness is returned.
        weights (Callable[[pd.DataFrame]. pd.Series], Optional): A function that when given the edges \
            DataFrame, returns a pandas series representing custom edge weights.
        sources (list[int], Optional): The set of source vertices to consider when calculating shortest paths.
        targets (list[int], Optional): The set of target vertices to consider when calculating shortest paths.

    Returns:
        The (possibly cutoff-limited) betweenness of the given vertices in a list.

    References:
        [1] https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html#networkx.algorithms.centrality.betweenness_centrality
        [2] https://python.igraph.org/en/1.0.0/api/igraph.GraphBase.html#betweenness
    """
    if active != ActiveType.NODES:
        raise TidygraphValueError("`centrality_betweenness` can only be applied on Nodes context.")

    valid_keys = set(["vertices", "directed", "cutoff", "weights", "sources", "targets"])
    keys = set(kwargs.keys())
    if not keys.issubset(valid_keys):
        diff = keys = valid_keys
        raise TidygraphValueError(f"`centrality_betweenness` received unexpected keyword arguments: {diff}")

    weights = pd.Series()
    if "weights" in kwargs:
        edges_df = g.get_edge_dataframe()
        weights_func: Callable[[pd.DataFrame], pd.Series] = kwargs.pop("weights")
        weights = weights_func(edges_df)

    if weights.empty:
        return g.betweenness(**kwargs)
    else:
        return g.betweenness(weights=weights, **kwargs)


def centrality_edge_betweenness(
    active: ActiveType,
    g: ig.Graph,
    **kwargs: object,
) -> list[float]:
    """Computes the edge betweenness centralities.

    An edge betweenness is the sum of the fraction of all-pairs shortest paths that pass through the edge [1].
    (i.e. how many shortest path pairs pass through edge `e`?)

    Args:
        directed (bool, Optional): Whether to consider directed paths. Defaults to True.
        cutoff (int, Optional): If given, only paths less than or equal to this length are considered, \
            effectively resulting in an estimation of the betweenness for the given values. If `None`, the \
            exact betweenness is returned.
        weights (Callable[[pd.DataFrame]. pd.Series], Optional): A function that when given the edges \
            DataFrame, returns a pandas series representing custom edge weights.
        sources (list[int], Optional): The set of source vertices to consider when calculating shortest paths.
        targets (list[int], Optional): The set of target vertices to consider when calculating shortest paths.

    Returns:
        A list with the (exact or estimated) edge betweenness of all edges.

    References:
        [1] https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.edge_betweenness_centrality.html
        [2] https://python.igraph.org/en/1.0.0/api/igraph.GraphBase.html#edge-betweenness
    """
    if active != ActiveType.EDGES:
        raise TidygraphValueError("`centrality_edge_betweenness` can only be applied on Edges context.")

    valid_keys = set(["directed", "cutoff", "weights", "sources", "targets"])
    keys = set(kwargs.keys())
    if not keys.issubset(valid_keys):
        diff = keys = valid_keys
        raise TidygraphValueError(f"`centrality_edge_betweenness` received unexpected arguments: {diff}")

    weights = pd.Series()
    if "weights" in kwargs:
        edges_df = g.get_edge_dataframe()
        weights_func: Callable[[pd.DataFrame], pd.Series] = kwargs.pop("weights")
        weights = weights_func(edges_df)

    if weights.empty:
        return g.edge_betweenness(**kwargs)
    else:
        return g.edge_betweenness(weights=weights, **kwargs)


def centrality_closeness(
    active: ActiveType,
    g: ig.Graph,
    **kwargs: object,
) -> list[float]:
    """Computes the closeness centrality for vertices.

    The closeness centrality of a vertex measures how easily other vertices can be reached from it (or the other way).
    It is defined as the number of vertices minus one divided by the sum of the lengths of all shortest paths
    from/to the given vertex [1].

    Args:
        vertices (int | list[int], Optional): The vertices for which betweenness should be returned.
        mode (str, Optional): `in`, `out`, or `all`. `in` is the length of incoming paths, `out` is the length \
            of outgoing paths, and `all` means both should be calculated. Defaults to `all`.
        cutoff (int, Optional): If given, only paths less than or equal to this length are considered, \
            effectively resulting in an estimation of the betweenness for the given vertices. If `None`, the \
            exact betweenness is returned.
        weights (Callable[[pd.DataFrame]. pd.Series], Optional): A function that when given the edges \
            DataFrame, returns a pandas series representing custom edge weights.
        normalized (bool, Optional): Whether to normalize the raw closeness scores by multiplying by the \
            number of vertices minus one. Defaults to True.

    Returns:
        The calculated closeness in a list.

    References:
        [1] https://python.igraph.org/en/1.0.0/api/igraph.GraphBase.html#closeness
    """
    if active != ActiveType.NODES:
        raise TidygraphValueError("`centrality_closeness` can only be applied on Nodes context.")

    valid_keys = set(["vertices", "mode", "cutoff", "weights", "normalized"])
    keys = set(kwargs.keys())
    if not keys.issubset(valid_keys):
        diff = keys = valid_keys
        raise TidygraphValueError(f"`centrality_closeness` received unexpected arguments: {diff}")

    weights = pd.Series()
    if "weights" in kwargs:
        edges_df = g.get_edge_dataframe()
        weights_func: Callable[[pd.DataFrame], pd.Series] = kwargs.pop("weights")
        weights = weights_func(edges_df)

    if weights.empty:
        return g.closeness(**kwargs)
    else:
        return g.closeness(weights=weights, **kwargs)


def centrality_eigenvector(
    active: ActiveType,
    g: ig.Graph,
    **kwargs: object,
):
    """Computes the eigenvector centrality for vertices in a graph.

    It assigns relative scores to all nodes in the network based on the principle that connections from high-scoring
    nodes contribute more to the score of the node in question than equal connections from low-scoring nodes.
    In practice, the centralities are determined by calculating eigenvector corresponding to the largest positive
    eigenvalue of the adjacency matrix. In the undirected case, this function considers the diagonal entries of the
    adjacency matrix to be twice the number of self-loops on the corresponding vertex [1].

    In the directed case, the left eigenvector of the adjacency matrix is calculated. In other words, the centrality
    of a vertex is proportional to the sum of centralities of vertices pointing to it [1].

    Eigenvector centrality is meaningful only for connected graphs. Graphs that are not connected should be decomposed
    into connected components, and the eigenvector centrality calculated for each separately [1].

    Args:
        directed (bool, Optional): Whether to consider directed paths. Defaults to True.
        scale (bool, Optional): Whether to normalize the results wherein the largest value is scaled to 1 (and others \
            relative to that). Defaults to True.
        weights (Callable[[pd.DataFrame]. pd.Series], Optional): A function that when given the edges \
            DataFrame, returns a pandas series representing custom edge weights.
        return_eigenvalue (bool, Optional): Whether to return the largest eigenvalue along with centralities. Defaults \
            to False.
        argpack_options (ARGPACKOptions, Optional): Object used to fine-tune the calculation. If omitted, a default \
            variant is used.

    References:
        [1] https://python.igraph.org/en/1.0.0/api/igraph.GraphBase.html#eigenvector_centrality
    """
    if active != ActiveType.NODES:
        raise TidygraphValueError("`centrality_eigenvector` can only be applied on Nodes context.")

    valid_keys = set(["directed", "scale", "weights", "return_eigenvalue", "argpack_options"])
    keys = set(kwargs.keys())
    if not keys.issubset(valid_keys):
        diff = keys = valid_keys
        raise TidygraphValueError(f"`centrality_eigenvector` received unexpected arguments: {diff}")

    weights = pd.Series()
    if "weights" in kwargs:
        edges_df = g.get_edge_dataframe()
        weights_func: Callable[[pd.DataFrame], pd.Series] = kwargs.pop("weights")
        weights = weights_func(edges_df)

    if weights.empty:
        return g.eigenvector_centrality(**kwargs)
    else:
        return g.eigenvector_centrality(weights=weights, **kwargs)


def centrality_pagerank(
    active: ActiveType,
    g: ig.Graph,
    **kwargs: object,
) -> list[float]:
    """Computes the pagerank centrality.

    PageRank centrality measures the importance of a vertex by assigning scores based on:
     - how many links are pointing to the vertex
     - where the links come from
     - recursive influence

    Args:
        vertices (int | list[int], Optional): The vertices for which betweenness should be returned.
        directed (bool, Optional): Whether to consider directed paths. Defaults to True.
        damping (float, Optional): The damping factor. Damping is the probability of resetting the random walk
            to a uniform distribution in each step. Defaults to `0.85`.
        weights (Callable[[pd.DataFrame]. pd.Series], Optional): A function that when given the edges \
            DataFrame, returns a pandas series representing custom edge weights.
        argpack_options (ARGPACKOptions, Optional): Object used to fine-tune the calculation. If omitted, a default \
            variant is used.
        implementation (str, Optional): `prpack` or `arpack`. Determines which implementation used to solve the \
            PageRank eigenproblem. Defaults to `prpack`.
        personalized (Callable[[pd.DataFrame], pd.Series], Optional): A function that when given the nodes \
            DataFrame, returns the distribution over the vertices to be used when resetting the random walk.

    Returns:
        A list with personalized or non-personalized PageRank values of specified vertices.

    References:
        [1] https://python.igraph.org/en/1.0.0/api/igraph.Graph.html#pagerank
    """
    if active != ActiveType.NODES:
        raise TidygraphValueError("`centrality_pagerank` can only be applied on Nodes context.")

    valid_keys = set(
        [
            "vertices",
            "directed",
            "damping",
            "weights",
            "argpack_options",
            "implementation",
            "personalized",
        ]
    )
    keys = set(kwargs.keys())
    if not keys.issubset(valid_keys):
        diff = keys = valid_keys
        raise TidygraphValueError(f"`centrality_pagerank` received unexpected arguments: {diff}")

    weights = pd.Series()
    if "weights" in kwargs:
        edges_df = g.get_edge_dataframe()
        weights_func: Callable[[pd.DataFrame], pd.Series] = kwargs.pop("weights")
        weights = weights_func(edges_df)

    personalized = pd.Series()
    if "personalized" in kwargs:
        edges_df = g.get_edge_dataframe()
        nodes_df = g.get_vertex_dataframe()
        reset_func: Callable[[pd.DataFrame], pd.Series] = kwargs.pop("personalized")
        personalized = reset_func(nodes_df)

    has_weights, has_personalized = not weights.empty, not personalized.empty

    if has_personalized:
        if has_weights:
            # ! NOTE: `pagerank` only supports python list variant for weight param
            return g.personalized_pagerank(weights=weights.to_list(), reset=personalized, **kwargs)
        return g.personalized_pagerank(reset=personalized, **kwargs)

    if has_weights:
        return g.pagerank(weights=weights.to_list(), **kwargs)

    return g.pagerank(**kwargs)
