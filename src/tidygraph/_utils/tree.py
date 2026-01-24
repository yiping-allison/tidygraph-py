import igraph as ig

from tidygraph.exceptions import TidygraphValueError


def is_tree(g: ig.Graph) -> bool:
    """Returns True if the graph `g` is a tree.

    A tree is a connected graph with no undirected cycles.

    For directed graphs, `g` is a tree if the underlying graph is a tree. The
    underlying graph is obtained by treating each directed edge as a single undirected edge in a multigraph.

    Args:
        g: The igraph graph to test

    Returns:
        True if the graph is a tree, False otherwise.

    Raises:
        TidygraphValueError: If the graph has no nodes.

    References:
        - https://networkx.org/documentation/stable/_modules/networkx/algorithms/tree/recognition.html
    """
    if not g:
        raise TidygraphValueError("graph `g` has no nodes")

    is_connected = g.is_connected()
    if g.is_directed():
        is_connected = g.is_connected(mode="weak")

    return g.vcount() - 1 == g.ecount() and is_connected


def is_forest(g: ig.Graph) -> bool:
    """Returns True if the graph `g` is a forest.

    A forest is a graph with no undirected cycles.

    For directed graphs, `g` is a forest if the underlying graph is a forest. The
    underlying graph is obtained by treating each directed edge as a single undirected edge in a multigraph.

    Args:
        g: The igraph graph to test

    Returns:
        True if the graph is a forest, False otherwise.

    Raises:
        TidygraphValueError: If the graph has no nodes.

    References:
        - https://networkx.org/documentation/stable/_modules/networkx/algorithms/tree/recognition.html
    """
    if not g:
        raise TidygraphValueError("graph `g` has no nodes")

    components = (g.subgraph(c) for c in g.connected_components())
    if g.is_directed():
        components = (g.subgraph(c) for c in g.connected_components(mode="weak"))

    components = list(components)

    return all([c.vcount() - 1 == c.ecount() for c in components])
