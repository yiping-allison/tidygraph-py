import igraph as ig
import pytest

from tidygraph import Tidygraph


@pytest.mark.parametrize(
    "nodes,edges,directed,expected",
    [
        pytest.param(
            ["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("b", "d")], False, "unrooted tree", id="unrooted tree"
        ),
        pytest.param(["a", "b"], [("a", "b")], True, "rooted tree", id="rooted tree"),
        pytest.param(
            ["a", "b", "c", "d"], [("a", "b"), ("c", "d")], False, "unrooted forest with 2 trees", id="unrooted forest"
        ),
        pytest.param(
            ["a", "b", "c", "d", "e"], [("a", "b"), ("c", "d")], True, "rooted forest with 3 trees", id="rooted forest"
        ),
        pytest.param(
            ["a", "b", "c", "d"],
            [("a", "b"), ("b", "c"), ("c", "a")],
            False,
            "undirected simple graph with 2 component(s)",
            id="undirected simple graph",
        ),
        pytest.param(
            ["a", "b", "c", "d"],
            [("c", "a"), ("a", "b"), ("b", "c"), ("c", "a")],
            False,
            "undirected multigraph with 2 component(s)",
            id="undirected multigraph",
        ),
        pytest.param(
            ["a", "b", "c", "d"],
            [("a", "b"), ("b", "c"), ("c", "b"), ("c", "d")],
            False,
            "bipartite multigraph with 1 component(s)",
            id="bipartite multigraph",
        ),
        pytest.param(
            ["a", "b", "c", "d"],
            [("a", "b"), ("a", "d"), ("c", "b"), ("c", "d")],
            False,
            "bipartite simple graph with 1 component(s)",
            id="bipartite simple graph",
        ),
        pytest.param(
            ["a", "b", "c", "d"],
            [("a", "b"), ("a", "c"), ("c", "d"), ("b", "d")],
            True,
            "directed acyclic simple graph with 4 component(s)",
            id="directed acyclic simple graph",
        ),
        pytest.param(
            ["a", "b", "c", "d"],
            [("a", "b"), ("b", "c"), ("c", "a")],
            True,
            "directed simple graph with 2 component(s)",
            id="directed simple graph",
        ),
        pytest.param(
            ["a", "b", "c", "d"],
            [("a", "b"), ("b", "c"), ("c", "a"), ("a", "b")],
            True,
            "directed multigraph with 2 component(s)",
            id="directed multigraph",
        ),
    ],
)
def test_describe(nodes, edges, directed, expected):
    g = ig.Graph(directed=directed)
    g.add_vertices(nodes)
    g.add_edges(edges)

    tidygraph = Tidygraph(g)
    description = tidygraph.describe()
    assert description == expected
