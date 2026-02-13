from typing import Any

import igraph as ig
import pytest

from tidygraph import Tidygraph


@pytest.fixture(scope="function", params=["directed", "undirected"])
def graph(request) -> ig.Graph:
    kind = request.param
    n = 4
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ]
    max = 26  # assume we have max of 26 nodes
    start = ord("a")
    g = ig.Graph(
        n=n,
        edges=edges,
        directed=(kind == "directed"),
        vertex_attrs={"name": [chr(start + (char % max)) for char in range(n)]},
    )

    return g


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

    tidygraph = Tidygraph(graph=g)
    description = tidygraph.describe()
    assert description == expected


@pytest.mark.parametrize(
    "method,args",
    [
        pytest.param("degree", {"mode": "out"}),
        pytest.param("components", {"mode": "weak"}),
        pytest.param("layout", {"layout": "star"}),
        pytest.param("plot", {"backend": "cairo"}),
    ],
)
def test_proxy_to_igraph_succeeds(graph: ig.Graph, method: str, args: dict[str, Any]):
    tg = Tidygraph(graph=graph)
    func = getattr(tg, method)
    _ = func(**args)
