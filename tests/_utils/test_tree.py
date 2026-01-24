import igraph as ig
import pytest

from tidygraph._utils import tree
from tidygraph.exceptions import TidygraphValueError


@pytest.fixture(scope="function")
def empty_graph() -> ig.Graph:
    g = ig.Graph()
    return g


def test_invalid_graph_raises(empty_graph: ig.Graph):
    funcs = [tree.is_tree, tree.is_forest]
    for func in funcs:
        with pytest.raises(TidygraphValueError):
            _ = func(empty_graph)


@pytest.mark.parametrize(
    "nodes,edges,directed,expected",
    [
        pytest.param(
            ["a", "b", "c"],
            [("a", "b"), ("a", "c")],
            False,
            True,
            id="undirected tree",
        ),
        pytest.param(
            ["a", "b", "c"],
            [("a", "b"), ("a", "c")],
            True,
            True,
            id="directed tree",
        ),
        pytest.param(
            ["a", "b", "c"],
            [("a", "b"), ("b", "c"), ("c", "a")],
            False,
            False,
            id="not a tree",
        ),
    ],
)
def test_is_tree(nodes, edges, directed, expected):
    g = ig.Graph(directed=directed)
    g.add_vertices(nodes)
    g.add_edges(edges)

    result = tree.is_tree(g)
    assert result == expected


@pytest.mark.parametrize(
    "nodes,edges,directed,expected",
    [
        pytest.param(
            ["a", "b", "c", "d", "e", "f"],
            [("a", "b"), ("a", "c"), ("d", "e")],
            False,
            True,
            id="undirected forest",
        ),
        pytest.param(
            ["a", "b", "c", "d", "e", "f"],
            [("a", "b"), ("a", "c"), ("d", "e")],
            True,
            True,
            id="directed forest",
        ),
        pytest.param(
            ["a", "b", "c", "d"],
            [("a", "b"), ("b", "c"), ("c", "a"), ("d", "a")],
            False,
            False,
            id="not a forest",
        ),
    ],
)
def test_is_forest(nodes, edges, directed, expected):
    g = ig.Graph(directed=directed)
    g.add_vertices(nodes)
    g.add_edges(edges)

    result = tree.is_forest(g)
    assert result == expected
