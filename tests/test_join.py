from typing import Any, Literal

import igraph as ig
import numpy as np
import pandas as pd
import pytest

from tidygraph import Tidygraph
from tidygraph._utils import RESERVED_JOIN_KEYWORD, ReservedGraphKeywords
from tidygraph.activate import ActiveType
from tidygraph.exceptions import TidygraphValueError

N: int = 4


@pytest.fixture(scope="function", params=["directed", "undirected"])
def graph(request) -> ig.Graph:
    """Creates a sample diamond graph for tests."""
    kind = request.param
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ]
    max = 26  # assume we have max of 26 nodes
    start = ord("a")
    g = ig.Graph(
        n=N,
        directed=(kind == "directed"),
        edges=edges,
        vertex_attrs={"name": [chr(start + (char % max)) for char in range(N)]},
    )

    return g


@pytest.fixture(scope="module", params=[ActiveType.NODES, ActiveType.EDGES])
def active_type(request) -> ActiveType:
    return request.param


def test_invalid_input_raises(active_type: ActiveType, graph: ig.Graph) -> None:
    tg = Tidygraph(graph=graph)
    # reserved join keyword should error
    with pytest.raises(TidygraphValueError):
        _ = tg.activate(active_type).join(
            pd.DataFrame(
                {
                    RESERVED_JOIN_KEYWORD: [1, 2, 3],
                    "name": ["x", "y", "z"],
                }
            ),
            how="inner",
        )


@pytest.mark.parametrize(
    "y,expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["e"],
                }
            ),
            {
                "num_vertices": 5,
                "attributes": {
                    "name": pd.Series(["a", "b", "c", "d", "e"]),
                },
            },
            id="add new node",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b"],
                    "new_attr": ["hello", "world"],
                }
            ),
            {
                "num_vertices": 4,
                "attributes": {
                    "name": pd.Series(["a", "b", "c", "d"]),
                    "new_attr": pd.Series(["hello", "world", np.nan, np.nan]),
                },
            },
            id="new attr to existing nodes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b", "a"],
                    "new_attr": ["hello", "world", "!"],
                }
            ),
            {
                "num_vertices": 5,
                "attributes": {
                    "name": pd.Series(["a", "b", "c", "d", "a"]),
                    "new_attr": pd.Series(["hello", "world", np.nan, np.nan, "!"]),
                },
            },
            id="cartesian creates exploded nodes",
        ),
    ],
)
def test_outer_join_nodes(graph: ig.Graph, y: pd.DataFrame, expected: dict[str, Any]) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.NODES).join(y, how="outer")

    node_df = tg.vertex_dataframe

    assert len(node_df) == expected["num_vertices"]

    expected_attributes = expected["attributes"]
    attr_cols = [col for col in node_df.columns if col not in ReservedGraphKeywords.NODES]
    assert len(attr_cols) == len(expected_attributes.keys())
    for col, expected_attr in expected_attributes.items():
        assert expected_attr.equals(node_df[col])


@pytest.mark.parametrize(
    "y,expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a"],
                    "to": ["d"],
                }
            ),
            {
                "num_edges": 5,
                "attributes": {},
            },
            id="add new edge",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a"],
                    "to": ["b"],
                    "new_attr": ["hello"],
                }
            ),
            {
                "num_edges": 4,
                "attributes": {
                    "new_attr": pd.Series(["hello", np.nan, np.nan, np.nan]),
                },
            },
            id="with attrs on existing edges",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "a"],
                    "to": ["b", "b"],
                    "new_attr": ["hello", "!"],
                }
            ),
            {
                "num_edges": 5,
                "attributes": {
                    "new_attr": pd.Series(["hello", np.nan, np.nan, np.nan, "!"]),
                },
            },
            id="cartesian creates exploded edges",
        ),
    ],
)
def test_outer_join_edges(graph: ig.Graph, y: pd.DataFrame, expected: dict[str, Any]) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.EDGES).join(y, how="outer")

    edge_df = tg.edge_dataframe

    assert len(edge_df) == expected["num_edges"]

    expected_attributes = expected["attributes"]
    attr_cols = [col for col in edge_df.columns if col not in ReservedGraphKeywords.EDGES]
    assert len(attr_cols) == len(expected_attributes.keys())
    for col, expected_attr in expected_attributes.items():
        assert expected_attr.equals(edge_df[col])


@pytest.mark.parametrize(
    "y,expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b"],
                }
            ),
            {
                "num_vertices": 2,
                "attributes": {
                    "name": pd.Series(["a", "b"]),
                },
            },
            id="inner join drops non-matching nodes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b"],
                    "new_attr": ["hello", "world"],
                }
            ),
            {
                "num_vertices": 2,
                "attributes": {
                    "name": pd.Series(["a", "b"]),
                    "new_attr": pd.Series(["hello", "world"]),
                },
            },
            id="inner join with new attributes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "a"],
                    "new_attr": ["hello", "!"],
                }
            ),
            {
                "num_vertices": 2,
                "attributes": {
                    "name": pd.Series(["a", "a"]),
                    "new_attr": pd.Series(["hello", "!"]),
                },
            },
            id="cartesian creates exploded nodes",
        ),
    ],
)
def test_inner_join_nodes(graph: ig.Graph, y: pd.DataFrame, expected: dict[str, Any]) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.NODES).join(y, how="inner")

    node_df = tg.vertex_dataframe

    assert len(node_df) == expected["num_vertices"]

    expected_attributes = expected["attributes"]
    attr_cols = [col for col in node_df.columns if col not in ReservedGraphKeywords.NODES]
    assert len(attr_cols) == len(expected_attributes.keys())
    for col, expected_attr in expected_attributes.items():
        assert expected_attr.equals(node_df[col])


@pytest.mark.parametrize(
    "y,expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a"],
                    "to": ["d"],
                }
            ),
            {
                "num_edges": 0,
                "attributes": {},
            },
            id="inner join drops non-matching edges",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "a"],
                    "to": ["c", "b"],
                    "new_attr": ["hello", None],
                }
            ),
            {
                "num_edges": 2,
                "attributes": {
                    "new_attr": pd.Series([np.nan, "hello"]),
                },
            },
            id="inner join with new attributes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "a"],
                    "to": ["b", "b"],
                    "new_attr": ["hello", "!"],
                }
            ),
            {
                "num_edges": 2,
                "attributes": {
                    "new_attr": pd.Series(["hello", "!"]),
                },
            },
            id="cartesian creates exploded edges",
        ),
    ],
)
def test_inner_join_edges(graph: ig.Graph, y: pd.DataFrame, expected: dict[str, Any]) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.EDGES).join(y, how="inner")

    edge_df = tg.edge_dataframe

    assert len(edge_df) == expected["num_edges"]

    expected_attributes = expected["attributes"]
    attr_cols = [col for col in edge_df.columns if col not in ReservedGraphKeywords.EDGES]
    assert len(attr_cols) == len(expected_attributes.keys())
    for col, expected_attr in expected_attributes.items():
        assert expected_attr.equals(edge_df[col])


@pytest.mark.parametrize(
    "y,expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["g"],
                }
            ),
            {
                "num_vertices": 4,
                "attributes": {
                    "name": pd.Series(["a", "b", "c", "d"]),
                },
            },
            id="drop non-matching nodes in y",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["b", "a"],
                    "new_attr": ["hello", "world"],
                }
            ),
            {
                "num_vertices": 4,
                "attributes": {
                    "name": pd.Series(["a", "b", "c", "d"]),
                    "new_attr": pd.Series(["world", "hello", np.nan, np.nan]),
                },
            },
            id="left join with new attributes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "a"],
                    "new_attr": ["hello", "!"],
                }
            ),
            {
                "num_vertices": 5,
                "attributes": {
                    "name": pd.Series(["a", "b", "c", "d", "a"]),
                    "new_attr": pd.Series(["hello", np.nan, np.nan, np.nan, "!"]),
                },
            },
            id="cartesian explodes nodes",
        ),
    ],
)
def test_left_join_nodes(graph: ig.Graph, y: pd.DataFrame, expected: dict[str, Any]) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.NODES).join(y, how="left")

    node_df = tg.vertex_dataframe

    assert len(node_df) == expected["num_vertices"]

    expected_attributes = expected["attributes"]
    attr_cols = [col for col in node_df.columns if col not in ReservedGraphKeywords.NODES]
    assert len(attr_cols) == len(expected_attributes.keys())
    for col, expected_attr in expected_attributes.items():
        assert expected_attr.equals(node_df[col])


@pytest.mark.parametrize(
    "y,expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a"],
                    "to": ["d"],
                }
            ),
            {
                "num_edges": 4,
                "attributes": {},
            },
            id="drop non-matching edges in y",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a"],
                    "to": ["b"],
                    "new_attr": ["hello"],
                }
            ),
            {
                "num_edges": 4,
                "attributes": {
                    "new_attr": pd.Series(["hello", np.nan, np.nan, np.nan]),
                },
            },
            id="accept new attributes on existing edge",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "a"],
                    "to": ["b", "b"],
                    "new_attr": ["hello", "world"],
                }
            ),
            {
                "num_edges": 5,
                "attributes": {
                    "new_attr": pd.Series(["hello", np.nan, np.nan, np.nan, "world"]),
                },
            },
            id="cartesian explodes edges",
        ),
    ],
)
def test_left_join_edges(graph: ig.Graph, y: pd.DataFrame, expected: dict[str, Any]) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.EDGES).join(y, how="left")

    edge_df = tg.edge_dataframe

    assert len(edge_df) == expected["num_edges"]

    expected_attributes = expected["attributes"]
    attr_cols = [col for col in edge_df.columns if col not in ReservedGraphKeywords.EDGES]
    assert len(attr_cols) == len(expected_attributes.keys())
    for col, expected_attr in expected_attributes.items():
        assert expected_attr.equals(edge_df[col])


@pytest.mark.parametrize(
    "y,expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b"],
                }
            ),
            {
                "num_vertices": 2,
                "attributes": {
                    "name": pd.Series(["a", "b"]),
                },
            },
            id="right join drops non-matching x nodes in graph",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "g", "b"],
                }
            ),
            {
                "num_vertices": 3,
                "attributes": {
                    "name": pd.Series(["a", "b", "g"]),
                },
            },
            id="right join with new node in y",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b", "c"],
                    "new_attr": ["hello", "world", None],
                }
            ),
            {
                "num_vertices": 3,
                "attributes": {
                    "name": pd.Series(["a", "b", "c"]),
                    "new_attr": pd.Series(["hello", "world", np.nan]),
                },
            },
            id="right join with new attributes in y",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b", "a"],
                    "new_attr": ["hello", "!", "world"],
                }
            ),
            {
                "num_vertices": 3,
                "attributes": {
                    "name": pd.Series(["a", "b", "a"]),
                    "new_attr": pd.Series(["hello", "!", "world"]),
                },
            },
            id="cartesian explodes nodes",
        ),
    ],
)
def test_right_join_nodes(graph: ig.Graph, y: pd.DataFrame, expected: dict[str, Any]) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.NODES).join(y, how="right")

    node_df = tg.vertex_dataframe

    assert len(node_df) == expected["num_vertices"]

    expected_attributes = expected["attributes"]
    attr_cols = [col for col in node_df.columns if col not in ReservedGraphKeywords.NODES]
    assert len(attr_cols) == len(expected_attributes.keys())
    for col, expected_attr in expected_attributes.items():
        assert expected_attr.equals(node_df[col])


@pytest.mark.parametrize(
    "y,expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a"],
                    "to": ["b"],
                }
            ),
            {
                "num_edges": 1,
                "attributes": {},
            },
            id="right join drops non-matching x edges in graph",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "a"],
                    "to": ["d", "b"],
                }
            ),
            {
                "num_edges": 2,
                "attributes": {},
            },
            id="right join with new edge in y",
        ),
        pytest.param(
            pd.DataFrame({"from": ["b", "a", "b"], "to": ["d", "b", "d"], "new_attrs": ["hello", "world", "!"]}),
            {"num_edges": 3, "attributes": {"new_attrs": pd.Series(["world", "hello", "!"])}},
            id="cartesian explodes edges",
        ),
    ],
)
def test_right_join_edges(graph: ig.Graph, y: pd.DataFrame, expected: dict[str, Any]) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.EDGES).join(y, how="right")

    edge_df = tg.edge_dataframe

    assert len(edge_df) == expected["num_edges"]

    expected_attributes = expected["attributes"]
    attr_cols = [col for col in edge_df.columns if col not in ReservedGraphKeywords.EDGES]
    assert len(attr_cols) == len(expected_attributes.keys())
    for col, expected_attr in expected_attributes.items():
        assert expected_attr.equals(edge_df[col])


@pytest.mark.parametrize(
    "how,y,expected",
    [
        pytest.param(
            "outer",
            pd.DataFrame(
                {
                    "from": ["a"],
                    "to": ["b"],
                    "weight": [5.3],
                }
            ),
            {
                "attributes": {
                    "x": pd.Series([1.0, 2.0, 3.0, 4.0]),
                    "y": pd.Series([5.3, np.nan, np.nan, np.nan]),
                }
            },
            id="outer join with conflicting column name",
        ),
        pytest.param(
            "inner",
            pd.DataFrame(
                {
                    "from": ["a"],
                    "to": ["b"],
                    "weight": [5.3],
                }
            ),
            {
                "num_edges": 1,
                "attributes": {
                    "x": pd.Series([1.0]),
                    "y": pd.Series([5.3]),
                },
            },
            id="inner join with conflicting column name",
        ),
        pytest.param(
            "left",
            pd.DataFrame(
                {
                    "from": ["a"],
                    "to": ["b"],
                    "weight": [5.3],
                }
            ),
            {
                "attributes": {
                    "x": pd.Series([1.0, 2.0, 3.0, 4.0]),
                    "y": pd.Series([5.3, np.nan, np.nan, np.nan]),
                }
            },
            id="left join with conflicting column name",
        ),
        pytest.param(
            "right",
            pd.DataFrame({"from": ["b"], "to": ["d"], "weight": [5.3]}),
            {"attributes": {"x": pd.Series([3.0]), "y": pd.Series([5.3])}},
            id="right join with conflicting column name",
        ),
    ],
)
def test_conflicted_joins(
    graph: ig.Graph, how: Literal["outer", "inner", "left", "right"], y: pd.DataFrame, expected: dict[str, Any]
) -> None:
    """Tests that we handle conflicts in column names by suffixing them."""
    original_weights = [1.0, 2.0, 3.0, 4.0]
    attr_name = "weight"
    graph.es[attr_name] = original_weights
    tg = Tidygraph(graph=graph).activate(ActiveType.EDGES).join(y, how=how)

    edges = tg.edge_dataframe

    attributes = expected["attributes"]
    expected_x_attr = attributes["x"]
    expected_y_attr = attributes["y"]

    left, right = f"{attr_name}.x", f"{attr_name}.y"
    assert expected_x_attr.equals(edges[left])
    assert expected_y_attr.equals(edges[right])
