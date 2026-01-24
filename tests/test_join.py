from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd
import pytest

from tidygraph import Tidygraph
from tidygraph._utils import ReservedKeywords
from tidygraph.activate import ActiveType


@pytest.fixture(scope="function", params=["directed", "undirected"])
def graph(request) -> ig.Graph:
    kind = request.param
    g = ig.Graph(directed=(kind == "directed"))
    g.add_vertices(["a", "b", "c"])
    g.add_edges([("a", "b"), ("b", "c")])

    return g


@dataclass
class _Expected:
    directed: dict[str, Any] = field(default_factory=dict)
    undirected: dict[str, Any] = field(default_factory=dict)


@pytest.mark.parametrize(
    "y,expected_set",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b", "c", "d"],
                }
            ),
            _Expected(
                undirected={
                    "num_nodes": 4,
                    "names": ["a", "b", "c", "d"],
                },
            ),
            id="new nodes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b"],
                }
            ),
            _Expected(
                undirected={
                    "num_nodes": 3,
                    "names": ["a", "b", "c"],
                },
            ),
            id="no new nodes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b", "c"],
                    "color": ["red", "blue", "green"],
                }
            ),
            _Expected(
                undirected={
                    "num_nodes": 3,
                    "names": ["a", "b", "c"],
                    "attributes": {
                        "a": {"color": "red"},
                        "b": {"color": "blue"},
                        "c": {"color": "green"},
                    },
                },
            ),
            id="new node attributes",
        ),
    ],
)
def test_outer_join_node(graph: ig.Graph, y: pd.DataFrame, expected_set: _Expected) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.NODES).join(y, how="outer")

    vertex_df = tg.vertex_dataframe

    # We do not separate by undirected vs undirected graphs since nodes are the same in both cases
    expected = expected_set.undirected

    assert vertex_df["name"].count() == expected["num_nodes"]

    if "attributes" not in expected:
        return

    expected_attrs = pd.DataFrame(expected["attributes"]).transpose().reset_index(drop=True)
    attr_cols = [col for col in expected_attrs.columns if col not in ReservedKeywords.NODES]
    actual_attrs = vertex_df[attr_cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_attrs, expected_attrs)


@pytest.mark.parametrize(
    "y,expected_set",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "b", "c"],
                    "to": ["b", "c", "a"],
                }
            ),
            _Expected(
                directed={
                    "expected_edges": 3,
                },
                undirected={
                    "expected_edges": 3,
                },
            ),
            id="one new edge",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "b"],
                    "to": ["b", "c"],
                }
            ),
            _Expected(
                directed={
                    "expected_edges": 2,
                },
                undirected={
                    "expected_edges": 2,
                },
            ),
            id="no new edges",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "c"],
                    "to": ["b", "b"],
                }
            ),
            _Expected(
                directed={
                    "expected_edges": 3,
                },
                undirected={
                    "expected_edges": 2,
                },
            ),
            id="reverse edge pair",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a"],
                    "to": ["b"],
                    "weight": [1.5],
                }
            ),
            _Expected(
                directed={
                    "expected_edges": 2,
                    "attributes": [
                        {"from": "a", "to": "b", "weight": 1.5},
                        {"from": "b", "to": "c", "weight": np.nan},
                    ],
                },
                undirected={
                    "expected_edges": 2,
                    "attributes": [
                        {"from": "a", "to": "b", "weight": 1.5},
                        {"from": "b", "to": "c", "weight": np.nan},
                    ],
                },
            ),
            id="with attribute on existing edge",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["c"],
                    "to": ["a"],
                    "weight": [2.5],
                }
            ),
            _Expected(
                directed={
                    "expected_edges": 3,
                    "attributes": [
                        {"from": "c", "to": "a", "weight": 2.5},
                        {"from": "a", "to": "b", "weight": np.nan},
                        {"from": "b", "to": "c", "weight": np.nan},
                    ],
                },
                undirected={
                    "expected_edges": 3,
                    "attributes": [
                        {"from": "a", "to": "c", "weight": 2.5},
                        {"from": "a", "to": "b", "weight": np.nan},
                        {"from": "b", "to": "c", "weight": np.nan},
                    ],
                },
            ),
            id="with attribute on new edge",
        ),
    ],
)
def test_outer_join_edge(graph: ig.Graph, y: pd.DataFrame, expected_set: _Expected) -> None:
    tg = Tidygraph(graph=graph)
    tg = tg.activate(ActiveType.EDGES).join(y, how="outer")
    node_df = tg.vertex_dataframe
    edge_df = tg.edge_dataframe

    # augment the edge dataframe with source and target names for easier testing
    edge_df["from"] = edge_df["source"].map(lambda x: node_df.loc[x, "name"])
    edge_df["to"] = edge_df["target"].map(lambda x: node_df.loc[x, "name"])

    expected = expected_set.directed if graph.is_directed() else expected_set.undirected

    assert len(edge_df) == expected["expected_edges"]

    attributes = expected.get("attributes", [])
    if not attributes:
        return

    expected_attrs = pd.DataFrame(attributes).reset_index(drop=True)
    actual_attrs = edge_df.reset_index(drop=True)

    attr_set = expected_attrs.merge(actual_attrs, how="left", on=["from", "to"], suffixes=("_expected", "_actual"))

    attr_cols = [col for col in expected_attrs.columns if col not in ReservedKeywords.EDGES]
    for col in attr_cols:
        pd.testing.assert_series_equal(
            attr_set[f"{col}_expected"],
            attr_set[f"{col}_actual"],
            check_index=False,
            check_names=False,
        )


@pytest.mark.parametrize(
    "y,expected_set",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["b", "c"],
                }
            ),
            _Expected(
                undirected={
                    "num_nodes": 2,
                    "names": ["b", "c"],
                },
            ),
            id="remove one existing node",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "b", "c", "d"],
                }
            ),
            _Expected(
                undirected={
                    "num_nodes": 3,
                    "names": ["a", "b", "c"],
                },
            ),
            id="no change when given extra node",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["a", "c"],
                    "color": ["red", "cyan"],
                }
            ),
            _Expected(
                undirected={
                    "num_nodes": 2,
                    "names": ["a", "c"],
                    "attributes": {
                        "a": {"color": "red"},
                        "c": {"color": "cyan"},
                    },
                },
            ),
            id="node attributes with removal",
        ),
    ],
)
def test_inner_join_node(graph: ig.Graph, y: pd.DataFrame, expected_set: _Expected) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.NODES).join(y, how="inner")

    vertex_df = tg.vertex_dataframe

    # We do not separate by undirected vs undirected graphs since nodes are the same in both cases
    expected = expected_set.undirected

    assert vertex_df["name"].count() == expected["num_nodes"]
    assert set(vertex_df["name"]) == set(expected["names"])

    if "attributes" not in expected:
        return

    expected_attrs = pd.DataFrame(expected["attributes"]).transpose().reset_index(drop=True)
    attr_cols = [col for col in expected_attrs.columns if col not in ReservedKeywords.NODES]
    actual_attrs = vertex_df[attr_cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_attrs, expected_attrs)


@pytest.mark.parametrize(
    "y,expected_set",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["b", "c"],
                    "to": ["c", "a"],
                }
            ),
            _Expected(
                directed={
                    "expected_edges": 1,
                },
                undirected={
                    "expected_edges": 1,
                },
            ),
            id="remove one existing edge",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "b", "c"],
                    "to": ["b", "c", "a"],
                    "color": ["red", "blue", "green"],
                }
            ),
            _Expected(
                directed={
                    "expected_edges": 2,
                    "attributes": [
                        {"from": "a", "to": "b", "color": "red"},
                        {"from": "b", "to": "c", "color": "blue"},
                    ],
                },
                undirected={
                    "expected_edges": 2,
                    "attributes": [
                        {"from": "a", "to": "b", "color": "red"},
                        {"from": "b", "to": "c", "color": "blue"},
                    ],
                },
            ),
            id="with attributes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["b"],
                    "to": ["c"],
                    "color": ["yellow"],
                }
            ),
            _Expected(
                directed={
                    "expected_edges": 1,
                    "attributes": [
                        {"from": "b", "to": "c", "color": "yellow"},
                    ],
                },
                undirected={
                    "expected_edges": 1,
                    "attributes": [
                        {"from": "b", "to": "c", "color": "yellow"},
                    ],
                },
            ),
            id="single attribute with out of order removal",
        ),
    ],
)
def test_inner_join_edge(graph: ig.Graph, y: pd.DataFrame, expected_set: _Expected) -> None:
    tg = Tidygraph(graph=graph)
    tg = tg.activate(ActiveType.EDGES).join(y, how="inner")
    node_df = tg.vertex_dataframe
    edge_df = tg.edge_dataframe

    # augment the edge dataframe with source and target names for easier testing
    edge_df["from"] = edge_df["source"].map(lambda x: node_df.loc[x, "name"])
    edge_df["to"] = edge_df["target"].map(lambda x: node_df.loc[x, "name"])

    expected = expected_set.directed if graph.is_directed() else expected_set.undirected

    assert len(edge_df) == expected["expected_edges"]

    attributes = expected.get("attributes", [])
    if not attributes:
        return

    expected_attrs = pd.DataFrame(attributes).reset_index(drop=True)
    actual_attrs = edge_df.reset_index(drop=True)

    attr_set = expected_attrs.merge(actual_attrs, how="left", on=["from", "to"], suffixes=("_expected", "_actual"))

    attr_cols = [col for col in expected_attrs.columns if col not in ReservedKeywords.EDGES]
    for col in attr_cols:
        pd.testing.assert_series_equal(
            attr_set[f"{col}_expected"],
            attr_set[f"{col}_actual"],
            check_index=False,
            check_names=False,
        )
