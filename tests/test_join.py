from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd
import pytest

from tidygraph import Tidygraph
from tidygraph._utils import RESERVED_JOIN_KEYWORD, ReservedGraphKeywords
from tidygraph.activate import ActiveType
from tidygraph.exceptions import TidygraphValueError


@pytest.fixture(scope="module", params=["with_attrs", "no_attrs"])
def graph_base(request):
    """Returns pre-generated vertices and edges pairs for tests dependent on existing attributes flag."""
    kind = request.param
    vertices = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "c")]
    if kind == "no_attrs":
        return vertices, edges, None

    attrs = {"color": [f"attr_{i}" for i in range(len(edges))]}

    return vertices, edges, attrs


@pytest.fixture(scope="function", params=["directed", "undirected"])
def graph(request, graph_base: tuple[list[str], list[tuple[str, str, dict[str, str] | None]]]) -> ig.Graph:
    kind = request.param
    g = ig.Graph(directed=(kind == "directed"))
    vertices, edges, attrs = graph_base
    attrs = attrs or {}
    g.add_vertices(vertices)
    g.add_edges(edges, attrs)

    return g


@pytest.fixture(scope="module", params=[ActiveType.NODES, ActiveType.EDGES])
def active_type(request) -> ActiveType:
    return request.param


def test_invalid_input_raises(active_type: ActiveType, graph: ig.Graph) -> None:
    tg = Tidygraph(graph)
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
    attr_cols = [col for col in expected_attrs.columns if col not in ReservedGraphKeywords.NODES]
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
    # record the set of attributes that already exist for easier testing
    existing_attrs = set(graph.es.attribute_names())
    existing_edges_df = graph.get_edge_dataframe()

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

    # format our expected set if we are working with existing attributes
    expected_attrs = pd.DataFrame(attributes).reset_index(drop=True)
    for attr in expected_attrs.columns:
        if attr in existing_attrs:
            # add the value as "y" since we are merging to an existing
            expected_attrs[f"{attr}.y"] = expected_attrs[attr]

            def edge_attr_for_row(row, attr=attr):
                source = graph.vs.find(name=row["from"]).index
                target = graph.vs.find(name=row["to"]).index

                match = existing_edges_df.loc[
                    (existing_edges_df["source"] == source) & (existing_edges_df["target"] == target)
                ]

                return match[attr].iloc[0] if not match.empty else np.nan

            expected_attrs[f"{attr}.x"] = expected_attrs.apply(edge_attr_for_row, axis=1)

            expected_attrs.drop(columns=[attr], inplace=True)

    actual_attrs = edge_df.reset_index(drop=True)

    attr_set = expected_attrs.merge(actual_attrs, how="left", on=["from", "to"], suffixes=("_expected", "_actual"))

    attr_cols = [col for col in expected_attrs.columns if col not in ReservedGraphKeywords.EDGES]
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
    attr_cols = [col for col in expected_attrs.columns if col not in ReservedGraphKeywords.NODES]
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
    # record the set of attributes that already exist for easier testing
    existing_attrs = set(graph.es.attribute_names())
    existing_edges_df = graph.get_edge_dataframe()

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

    # format our expected set if we are working with existing attributes
    expected_attrs = pd.DataFrame(attributes).reset_index(drop=True)
    for attr in expected_attrs.columns:
        if attr in existing_attrs:
            # add the value as "y" since we are merging to an existing
            expected_attrs[f"{attr}.y"] = expected_attrs[attr]

            def edge_attr_for_row(row, attr=attr):
                source = graph.vs.find(name=row["from"]).index
                target = graph.vs.find(name=row["to"]).index

                match = existing_edges_df.loc[
                    (existing_edges_df["source"] == source) & (existing_edges_df["target"] == target)
                ]

                return match[attr].iloc[0] if not match.empty else np.nan

            expected_attrs[f"{attr}.x"] = expected_attrs.apply(edge_attr_for_row, axis=1)

            expected_attrs.drop(columns=[attr], inplace=True)

    actual_attrs = edge_df.reset_index(drop=True)

    attr_set = expected_attrs.merge(actual_attrs, how="left", on=["from", "to"], suffixes=("_expected", "_actual"))

    attr_cols = [col for col in expected_attrs.columns if col not in ReservedGraphKeywords.EDGES]
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
                    "name": ["d", "e"],
                }
            ),
            _Expected(
                undirected={
                    "num_nodes": 3,
                    "names": ["a", "b", "c"],
                },
            ),
            id="add no new nodes",
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
            id="add attributes",
        ),
    ],
)
def test_left_join_node(graph: ig.Graph, y: pd.DataFrame, expected_set: _Expected) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.NODES).join(y, how="left")

    vertex_df = tg.vertex_dataframe

    # We do not separate by undirected vs undirected graphs since nodes are the same in both cases
    expected = expected_set.undirected

    assert vertex_df["name"].count() == expected["num_nodes"]

    if "attributes" not in expected:
        return

    expected_attrs = pd.DataFrame(expected["attributes"]).transpose().reset_index(drop=True)
    attr_cols = [col for col in expected_attrs.columns if col not in ReservedGraphKeywords.NODES]
    actual_attrs = vertex_df[attr_cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_attrs, expected_attrs)


@pytest.mark.parametrize(
    "y,expected_set",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "b", "d"],
                    "to": ["b", "c", "e"],
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
            id="left join edges does not accept new edges",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "c"],
                    "to": ["b", "a"],
                    "weight": [1.5, 2.5],
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
            id="left join edges with attributes on existing edges",
        ),
    ],
)
def test_left_join_edge(graph: ig.Graph, y: pd.DataFrame, expected_set: _Expected) -> None:
    # record the set of attributes that already exist for easier testing
    existing_attrs = set(graph.es.attribute_names())
    existing_edges_df = graph.get_edge_dataframe()

    tg = Tidygraph(graph=graph)
    tg = tg.activate(ActiveType.EDGES).join(y, how="left")
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

    # format our expected set if we are working with existing attributes
    expected_attrs = pd.DataFrame(attributes).reset_index(drop=True)
    for attr in expected_attrs.columns:
        if attr in existing_attrs:
            # add the value as "y" since we are merging to an existing
            expected_attrs[f"{attr}.y"] = expected_attrs[attr]

            def edge_attr_for_row(row, attr=attr):
                source = graph.vs.find(name=row["from"]).index
                target = graph.vs.find(name=row["to"]).index

                match = existing_edges_df.loc[
                    (existing_edges_df["source"] == source) & (existing_edges_df["target"] == target)
                ]

                return match[attr].iloc[0] if not match.empty else np.nan

            expected_attrs[f"{attr}.x"] = expected_attrs.apply(edge_attr_for_row, axis=1)

            expected_attrs.drop(columns=[attr], inplace=True)

    actual_attrs = edge_df.reset_index(drop=True)

    attr_set = expected_attrs.merge(actual_attrs, how="left", on=["from", "to"], suffixes=("_expected", "_actual"))

    attr_cols = [col for col in expected_attrs.columns if col not in ReservedGraphKeywords.EDGES]
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
                    "name": ["d", "e"],
                }
            ),
            _Expected(
                undirected={
                    "num_nodes": 2,
                    "names": ["d", "e"],
                },
            ),
            id="add new nodes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "name": ["b", "c"],
                    "color": ["blue", "green"],
                }
            ),
            _Expected(
                undirected={
                    "num_nodes": 2,
                    "names": ["b", "c"],
                    "attributes": {
                        "b": {"color": "blue"},
                        "c": {"color": "green"},
                    },
                },
            ),
            id="existing nodes with attributes",
        ),
    ],
)
def test_right_join_node(graph: ig.Graph, y: pd.DataFrame, expected_set: _Expected) -> None:
    tg = Tidygraph(graph=graph).activate(ActiveType.NODES).join(y, how="right")

    vertex_df = tg.vertex_dataframe

    # We do not separate by undirected vs undirected graphs since nodes are the same in both cases
    expected = expected_set.undirected

    assert vertex_df["name"].count() == expected["num_nodes"]

    if "attributes" not in expected:
        return

    expected_attrs = pd.DataFrame(expected["attributes"]).transpose().reset_index(drop=True)
    attr_cols = [col for col in expected_attrs.columns if col not in ReservedGraphKeywords.NODES]
    actual_attrs = vertex_df[attr_cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_attrs, expected_attrs)


@pytest.mark.parametrize(
    "y,expected_set",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["a", "a"],
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
            id="with new edges",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "from": ["b", "c"],
                    "to": ["c", "a"],
                    "color": ["blue", "cyan"],
                }
            ),
            _Expected(
                directed={
                    "expected_edges": 2,
                    "attributes": [
                        {"from": "b", "to": "c", "color": "blue"},
                        {"from": "c", "to": "a", "color": "cyan"},
                    ],
                },
                undirected={
                    "expected_edges": 2,
                    "attributes": [
                        {"from": "a", "to": "c", "color": "cyan"},
                        {"from": "b", "to": "c", "color": "blue"},
                    ],
                },
            ),
            id="with attributes on new edges",
        ),
    ],
)
def test_right_join_edge(graph: ig.Graph, y: pd.DataFrame, expected_set: _Expected) -> None:
    # record the set of attributes that already exist for easier testing
    existing_attrs = set(graph.es.attribute_names())
    existing_edges_df = graph.get_edge_dataframe()

    tg = Tidygraph(graph=graph)
    tg = tg.activate(ActiveType.EDGES).join(y, how="right")
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

    # format our expected set if we are working with existing attributes
    expected_attrs = pd.DataFrame(attributes).reset_index(drop=True)
    for attr in expected_attrs.columns:
        if attr in existing_attrs:
            # add the value as "y" since we are merging to an existing
            expected_attrs[f"{attr}.y"] = expected_attrs[attr]

            def edge_attr_for_row(row, attr=attr):
                source = graph.vs.find(name=row["from"]).index
                target = graph.vs.find(name=row["to"]).index

                match = existing_edges_df.loc[
                    (existing_edges_df["source"] == source) & (existing_edges_df["target"] == target)
                ]

                return match[attr].iloc[0] if not match.empty else np.nan

            expected_attrs[f"{attr}.x"] = expected_attrs.apply(edge_attr_for_row, axis=1)

            expected_attrs.drop(columns=[attr], inplace=True)

    actual_attrs = edge_df.reset_index(drop=True)

    attr_set = expected_attrs.merge(actual_attrs, how="left", on=["from", "to"], suffixes=("_expected", "_actual"))

    attr_cols = [col for col in expected_attrs.columns if col not in ReservedGraphKeywords.EDGES]
    for col in attr_cols:
        pd.testing.assert_series_equal(
            attr_set[f"{col}_expected"],
            attr_set[f"{col}_actual"],
            check_index=False,
            check_names=False,
        )
