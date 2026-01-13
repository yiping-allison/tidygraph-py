from dataclasses import dataclass
from typing import Callable

import pandas as pd
import polars as pl
import pytest

from tidygraph import Tidygraph
from tidygraph.activate import ActiveType
from tidygraph.exceptions import TidygraphValueError


def test_mutate_raises():
    nodes_df = pl.DataFrame(
        {
            "name": ["a", "b", "c"],
        }
    )
    edges_df = pl.DataFrame(
        {
            "from": ["a", "b"],
            "to": ["b", "c"],
            "weights": [1, 2],
        }
    )
    tg = Tidygraph.from_dataframe(edges=edges_df, nodes=nodes_df)
    with pytest.raises(TidygraphValueError, match="attempted to derive expr from non-existent column: 'weights'"):
        tg = tg.activate(ActiveType.NODES).mutate(distance=lambda x: 1 - x["weights"])


@dataclass
class _Expected:
    attributes: dict[str, list[str]]
    edges_df: pl.DataFrame
    nodes_df: pl.DataFrame


@pytest.mark.parametrize(
    "nodes,edges,mutations,expected",
    [
        pytest.param(
            pl.DataFrame(
                {
                    "name": ["a", "b", "c"],
                }
            ),
            pl.DataFrame(
                {
                    "from": ["a", "b"],
                    "to": ["b", "c"],
                    "weights": [1, 2],
                }
            ),
            {"distance": lambda x: 1 - x["weights"]},
            _Expected(
                attributes={
                    "graph": [],
                    "nodes": ["name"],
                    "edges": ["source", "target", "weights", "distance"],
                },
                edges_df=pl.DataFrame(
                    {
                        "edge ID": [0, 1],
                        "source": [0, 1],
                        "target": [1, 2],
                        "weights": [1, 2],
                        "distance": [0, -1],
                    }
                ),
                nodes_df=pl.DataFrame(
                    {
                        "vertex ID": [0, 1, 2],
                        "name": ["a", "b", "c"],
                    }
                ),
            ),
            id="derive new edge attr",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "name": ["a", "b", "c"],
                }
            ),
            pl.DataFrame(
                {
                    "from": ["a", "b"],
                    "to": ["b", "c"],
                    "weights": [1, 2],
                }
            ),
            {"weights": lambda x: x["weights"] * 10},
            _Expected(
                attributes={
                    "graph": [],
                    "nodes": ["name"],
                    "edges": ["source", "target", "weights"],
                },
                edges_df=pl.DataFrame(
                    {
                        "edge ID": [0, 1],
                        "source": [0, 1],
                        "target": [1, 2],
                        "weights": [10, 20],
                    }
                ),
                nodes_df=pl.DataFrame(
                    {
                        "vertex ID": [0, 1, 2],
                        "name": ["a", "b", "c"],
                    }
                ),
            ),
            id="modify existing edge attr",
        ),
    ],
)
def test_simple_mutate_edges(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    mutations: dict[str, Callable[[pd.DataFrame], None]],
    expected: _Expected,
):
    tg = Tidygraph.from_dataframe(edges=edges, nodes=nodes)
    tg = tg.activate(ActiveType.EDGES).mutate(**mutations)

    attributes = set(tg.attributes)
    assert set(expected.attributes) == attributes

    edges_result = tg.edge_dataframe
    nodes_result = tg.vertex_dataframe

    assert expected.edges_df.to_pandas().set_index("edge ID").equals(edges_result)
    assert expected.nodes_df.to_pandas().set_index("vertex ID").equals(nodes_result)


@pytest.mark.parametrize(
    "nodes,edges,mutations,expected",
    [
        pytest.param(
            pl.DataFrame(
                {
                    "name": ["a", "b", "c"],
                }
            ),
            pl.DataFrame(
                {
                    "from": ["a", "b"],
                    "to": ["b", "c"],
                    "weights": [1, 2],
                }
            ),
            {"name_upper": lambda x: x["name"].str.upper()},
            _Expected(
                attributes={
                    "graph": [],
                    "nodes": ["name", "name_upper"],
                    "edges": ["source", "target", "weights"],
                },
                edges_df=pl.DataFrame(
                    {
                        "edge ID": [0, 1],
                        "source": [0, 1],
                        "target": [1, 2],
                        "weights": [1, 2],
                    }
                ),
                nodes_df=pl.DataFrame(
                    {
                        "vertex ID": [0, 1, 2],
                        "name": ["a", "b", "c"],
                        "name_upper": ["A", "B", "C"],
                    }
                ),
            ),
            id="derive new node attr",
        ),
        pytest.param(
            pl.DataFrame({"name": ["a", "b", "c"], "tag": ["apples", "bananas", "cherries"]}),
            pl.DataFrame(
                {
                    "from": ["a", "b"],
                    "to": ["b", "c"],
                    "weights": [1, 2],
                }
            ),
            {"tag": lambda x: "fruit_" + x["tag"]},
            _Expected(
                attributes={
                    "graph": [],
                    "nodes": ["name"],
                    "edges": ["source", "target", "weights"],
                },
                edges_df=pl.DataFrame(
                    {
                        "edge ID": [0, 1],
                        "source": [0, 1],
                        "target": [1, 2],
                        "weights": [1, 2],
                    }
                ),
                nodes_df=pl.DataFrame(
                    {
                        "vertex ID": [0, 1, 2],
                        "name": ["a", "b", "c"],
                        "tag": ["fruit_apples", "fruit_bananas", "fruit_cherries"],
                    }
                ),
            ),
            id="modify existing node attr",
        ),
    ],
)
def test_simple_mutate_nodes(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    mutations: dict[str, Callable[[pd.DataFrame], None]],
    expected: _Expected,
):
    tg = Tidygraph.from_dataframe(edges=edges, nodes=nodes)
    tg = tg.activate(ActiveType.NODES).mutate(**mutations)

    attributes = set(tg.attributes)
    assert set(expected.attributes) == attributes

    edges_result = tg.edge_dataframe
    nodes_result = tg.vertex_dataframe

    assert expected.edges_df.to_pandas().set_index("edge ID").equals(edges_result)
    assert expected.nodes_df.to_pandas().set_index("vertex ID").equals(nodes_result)


def test_chained_mutations():
    nodes_df = pl.DataFrame(
        {
            "name": ["a", "b", "c"],
        }
    )
    edges_df = pl.DataFrame(
        {
            "from": ["a", "b"],
            "to": ["b", "c"],
            "weights": [1, 2],
        }
    )
    tg = Tidygraph.from_dataframe(edges=edges_df, nodes=nodes_df)
    tg = (
        tg.activate(ActiveType.EDGES)
        .mutate(distance=lambda x: 1 - x["weights"])
        .activate(ActiveType.NODES)
        .mutate(name_upper=lambda x: x["name"].str.upper())
    )

    expected_edges_df = pl.DataFrame(
        {
            "edge ID": [0, 1],
            "source": [0, 1],
            "target": [1, 2],
            "weights": [1, 2],
            "distance": [0, -1],
        }
    )
    expected_nodes_df = pl.DataFrame(
        {
            "vertex ID": [0, 1, 2],
            "name": ["a", "b", "c"],
            "name_upper": ["A", "B", "C"],
        }
    )

    edges_result = tg.edge_dataframe
    nodes_result = tg.vertex_dataframe

    assert expected_edges_df.to_pandas().set_index("edge ID").equals(edges_result)
    assert expected_nodes_df.to_pandas().set_index("vertex ID").equals(nodes_result)
