import pandas as pd
import polars as pl
import pytest

from tidygraph import Tidygraph
from tidygraph.exceptions import TidygraphValueError


@pytest.mark.parametrize(
    "nodes_df,edges_df,use_vids",
    [
        pytest.param(
            pl.DataFrame(
                {
                    "not_a_name": ["a", "b", "c"],
                }
            ),
            pl.DataFrame(
                {
                    "from": ["a", "b", "c"],
                    "to": ["b", "c", "a"],
                    "weight": [1.0, 2.0, 3.0],
                }
            ),
            False,
            id="invalid nodes dataframe",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "name": ["a", "b", "c"],
                }
            ),
            pl.DataFrame(
                {
                    "source": ["a", "b", "c"],
                    "target": ["b", "c", "a"],
                    "weight": [1.0, 2.0, 3.0],
                }
            ),
            False,
            id="invalid edges dataframe",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "name": ["a", "b", "c"],
                }
            ),
            pl.DataFrame(
                {
                    "from": ["a", "b", "c"],
                    "to": ["b", "c", "a"],
                    "weight": [1.0, 2.0, 3.0],
                }
            ),
            True,
            id="invalid vids dataframes",
        ),
    ],
)
def test_invalid_from_dataframe(nodes_df: pl.DataFrame, edges_df: pl.DataFrame, use_vids: bool):
    with pytest.raises(TidygraphValueError):
        _ = Tidygraph.from_dataframe(edges=edges_df, nodes=nodes_df, use_vids=use_vids)


def test_from_dataframe_with_nodes():
    edges_df = pl.DataFrame(
        {
            "from": ["a", "b", "c"],
            "to": ["b", "c", "a"],
            "weight": [1.0, 2.0, 3.0],
        }
    )
    nodes_df = pl.DataFrame(
        {
            "name": ["a", "b", "c"],
        }
    )
    g = Tidygraph.from_dataframe(edges=edges_df, nodes=nodes_df)

    vertex_df = g.vertex_dataframe
    edge_df = g.edge_dataframe

    expected_nodes = pd.Series(["a", "b", "c"])
    assert expected_nodes.equals(vertex_df["name"])

    description = g.describe()
    assert description == "undirected simple graph with 1 component(s)"

    expected_weights = pd.Series([1.0, 2.0, 3.0])
    assert expected_weights.equals(edge_df["weight"])


def test_from_dataframe_from_edges():
    edges_df = pl.DataFrame(
        {
            "from": ["a", "b", "c"],
            "to": ["b", "c", "a"],
            "weight": [1.0, 2.0, 3.0],
        }
    )
    g = Tidygraph.from_dataframe(edges=edges_df)

    vertex_df = g.vertex_dataframe
    edge_df = g.edge_dataframe

    expected_nodes = pd.Series(["a", "b", "c"])
    assert expected_nodes.equals(vertex_df["name"])

    description = g.describe()
    assert description == "undirected simple graph with 1 component(s)"

    expected_weights = pd.Series([1.0, 2.0, 3.0])
    assert expected_weights.equals(edge_df["weight"])


def test_from_dataframe_with_vids():
    nodes_df = pl.DataFrame({"name": ["a", "b", "c", "b"]})
    edges_df = pl.DataFrame(
        {
            "from": [1, 3],
            "to": [0, 2],
        }
    )
    g = Tidygraph.from_dataframe(edges=edges_df, nodes=nodes_df, use_vids=True)

    vertex_df = g.vertex_dataframe
    edges_df = g.edge_dataframe

    expected_nodes = pd.Series(["a", "b", "c", "b"])
    assert expected_nodes.equals(vertex_df["name"])
