from typing import Any, Callable, Literal

import igraph as ig
import pandas as pd
import pytest

from tidygraph import Tidygraph
from tidygraph.activate import ActiveType
from tidygraph.exceptions import TidygraphValueError


@pytest.fixture(scope="function")
def graph() -> ig.Graph:
    """Creates a sample diamond graph for tests."""
    n = 4
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ]
    max = 26  # assume we have max of 26 nodes
    start = ord("a")
    g = ig.Graph(n=n, edges=edges, vertex_attrs={"name": [chr(start + (char % max)) for char in range(n)]})

    return g


def test_centrality_raises_on_unknown_type(graph: ig.Graph):
    with pytest.raises(TidygraphValueError):
        tg = Tidygraph(graph=graph)
        _ = tg.centrality(how="this should fail")


@pytest.mark.parametrize(
    "how,inputs",
    [
        pytest.param(
            "degree",
            {"what": "something"},
        )
    ],
)
def test_centrality_fails_on_unknown_args(graph: ig.Graph, how: Literal["degree", "alpha"], inputs: dict[str, Any]):
    with pytest.raises(TidygraphValueError):
        tg = Tidygraph(graph=graph)
        _ = tg.activate(ActiveType.NODES).centrality(how=how, **inputs)


@pytest.mark.parametrize(
    "how,weights_func", [pytest.param("degree", lambda x: pd.Series([1.0] * 4), id="degree with weights")]
)
def test_centrality_accepts_custom_weights(
    graph: ig.Graph, how: Literal["degree", "alpha"], weights_func: Callable[[pd.DataFrame], pd.Series]
):
    tg = Tidygraph(graph=graph)
    actual = tg.activate(ActiveType.NODES).centrality(how=how, weights=weights_func)
    actual_len = len(actual) if isinstance(actual, list) else 1
    assert actual_len == 4


@pytest.mark.parametrize(
    "how,input",
    [
        pytest.param(
            "degree",
            ActiveType.EDGES,
            id="degree requires nodes",
        )
    ],
)
def test_centrality_requires_correct_activation(graph: ig.Graph, how: Literal["degree", "alpha"], input: ActiveType):
    with pytest.raises(TidygraphValueError):
        tg = Tidygraph(graph=graph)
        _ = tg.activate(input).centrality(how=how)


@pytest.mark.parametrize("how,expected", [pytest.param("degree", 4.0, id="degree")])
def test_centrality_returns_expected_len(graph: ig.Graph, how: str, expected: float):
    tg = Tidygraph(graph=graph)
    actual = tg.centrality(how=how)
    actual_len = len(actual) if isinstance(actual, list) else 1
    assert actual_len == expected, f"Expected {how} results to have {expected} items but got {actual}"
