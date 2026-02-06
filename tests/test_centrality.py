from typing import Any, Callable

import igraph as ig
import pandas as pd
import pytest

from tidygraph import Tidygraph
from tidygraph.activate import ActiveType
from tidygraph.exceptions import TidygraphValueError
from tidygraph.tidygraph import CentralityKind

NODE_KINDS = ["degree", "harmonic", "betweenness", "pagerank", "closeness", "eigenvector"]
EDGE_KINDS = ["edge_betweenness"]
ALL = NODE_KINDS + EDGE_KINDS

KIND_MAPPING = {kind: ActiveType.NODES if kind in NODE_KINDS else ActiveType.EDGES for kind in ALL}


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


@pytest.fixture(scope="module")
def kind_mapping():
    return KIND_MAPPING


def test_centrality_raises_on_unknown_type(graph: ig.Graph):
    with pytest.raises(TidygraphValueError):
        tg = Tidygraph(graph=graph)
        _ = tg.centrality(how="this should fail")


@pytest.mark.parametrize(
    "how,inputs",
    [pytest.param(kind, {"what": "something"}, id=f"{kind} fails on unknown arg") for kind in ALL],
)
def test_centrality_fails_on_unknown_args(
    graph: ig.Graph, kind_mapping: dict[str, ActiveType], how: CentralityKind, inputs: dict[str, Any]
):
    with pytest.raises(TidygraphValueError):
        tg = Tidygraph(graph=graph)
        active = kind_mapping[how]
        _ = tg.activate(active).centrality(how=how, **inputs)


@pytest.mark.parametrize(
    "how,weights_func",
    [pytest.param(kind, lambda x: pd.Series([1.0] * 4), id=f"{kind} accepts weights param") for kind in ALL],
)
def test_centrality_accepts_custom_weights(
    graph: ig.Graph,
    kind_mapping: dict[str, ActiveType],
    how: CentralityKind,
    weights_func: Callable[[pd.DataFrame], pd.Series],
):
    tg = Tidygraph(graph=graph)
    active = kind_mapping[how]
    actual = tg.activate(active).centrality(how=how, weights=weights_func)
    actual_len = len(actual) if isinstance(actual, list) else 1
    assert actual_len == 4


@pytest.mark.parametrize(
    "how",
    [pytest.param(kind, id=f"{kind} requires {KIND_MAPPING[kind]}") for kind in ALL],
)
def test_centrality_requires_correct_activation(
    graph: ig.Graph, kind_mapping: dict[str, ActiveType], how: CentralityKind
):
    with pytest.raises(TidygraphValueError):
        kind = kind_mapping[how]
        wrong = ActiveType.EDGES if kind == ActiveType.NODES else ActiveType.NODES
        tg = Tidygraph(graph=graph)
        _ = tg.activate(wrong).centrality(how=how)


@pytest.mark.parametrize(
    "how,expected",
    [
        pytest.param(
            kind,
            4.0,
            id=f"{kind} returns expected length",
        )
        for kind in ALL
    ],
)
def test_centrality_returns_expected_len(
    graph: ig.Graph, kind_mapping: dict[str, ActiveType], how: str, expected: float
):
    tg = Tidygraph(graph=graph)
    active = kind_mapping[how]
    actual = tg.activate(active).centrality(how=how)
    actual_len = len(actual) if isinstance(actual, list) else 1
    assert actual_len == expected, f"Expected {how} results to have {expected} items but got {actual}"
