"""tidygraph module providing Tidygraph class for igraph graphs."""

from collections.abc import Iterable
from typing import Any, Callable, Literal

import igraph as ig
import narwhals as nw
import pandas as pd
from narwhals.typing import IntoDataFrame

from tidygraph._utils import (
    RESERVED_JOIN_KEYWORD,
    ReservedGraphKeywords,
    centrality_betweenness,
    centrality_closeness,
    centrality_degree,
    centrality_edge_betweenness,
    centrality_eigenvector,
    centrality_harmonic,
    centrality_pagerank,
    inner_join,
    is_forest,
    is_tree,
    left_join,
    outer_join,
    right_join,
)
from tidygraph.activate import ActiveState, ActiveType
from tidygraph.exceptions import TidygraphError, TidygraphValueError

__all = ["Tidygraph"]

CentralityKind = Literal[
    "degree", "harmonic", "betweenness", "edge_betweenness", "closeness", "eigenvector", "pagerank"
]


class Tidygraph:
    """A class for holding tidy-like interface for igraph graphs."""

    _activate: ActiveState
    _graph: ig.Graph

    def __init__(self, graph: ig.Graph) -> None:
        """Initializes a Tidygraph object.

        You most likely want to use the dataframe classmethod to create the object instead.

        Example:
            ```python
            tg = Tidygraph.from_dataframe(edges_df, nodes_df, directed=True)
            ```
        """
        node_attributes = set(graph.vs.attributes())
        if "name" not in node_attributes:
            raise TidygraphError(
                'Tidygraph is designed to work with named nodes. Please ensure that the "name" \
                attribute is present in the node attributes.'
            )

        self._activate = ActiveState()
        self._graph = graph

    @property
    def attributes(self) -> dict[str, list[str]]:
        """Returns the attributes stored in the graph.

        Returns:
            A dictionary of attributes mapped to the main graph, edges, and vertices.
        """
        return {
            "graph": self._graph.attributes(),
            "nodes": self._graph.vs.attributes(),
            "edges": self._graph.es.attributes(),
        }

    @property
    def vertex_dataframe(self) -> pd.DataFrame:
        """Returns the vertices and associated attributes as a pandas DataFrame.

        Requires pandas to be installed.

        Returns:
            A pandas DataFrame containing the vertices and their attributes.
        """
        return self._graph.get_vertex_dataframe()

    @property
    def edge_dataframe(self) -> pd.DataFrame:
        """Returns the edges and associated attributes as a pandas DataFrame.

        Requires pandas to be installed.

        Returns:
            A pandas DataFrame containing the edges and their attributes.
        """
        return self._graph.get_edge_dataframe()

    def degree(self, *args: int | list[int], **kwargs: object) -> int | list[int]:
        """Calculates the degrees associated with the specified vertices.

        View the referenced `igraph` documentation for detailed notes.

        Example:
            ```py
            # calculate the out degree for each node in a directed graph
            tg.degree(mode="out")

            # calculate the degree for vertices `[0, 1]`
            tg.degree([0, 1])
            ```

        Args:
            vertices (int | list[int], Optional): Subset of vertices to find degrees for.
            mode (str, Optional): specify `in`, `out`, or `all` for in-degree, out-degree, or sum of both.
            loops (bool, Optional): Whether self-loops should be counted, defaults to True.

        Returns:
            Either an integer or list of integer degrees.

        References:
            - https://python.igraph.org/en/1.0.0/api/igraph.GraphBase.html#degree
        """
        return self._graph.degree(*args, **kwargs)

    def centrality(self, how: CentralityKind, **kwargs: object) -> float | list[float]:
        """Calculate node and edge centrality.

        The "centrality" of a node measures the importance for it within the network.
        This method is a wrapper that exposes different centrality functions.

        Args:
            how (str): Which centrality measure to calculate.
            kwargs (dict[str, Any]): Other parameters that can be passed to target centrality function. Accepted \
                parameters differ depending on chosen variant.

        Returns:
            Either a float or list of floats representing calculated centrality.

        References:
            - https://tidygraph.data-imaginist.com/reference/centrality.html
        """
        dispatcher = {
            "degree": centrality_degree,
            "harmonic": centrality_harmonic,
            "betweenness": centrality_betweenness,
            "edge_betweenness": centrality_edge_betweenness,
            "closeness": centrality_closeness,
            "eigenvector": centrality_eigenvector,
            "pagerank": centrality_pagerank,
        }
        func = dispatcher.get(how)
        if not func:
            raise TidygraphValueError(f"unsupported centrality method: {how}")

        return func(self._activate.active, self._graph, **kwargs)

    def activate(self, what: ActiveType) -> "Tidygraph":
        """Sets the currently active type (nodes or edges).

        Args:
            what (ActiveType): The type to activate.
        """
        self._activate.active = what

        return self

    def mutate(self, **kwargs: Callable[[pd.DataFrame], None]) -> "Tidygraph":
        """Mutate the graph attributes.

        This method uses `pandas.DataFrame.assign` under the hood to perform mutations. Hence, the callable mapped to
        each key in `kwargs` should follow the same guidelines.

        Args:
            **kwargs: A variable number of keyword arguments representing the columns to modify/add and their \
                corresponding modifications.

        Example:
            ```python
            tg = Tidygraph.from_dataframe(edges_df, nodes_df)
            tg.activate(ActiveType.NODES).mutate(new_attr=lambda x: x["weights"] * 2)
            ```
        """
        # NOTE: We do not create a copy here since these dataframes are exported from igraph. (i.e., modification does
        #       NOT mutate the original igraph object)
        data_as_df = (
            self._graph.get_vertex_dataframe()
            if self._activate.active == ActiveType.NODES
            else self._graph.get_edge_dataframe()
        )
        try:
            modified_df = data_as_df.assign(**kwargs)
        except KeyError as e:
            raise TidygraphValueError(f"attempted to derive expr from non-existent column: {e}") from None

        reserved = (
            ReservedGraphKeywords.NODES if self._activate.active == ActiveType.NODES else ReservedGraphKeywords.EDGES
        )
        columns = list(modified_df.columns)
        start_index = next((i for i, x in enumerate(columns) if x not in reserved), None)
        if start_index is None:
            # Technically this should never happen but adding for completeness
            raise TidygraphValueError("no attributes to modify")

        # TODO: Consider optimizing this by only updating the modified columns instead of all columns.
        target = self._graph.vs if self._activate.active == ActiveType.NODES else self._graph.es
        attributes = list(modified_df.columns[start_index:])
        for attribute in attributes:
            target[attribute] = modified_df[attribute]

        return self

    def join(
        self,
        df: IntoDataFrame,
        on: str | Iterable[str] | None = None,
        how: Literal["left", "right", "outer", "inner"] = "left",
        lsuffix: str = ".x",
        rsuffix: str = ".y",
    ) -> "Tidygraph":
        """Join the active graph context with data from an external DataFrame.

        This method only supports specific `on` keyword arguments depending on the active context.
        For example, when joining to edges, the `on` argument is expected to be `["from", "to"]` (i.e. it \
            does not make sense to join on other edge attributes). Similarly, when joining to nodes, the `on` argument \
            is expected to be `"name"`.

        ! NOTE: This method mutates graph relationships depending on the current active context. For example, when in \
            the edges context, if a right join has edge pairs that do not exist in the current graph, those edges will \
            be added to the graph (missing nodes WILL NOT be created; they are expected to exist). Likewise, edge \
            pairs that exist in the current graph but not in the given dataframe will be removed from the graph.

        Args:
            df (IntoDataFrame): A dataframe-like object to join with the active graph context.
            on (str | Iterable[str] | None): The column(s) to join on. If None, the decision will be automatic based \
                on the active context. Defaults to None.
            how (Literal["left", "right", "inner", "outer"]): The type of join to perform. Defaults to "left".
            lsuffix (str): Suffix to use for overlapping columns from the active graph context. Defaults to ".x".
            rsuffix (str): Suffix to use for overlapping columns from the given dataframe. Defaults to ".y".

        Raises:
            TidygraphValueError: If the given DataFrame does not match expected requirements based on the active \
                context, if the given DataFrame contains a reserved key, or if an unsupported join type is specified.
        """
        df_pd = nw.from_native(df).to_pandas()
        cols = set(list(df_pd.columns))
        if RESERVED_JOIN_KEYWORD in cols:
            raise TidygraphValueError(
                f"column name '{RESERVED_JOIN_KEYWORD}' is reserved for internal use. Please rename the column."
            )

        if self._activate.active == ActiveType.EDGES:
            if not all([required in cols for required in ["from", "to"]]):
                raise TidygraphValueError('when joining to edges, dataframe must contain "from" and "to" columns.')
        else:
            if "name" not in cols:
                raise TidygraphValueError('when joining to nodes, dataframe must contain "name" column.')

        dispatcher = {
            "outer": outer_join,
            "inner": inner_join,
            "left": left_join,
            "right": right_join,
        }

        join_func = dispatcher.get(how)
        if not join_func:
            raise TidygraphValueError(f'unsupported join type: "{how}"')

        join_func(self._activate.active, self._graph, df_pd, on, lsuffix, rsuffix)

        return self

    def describe(self) -> str:
        """Describes the graph.

        Returns:
            A string containing basic information about the graph.
        """
        if not self._graph:
            return "An empty graph"

        properties: dict[str, Any] = {
            "simple": self._graph.is_simple(),
            "directed": self._graph.is_directed(),
            "bipartite": self._graph.is_bipartite(),
            "tree": is_tree(self._graph),
            "forest": is_forest(self._graph),
            "dag": self._graph.is_dag(),
        }
        description: list[str] = []
        if properties["tree"] or properties["forest"]:
            if properties["directed"]:
                description.append("rooted")
            else:
                description.append("unrooted")

            components = len(self._graph.components())
            if properties["directed"]:
                components = len(self._graph.components(mode="weak"))

            if components > 1:
                description.append(f"forest with {components} trees")
            else:
                description.append("tree")
        else:
            if properties["dag"]:
                description.append("directed acyclic")
            elif properties["bipartite"]:
                description.append("bipartite")
            elif properties["directed"]:
                description.append("directed")
            else:
                description.append("undirected")
            if properties["simple"]:
                description.append("simple graph")
            else:
                description.append("multigraph")

            components = len(self._graph.components())
            description.append(f"with {components} component(s)")

        return " ".join(description)

    @classmethod
    def from_dataframe(cls, edges: IntoDataFrame, nodes: IntoDataFrame, directed: bool = False) -> "Tidygraph":
        """Constructs a Tidygraph object from edges and nodes represented as dataframes.

        Args:
            edges: A dataframe-like object representing the edges of the graph. Must contain "from" and "to" columns.
            nodes: A dataframe-like object representing the nodes of the graph. Must contain a "name" column.
            directed: A boolean indicating whether the graph is directed. Defaults to False.

        Raises:
            TidygraphValueError: If either the nodes or edges dataframe do not match expected requirements.
        """
        edge_df = nw.from_native(edges).to_pandas()
        node_df = nw.from_native(nodes).to_pandas()

        node_df_cols = set(list(node_df.columns))
        required_node_cols = set(["name"])
        if not required_node_cols.issubset(node_df_cols):
            raise TidygraphValueError('nodes dataframe must contain "name" column')

        node_col_order = ["name"]
        node_attributes = list(node_df_cols - required_node_cols)
        node_df = node_df[node_col_order + node_attributes]

        edge_df_cols = set(list(edge_df.columns))
        required_edge_cols = set(["from", "to"])
        if not required_edge_cols.issubset(edge_df_cols):
            raise TidygraphValueError('edges dataframe must contain "from" and "to" columns')

        edge_col_order = ["from", "to"]
        edge_attributes = list(edge_df_cols - required_edge_cols)
        edge_df = edge_df[edge_col_order + edge_attributes]

        graph = ig.Graph.DataFrame(edges=edge_df, directed=directed, vertices=node_df, use_vids=False)

        return cls(graph=graph)
