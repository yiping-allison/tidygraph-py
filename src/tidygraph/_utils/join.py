from collections.abc import Iterable

import igraph as ig
import numpy as np
import pandas as pd

from tidygraph._utils.const import ReservedGraphKeywords
from tidygraph.activate import ActiveType
from tidygraph.exceptions import TidygraphValueError


def outer_join(
    active: ActiveType,
    g: ig.Graph,
    y: pd.DataFrame,
    on: str | Iterable[str] | None = None,
    lsuffix: str = ".x",
    rsuffix: str = ".y",
) -> None:
    """Performs an outer join between the graph's active component (x) and a given DataFrame (y).

    Args:
        active (ActiveType): The active component of the graph (nodes or edges).
        g (ig.Graph): The igraph graph object.
        y (pd.DataFrame): The DataFrame to join with the graph's active component.
        on (str | Iterable[str] | None, optional): Column(s) to join on. Defaults to None.
        lsuffix (str, optional): Suffix to use for overlapping columns from the graph's active component. \
            Defaults to ".x".
        rsuffix (str, optional): Suffix to use for overlapping columns from the given (y) DataFrame. Defaults to ".y".
    """
    x_tmp = g.get_edge_dataframe() if active == ActiveType.EDGES else g.get_vertex_dataframe()
    x_tmp["_index"] = x_tmp.index.to_series()
    y = y.copy()

    if active == ActiveType.EDGES:
        # augment y with node IDs
        nodes_df = g.get_vertex_dataframe()
        id_map = nodes_df.set_index("name")
        name_to_index = pd.Series(data=nodes_df.index.to_numpy(), index=id_map.index)
        y["source"] = y["from"].map(name_to_index)
        y["target"] = y["to"].map(name_to_index)
        on = ["source", "target"]
        if not g.is_directed():
            # undirected graphs need to consider "mirrored" edges during joins. We do not want to keep
            # attributes for both (a->b) and (b->a) edges separately, so we handle this by:
            # 1. augmenting y with mirrored edges
            # 2. performing a left join between x_tmp and augmented y \
            #   (e.g., adding updated attributes to existing edges)
            # 3. creating mirrored edges
            # 4. performing a right-anti join to find edges in y that are not in x_tmp (in either direction)
            # 5. concatenating the results from step 2 and step 4 to complete the outer join
            y_mirror = y.rename(columns={"from": "to", "to": "from", "source": "target", "target": "source"})
            y_with_mirror = pd.concat([y, y_mirror], ignore_index=True)
            x_tmp = x_tmp.merge(y_with_mirror, how="left", on=on, suffixes=(lsuffix, rsuffix)).dropna(axis=1, how="all")
            edgelist_mirror = x_tmp.rename(columns={"source": "target", "target": "source"})
            edgelist_with_mirror = pd.concat([x_tmp, edgelist_mirror], ignore_index=True)
            # drop reserved columns from y since they are duplicate and are not needed in merge result
            y.drop(columns=[col for col in ReservedGraphKeywords.EDGES if col not in [*on, "node ID"]], inplace=True)
            x_tmp_new = edgelist_with_mirror.merge(y, how="right_anti", on=on, suffixes=(lsuffix, rsuffix)).dropna(
                axis=1, how="all"
            )
            x_tmp = pd.concat([x_tmp, x_tmp_new], ignore_index=True)
        else:
            x_tmp = x_tmp.merge(y, how="outer", on=on, suffixes=(lsuffix, rsuffix))
    else:
        x_tmp = x_tmp.merge(y, how="outer", on=on, suffixes=(lsuffix, rsuffix))

    x_tmp.dropna(axis=1, how="all", inplace=True)
    new_row_indices = x_tmp["_index"].isna()
    new_rows = x_tmp[new_row_indices]
    x_tmp = x_tmp[~new_row_indices]
    x_tmp = pd.concat([x_tmp, new_rows])
    x_tmp.drop(columns=["_index"], inplace=True)
    if active == ActiveType.NODES:
        g.add_vertices(len(new_rows))
    elif active == ActiveType.EDGES:
        new_edges = new_rows[["source", "target"]].to_numpy()
        g.add_edges(new_edges)

    _apply_attributes(active, g, x_tmp)


def inner_join(
    active: ActiveType,
    g: ig.Graph,
    y: pd.DataFrame,
    on: str | Iterable[str] | None = None,
    lsuffix: str = ".x",
    rsuffix: str = ".y",
) -> None:
    """Performs an inner join between the graph's active component (x) and a given DataFrame (y).

    Args:
        active (ActiveType): The active component of the graph (nodes or edges).
        g (ig.Graph): The igraph graph object.
        y (pd.DataFrame): The DataFrame to join with the graph's active component.
        on (str | Iterable[str] | None, optional): Column(s) to join on. Defaults to None.
        lsuffix (str, optional): Suffix to use for overlapping columns from the graph's active component. \
            Defaults to ".x".
        rsuffix (str, optional): Suffix to use for overlapping columns from the given (y) DataFrame. Defaults to ".y".
    """
    x_tmp = g.get_edge_dataframe() if active == ActiveType.EDGES else g.get_vertex_dataframe()
    y = y.copy()

    if active == ActiveType.EDGES:
        nodes_df = g.get_vertex_dataframe()
        id_map = nodes_df.set_index("name")
        name_to_index = pd.Series(data=nodes_df.index.to_numpy(), index=id_map.index)
        y["source"] = y["from"].map(name_to_index)
        y["target"] = y["to"].map(name_to_index)
        on = ["source", "target"]
        if not g.is_directed():
            y_mirror = y.rename(columns={"from": "to", "to": "from", "source": "target", "target": "source"})
            y = pd.concat([y, y_mirror], ignore_index=True)

    x_tmp_merged = x_tmp.merge(y, how="inner", on=on, suffixes=(lsuffix, rsuffix)).dropna(axis=1, how="all")
    # find existing elements that need to be removed (dropped after join)
    to_remove = x_tmp.merge(y, how="left_anti", on=on, suffixes=(lsuffix, rsuffix)).dropna(axis=1, how="all")

    if active == ActiveType.NODES and not to_remove.empty:
        g.delete_vertices(to_remove.index.to_numpy())
    elif active == ActiveType.EDGES and not to_remove.empty:
        source = to_remove["source"].to_numpy()
        target = to_remove["target"].to_numpy()
        # igraph strictly requires tuples
        edges = tuple(zip(source, target, strict=True))
        g.delete_edges(edges)

    _apply_attributes(active, g, x_tmp_merged)


def left_join(
    active: ActiveType,
    g: ig.Graph,
    y: pd.DataFrame,
    on: str | Iterable[str] | None = None,
    lsuffix: str = ".x",
    rsuffix: str = ".y",
) -> None:
    """Performs a left join between the graph's active component and a given DataFrame.

    Args:
        active (ActiveType): The active component of the graph (nodes or edges).
        g (ig.Graph): The igraph graph object.
        y (pd.DataFrame): The DataFrame to join with the graph's active component.
        on (str | Iterable[str] | None, optional): Column(s) to join on. Defaults to None.
        lsuffix (str, optional): Suffix to use for overlapping columns from the graph's active component. \
            Defaults to ".x".
        rsuffix (str, optional): Suffix to use for overlapping columns from the given (y) DataFrame. Defaults to ".y".
    """
    x_tmp = g.get_edge_dataframe() if active == ActiveType.EDGES else g.get_vertex_dataframe()
    y = y.copy()

    if active == ActiveType.EDGES:
        nodes_df = g.get_vertex_dataframe()
        id_map = nodes_df.set_index("name")
        name_to_index = pd.Series(data=nodes_df.index.to_numpy(), index=id_map.index)
        y["source"] = y["from"].map(name_to_index)
        y["target"] = y["to"].map(name_to_index)
        on = ["source", "target"]
        if not g.is_directed():
            y_mirror = y.rename(columns={"from": "to", "to": "from", "source": "target", "target": "source"})
            y = pd.concat([y, y_mirror], ignore_index=True)

    x_tmp = x_tmp.merge(y, how="left", on=on, suffixes=(lsuffix, rsuffix)).dropna(axis=1, how="all")

    _apply_attributes(active, g, x_tmp)


def right_join(
    active: ActiveType,
    g: ig.Graph,
    y: pd.DataFrame,
    on: str | Iterable[str] | None = None,
    lsuffix: str = ".x",
    rsuffix: str = ".y",
) -> None:
    """Performs a right join between the graph's active component and a given DataFrame.

    Args:
        active (ActiveType): The active component of the graph (nodes or edges).
        g (ig.Graph): The igraph graph object.
        y (pd.DataFrame): The DataFrame to join with the graph's active component.
        on (str | Iterable[str] | None, optional): Column(s) to join on. Defaults to None.
        lsuffix (str, optional): Suffix to use for overlapping columns from the graph's active component. \
            Defaults to ".x".
        rsuffix (str, optional): Suffix to use for overlapping columns from the given (y) DataFrame. Defaults to ".y".
    """
    x_tmp = g.get_edge_dataframe() if active == ActiveType.EDGES else g.get_vertex_dataframe()
    x_tmp["_index"] = x_tmp.index.to_series()
    y = y.copy()

    nodes_df = g.get_vertex_dataframe()
    id_map = nodes_df.set_index("name")
    name_to_index = pd.Series(data=nodes_df.index.to_numpy(), index=id_map.index)

    new_x_tmp: pd.DataFrame = pd.DataFrame()
    to_remove: pd.DataFrame = pd.DataFrame()

    if active == ActiveType.EDGES:
        y["source"] = y["from"].map(name_to_index)
        y["target"] = y["to"].map(name_to_index)
        if not y[["source", "target"]].notna().all().all():
            raise TidygraphValueError("Cannot perform edge join on non-existing nodes in the graph")

        on = ["source", "target"]
        if not g.is_directed():
            x_tmp_mirror = x_tmp.rename(columns={"source": "target", "target": "source"})
            x_tmp_with_mirror = pd.concat([x_tmp, x_tmp_mirror], ignore_index=False)
            y["_index"] = np.nan
            new_x_tmp = y.merge(x_tmp_with_mirror, how="left", on=on, suffixes=(lsuffix, rsuffix))
            # if index from x_tmp (graph) is non-null, keep it; else use index from y
            new_x_tmp["_index"] = new_x_tmp["_index.y"].combine_first(new_x_tmp["_index.x"])
            new_x_tmp.drop(columns=["_index.x", "_index.y"], inplace=True)
            new_x_tmp.rename(
                columns=lambda column: (  # pyright: ignore[reportAny]
                    column[:-2] + ".y"
                    if column.endswith(".x")
                    else column[:-2] + ".x"
                    if column.endswith(".y")
                    else column
                ),
                inplace=True,
            )
            y_mirror = y.rename(columns={"from": "to", "to": "from", "source": "target", "target": "source"})
            y_with_mirror = pd.concat([y, y_mirror], ignore_index=True)
            to_remove = y_with_mirror.merge(x_tmp, how="right_anti", on=on, suffixes=(lsuffix, rsuffix))
        else:
            y["_index"] = np.nan
            new_x_tmp = y.merge(x_tmp, how="left", on=on, suffixes=(lsuffix, rsuffix))
            new_x_tmp["_index"] = new_x_tmp["_index.y"].combine_first(new_x_tmp["_index.x"])
            new_x_tmp.drop(columns=["_index.x", "_index.y"], inplace=True)
            new_x_tmp.rename(
                columns=lambda column: (  # pyright: ignore[reportAny]
                    column[:-2] + ".y"
                    if column.endswith(".x")
                    else column[:-2] + ".x"
                    if column.endswith(".y")
                    else column
                ),
                inplace=True,
            )
            to_remove = y.merge(x_tmp, how="right_anti", on=on, suffixes=(lsuffix, rsuffix))
    else:
        y["_index"] = y["name"].map(name_to_index)
        new_x_tmp = x_tmp.merge(y, how="right", on=on, suffixes=(lsuffix, rsuffix))
        new_x_tmp.dropna(subset=new_x_tmp.columns.difference(["_index"]), inplace=True)
        to_remove = x_tmp.merge(y, how="left_anti", on=on, suffixes=(lsuffix, rsuffix)).dropna(axis=1, how="all")

    new_row_indices = new_x_tmp["_index"].isna()
    new_rows = new_x_tmp[new_row_indices]
    new_x_tmp = new_x_tmp[~new_row_indices]
    new_x_tmp = pd.concat([new_x_tmp, new_rows])
    if active == ActiveType.NODES:
        if not to_remove.empty:
            g.delete_vertices(to_remove.index.to_numpy())
        g.add_vertices(len(new_rows))
    elif active == ActiveType.EDGES:
        if not to_remove.empty:
            source = to_remove["source"].to_numpy()
            target = to_remove["target"].to_numpy()
            edges = tuple(zip(source, target, strict=True))
            g.delete_edges(edges)

        new_edges = new_rows[["source", "target"]].to_numpy()
        g.add_edges(new_edges)

    new_x_tmp.drop(columns=["_index"], inplace=True)

    _apply_attributes(active, g, new_x_tmp)


def _apply_attributes(
    active: ActiveType,
    g: ig.Graph,
    data: pd.DataFrame,
) -> None:
    """Internal helper to apply updated attributes to the graph."""
    target = g.vs if active == ActiveType.NODES else g.es

    # remove old attributes; these are already handled in merge
    old_attrs = set(target.attribute_names())
    for attr in old_attrs:
        del target[attr]

    reserved = ReservedGraphKeywords.NODES if active == ActiveType.NODES else ReservedGraphKeywords.EDGES
    for col in data.columns:
        if col in reserved:
            continue

        target[col] = data[col]
