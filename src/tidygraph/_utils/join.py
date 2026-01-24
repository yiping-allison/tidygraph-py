from collections.abc import Iterable

import igraph as ig
import pandas as pd

from tidygraph._utils.const import ReservedKeywords
from tidygraph.activate import ActiveType


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
    x_tmp["_index"] = range(len(x_tmp))

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
            y.drop(columns=[col for col in ReservedKeywords.EDGES if col not in [*on, "node ID"]], inplace=True)
            x_tmp_new = edgelist_with_mirror.merge(y, how="right_anti", on=on, suffixes=(lsuffix, rsuffix)).dropna(
                axis=1, how="all"
            )
            x_tmp = pd.concat([x_tmp, x_tmp_new], ignore_index=True)
        else:
            x_tmp = x_tmp.merge(y, how="outer", on=on, suffixes=(lsuffix, rsuffix))
    else:
        x_tmp = x_tmp.merge(y, how="outer", on=on, suffixes=(lsuffix, rsuffix))

    x_tmp.dropna(axis=1, how="all", inplace=True)
    new_rows = x_tmp["_index"].isna()
    x_tmp.drop(columns=["_index"], inplace=True)
    if active == ActiveType.NODES:
        g.add_vertices(new_rows.sum())
    elif active == ActiveType.EDGES:
        new_edges = x_tmp.loc[new_rows, ["source", "target"]].to_numpy()
        g.add_edges(new_edges)

    target = g.vs if active == ActiveType.NODES else g.es
    reserved = ReservedKeywords.NODES if active == ActiveType.NODES else ReservedKeywords.EDGES
    for col in x_tmp.columns:
        if col in reserved:
            continue

        target[col] = x_tmp[col]


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
    x_tmp["_index"] = range(len(x_tmp))

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
        g.delete_vertices(to_remove["_index"])
    elif active == ActiveType.EDGES and not to_remove.empty:
        source = to_remove["source"].to_numpy()
        target = to_remove["target"].to_numpy()
        # igraph strictly requires tuples
        edges = tuple(zip(source, target, strict=True))
        g.delete_edges(edges)

    x_tmp_merged.drop(columns=["_index"], inplace=True)

    target = g.vs if active == ActiveType.NODES else g.es
    reserved = ReservedKeywords.NODES if active == ActiveType.NODES else ReservedKeywords.EDGES
    for col in x_tmp_merged.columns:
        if col in reserved:
            continue

        target[col] = x_tmp_merged[col]


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
    x_tmp["_index"] = range(len(x_tmp))

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
    x_tmp.drop(columns=["_index"], inplace=True)

    target = g.vs if active == ActiveType.NODES else g.es
    reserved = ReservedKeywords.NODES if active == ActiveType.NODES else ReservedKeywords.EDGES
    for col in x_tmp.columns:
        if col in reserved:
            continue

        target[col] = x_tmp[col]
