from collections.abc import Iterable

import igraph as ig
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
    x = g.get_edge_dataframe() if active == ActiveType.EDGES else g.get_vertex_dataframe()
    x["_index"] = x.index.to_series()
    y = y.copy()

    if active == ActiveType.EDGES:
        # augment y with node IDs
        nodes_df = g.get_vertex_dataframe()
        id_map = nodes_df.set_index("name")
        name_to_index = pd.Series(data=nodes_df.index.to_numpy(), index=id_map.index)
        y["source"] = y["from"].map(name_to_index)
        y["target"] = y["to"].map(name_to_index)
        y.drop(columns=["from", "to"], inplace=True)
        on = ["source", "target"]
        if not g.is_directed():
            y_mirror = y.rename(columns={"source": "target", "target": "source"})
            y_with_mirror = pd.concat([y, y_mirror])
            x_merged = x.merge(y_with_mirror, how="left", on=on, suffixes=(lsuffix, rsuffix), indicator=True)
            explosion = x_merged[x_merged["_merge"] == "both"].groupby("_index").size()
            explosion = explosion[explosion > 1]
            for index in explosion.index:
                name = x.loc[index][["source", "target"]]
                new_filtered = x_merged[(x_merged["source"] == name["source"]) & (x_merged["target"] == name["target"])]
                old_filtered = x[(x["source"] == name["source"]) & (x["target"] == name["target"])]
                for i in range(1, len(new_filtered) - len(old_filtered) + 1):
                    index = new_filtered.iloc[i * -1].name
                    x_merged.at[index, "_merge"] = "right_only"
            x_merged_mirror = x_merged.rename(columns={"source": "target", "target": "source"})
            x_merged_with_mirror = pd.concat([x_merged, x_merged_mirror])
            y_merged = y.merge(
                x_merged_with_mirror.drop(columns=["_merge"]),
                how="left",
                on=on,
                suffixes=(lsuffix, rsuffix),
                indicator=True,
            )
            new_y = y_merged["_merge"] == "left_only"
            new_rows = y_merged[new_y]
            cols = [col for col in new_rows.columns if col.endswith(".x")]
            for col in cols:
                base = col[:-2]
                new_rows[base] = new_rows[col].combine_first(new_rows[f"{base}.y"])
                new_rows.drop(columns=[col, f"{base}.y"], inplace=True)
            new_rows = new_rows[new_rows.columns.intersection(y.columns)]
            new_rows["_merge"] = ["right_only"] * len(new_rows)
            x_merged = pd.concat([x_merged, new_rows])
        else:
            x_merged = x.merge(y, how="outer", on=on, suffixes=(lsuffix, rsuffix), indicator=True)
            explosion = x_merged[x_merged["_merge"] == "both"].groupby("_index").size()
            explosion = explosion[explosion > 1]
            for index in explosion.index:
                name = x.loc[index][["source", "target"]]
                new_filtered = x_merged[(x_merged["source"] == name["source"]) & (x_merged["target"] == name["target"])]
                old_filtered = x[(x["source"] == name["source"]) & (x["target"] == name["target"])]
                for i in range(1, len(new_filtered) - len(old_filtered) + 1):
                    index = new_filtered.iloc[i * -1].name
                    x_merged.at[index, "_merge"] = "right_only"
    else:
        x_merged = x.merge(y, how="outer", on=on, suffixes=(lsuffix, rsuffix), indicator=True)
        explosion = x_merged[x_merged["_merge"] == "both"].groupby("_index").size()
        explosion = explosion[explosion > 1]
        for index in explosion.index:
            name = x.loc[index]["name"]
            new_filtered = x_merged[x_merged["name"] == name]
            old_filtered = x[x["name"] == name]
            for i in range(1, len(new_filtered) - len(old_filtered) + 1):
                index = new_filtered.iloc[i * -1].name
                x_merged.at[index, "_merge"] = "right_only"

    x_merged.dropna(axis=1, how="all", inplace=True)
    new = x_merged["_merge"] == "right_only"
    new_rows = x_merged[new]
    x_merged = x_merged[~new]
    x_merged.sort_index(inplace=True)
    x_merged = pd.concat([x_merged, new_rows])
    x_merged.drop(columns=[col for col in x_merged if col.startswith("_")], inplace=True)
    if active == ActiveType.NODES:
        g.add_vertices(len(new_rows))
    elif active == ActiveType.EDGES:
        new_edges = new_rows[["source", "target"]].to_numpy()
        g.add_edges(new_edges)

    _apply_attributes(active, g, x_merged)


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
    x = g.get_edge_dataframe() if active == ActiveType.EDGES else g.get_vertex_dataframe()
    x["_index"] = x.index.to_series()
    y = y.copy()

    if active == ActiveType.EDGES:
        nodes_df = g.get_vertex_dataframe()
        id_map = nodes_df.set_index("name")
        name_to_index = pd.Series(data=nodes_df.index.to_numpy(), index=id_map.index)
        y["source"] = y["from"].map(name_to_index)
        y["target"] = y["to"].map(name_to_index)
        y.drop(columns=["from", "to"], inplace=True)
        on = ["source", "target"]
        if not g.is_directed():
            y_mirror = y.rename(
                columns={
                    "source": "target",
                    "target": "source",
                }
            )
            y = pd.concat([y, y_mirror])

    x_merged = x.merge(y, how="inner", on=on, suffixes=(lsuffix, rsuffix), indicator=True)
    explosion = x_merged[x_merged["_merge"] == "both"].groupby("_index").size()
    explosion = explosion[explosion > 1]

    if active == ActiveType.EDGES:
        for index in explosion.index:
            name = x.loc[index][["source", "target"]]
            new_filtered = x_merged[(x_merged["source"] == name["source"]) & (x_merged["target"] == name["target"])]
            old_filtered = x[(x["source"] == name["source"]) & (x["target"] == name["target"])]
            for i in range(1, len(new_filtered) - len(old_filtered) + 1):
                index = new_filtered.iloc[i * -1].name
                x_merged.at[index, "_merge"] = "right_only"
    else:
        for index in explosion.index:
            name = x.loc[index]["name"]
            new_filtered = x_merged[x_merged["name"] == name]
            old_filtered = x[x["name"] == name]
            for i in range(1, len(new_filtered) - len(old_filtered) + 1):
                index = new_filtered.iloc[i * -1].name
                x_merged.at[index, "_merge"] = "right_only"

    to_remove = x.merge(y, how="left_anti", on=on, suffixes=(lsuffix, rsuffix), indicator=True)
    to_remove = to_remove[to_remove["_merge"] == "left_only"]
    to_remove.set_index("_index", inplace=True)

    x_merged.dropna(axis=1, how="all", inplace=True)
    new = x_merged["_merge"] == "right_only" if not x_merged.empty else pd.Series([False] * len(x_merged))
    new_rows = x_merged[new]
    x_merged = x_merged[~new]
    x_merged = pd.concat([x_merged, new_rows])
    x_merged.drop(columns=[col for col in x_merged if col.startswith("_")], inplace=True)
    if active == ActiveType.NODES:
        if not to_remove.empty:
            g.delete_vertices(to_remove.index.to_numpy())
        if not new_rows.empty:
            g.add_vertices(len(new_rows))
    elif active == ActiveType.EDGES:
        if not to_remove.empty:
            source = to_remove["source"].to_numpy()
            target = to_remove["target"].to_numpy()
            # igraph strictly requires tuples
            edges = tuple(zip(source, target, strict=True))
            g.delete_edges(edges)
        if not new_rows.empty:
            new_edges = new_rows[["source", "target"]].to_numpy()
            g.add_edges(new_edges)

    _apply_attributes(active, g, x_merged)


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
    x = g.get_edge_dataframe() if active == ActiveType.EDGES else g.get_vertex_dataframe()
    x["_index"] = x.index.to_series()
    y = y.copy()

    if active == ActiveType.EDGES:
        nodes_df = g.get_vertex_dataframe()
        id_map = nodes_df.set_index("name")
        name_to_index = pd.Series(data=nodes_df.index.to_numpy(), index=id_map.index)
        y["source"] = y["from"].map(name_to_index)
        y["target"] = y["to"].map(name_to_index)
        y.drop(columns=["from", "to"], inplace=True)
        on = ["source", "target"]
        if not g.is_directed():
            y_mirror = y.rename(
                columns={
                    "source": "target",
                    "target": "source",
                }
            )
            y = pd.concat([y, y_mirror])

    x_merged = x.merge(y, how="left", on=on, suffixes=(lsuffix, rsuffix), indicator=True)
    explosion = x_merged[x_merged["_merge"] == "both"].groupby("_index").size()
    explosion = explosion[explosion > 1]

    if active == ActiveType.EDGES:
        for index in explosion.index:
            name = x.loc[index][["source", "target"]]
            new_filtered = x_merged[(x_merged["source"] == name["source"]) & (x_merged["target"] == name["target"])]
            old_filtered = x[(x["source"] == name["source"]) & (x["target"] == name["target"])]
            for i in range(1, len(new_filtered) - len(old_filtered) + 1):
                index = new_filtered.iloc[i * -1].name
                x_merged.at[index, "_merge"] = "right_only"
    else:
        for index in explosion.index:
            name = x.loc[index]["name"]
            new_filtered = x_merged[x_merged["name"] == name]
            old_filtered = x[x["name"] == name]
            for i in range(1, len(new_filtered) - len(old_filtered) + 1):
                index = new_filtered.iloc[i * -1].name
                x_merged.at[index, "_merge"] = "right_only"

    x_merged.dropna(axis=1, how="all", inplace=True)
    new = x_merged["_merge"] == "right_only" if not x_merged.empty else pd.Series([False] * len(x_merged))
    new_rows = x_merged[new]
    x_merged = x_merged[~new]
    x_merged = pd.concat([x_merged, new_rows])
    x_merged.drop(columns=[col for col in x_merged if col.startswith("_")], inplace=True)
    if active == ActiveType.NODES and not new_rows.empty:
        g.add_vertices(len(new_rows))
    elif active == ActiveType.EDGES and not new_rows.empty:
        new_edges = new_rows[["source", "target"]].to_numpy()
        g.add_edges(new_edges)

    _apply_attributes(active, g, x_merged)


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
    x = g.get_edge_dataframe() if active == ActiveType.EDGES else g.get_vertex_dataframe()
    x["_index"] = x.index.to_series()
    y = y.copy()

    nodes_df = g.get_vertex_dataframe()
    id_map = nodes_df.set_index("name")
    name_to_index = pd.Series(data=nodes_df.index.to_numpy(), index=id_map.index)

    if active == ActiveType.EDGES:
        y["source"] = y["from"].map(name_to_index)
        y["target"] = y["to"].map(name_to_index)
        # sort y by existing indices; we need this since the main driver table is y (which can be unsorted)
        y["_sort_index"] = [_get_edge_id(g, source, target) for source, target in y[["source", "target"]].to_numpy()]
        y = y.sort_values("_sort_index").drop(columns=["_sort_index"])
        if not y[["source", "target"]].notna().all().all():
            raise TidygraphValueError("Cannot perform edge join on non-existing nodes in the graph")
        y.drop(columns=["from", "to"], inplace=True)
        on = ["source", "target"]
        if not g.is_directed():
            x_mirror = x.rename(
                columns={
                    "source": "target",
                    "target": "source",
                }
            )
            x = pd.concat([x, x_mirror])
    else:
        y["_sort_index"] = y["name"].map(name_to_index)
        y = y.sort_values("_sort_index").drop(columns=["_sort_index"])

    x_merged = x.merge(y, how="right", on=on, suffixes=(lsuffix, rsuffix), indicator=True)
    to_remove = x.merge(y, how="left_anti", on=on, suffixes=(lsuffix, rsuffix), indicator=True)
    explosion = x_merged[x_merged["_merge"] == "both"].groupby("_index").size()
    explosion = explosion[explosion > 1]

    if active == ActiveType.EDGES:
        for index in explosion.index:
            name = y.loc[index][["source", "target"]]
            new_filtered = x_merged[(x_merged["source"] == name["source"]) & (x_merged["target"] == name["target"])]
            old_filtered = x[(x["source"] == name["source"]) & (x["target"] == name["target"])]
            for i in range(1, len(new_filtered) - len(old_filtered) + 1):
                index = new_filtered.iloc[i * -1].name
                x_merged.at[index, "_merge"] = "right_only"

        if not g.is_directed():
            y_mirror = y.rename(columns={"source": "target", "target": "source"})
            to_remove = to_remove.merge(y_mirror[["source", "target"]], on=on, how="left_anti")
    else:
        for index in explosion.index:
            name = y.loc[index]["name"]
            new_filtered = x_merged[x_merged["name"] == name]
            old_filtered = x[x["name"] == name]
            for i in range(1, len(new_filtered) - len(old_filtered) + 1):
                index = new_filtered.iloc[i * -1].name
                x_merged.at[index, "_merge"] = "right_only"

    to_remove = to_remove[to_remove["_merge"] == "left_only"]
    to_remove.set_index("_index", inplace=True)

    x_merged.dropna(axis=1, how="all", inplace=True)
    new = x_merged["_merge"] == "right_only" if not x_merged.empty else pd.Series([False] * len(x_merged))
    new_rows = x_merged[new]
    x_merged = x_merged[~new]
    x_merged = pd.concat([x_merged, new_rows])
    x_merged.drop(columns=[col for col in x_merged if col.startswith("_")], inplace=True)
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

    _apply_attributes(active, g, x_merged)


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

        target[col] = data[col].to_numpy()


def _get_edge_id(g: ig.Graph, source: int, target: int) -> int | None:
    """Internal wrapper around `igraph.get_eid`.

    This should be used when you do not want to crash the program when the edge
    does not exist. Instead, return None.
    """
    try:
        return g.get_eid(source, target)
    except Exception:
        return None
