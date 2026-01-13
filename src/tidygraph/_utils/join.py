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
        id_map = g.get_vertex_dataframe().set_index("name")
        y["source"] = y["from"].map(lambda x: id_map.index.get_loc(x))
        y["target"] = y["to"].map(lambda x: id_map.index.get_loc(x))
        on = ["source", "target"]
        if not g.is_directed():
            # undirected graphs need to consider "mirrored" edges during joins. We do not want to keep
            # attributes for both (a->b) and (b->a) edges separately, so we handle this by:
            # 1. augmenting y with mirrored edges
            # 2. performing a left join between x_tmp and augmented y \
            #   (e.g., adding updated attributes to existing edges)
            # 3. creating mirrored edges
            # 4. performing a right-anti join to find edges in y that are not in x_tmp (in either direction)
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
