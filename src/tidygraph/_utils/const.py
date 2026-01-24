from typing import NamedTuple, final


@final
class ReservedKeywords(NamedTuple):
    """Reserved keywords based on internal graph manipulation needs."""

    NODES = frozenset(["vertex ID"])
    EDGES = frozenset(["node ID", "from", "to", "source", "target"])
