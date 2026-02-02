from typing import NamedTuple, final


@final
class ReservedGraphKeywords(NamedTuple):
    """Reserved keywords based on internal graph manipulation needs."""

    NODES = frozenset(["vertex ID"])
    EDGES = frozenset(["node ID", "from", "to", "source", "target"])


RESERVED_JOIN_KEYWORD: str = "_index"
