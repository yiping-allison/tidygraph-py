"""Determines the context of subsequent manipulations."""

from enum import Enum

__all__ = ["ActiveType"]


class ActiveType(Enum):
    """An enumeration of the possible active dataframes in a Tidygraph object."""

    NODES = "nodes"
    EDGES = "edges"


class ActiveState:
    """Represents the internals of what is currently active in a Tidygraph object."""

    _active: ActiveType = ActiveType.NODES

    @property
    def active(self) -> ActiveType:
        """The currently activated dataframe."""
        return self._active

    @active.setter
    def active(self, value: ActiveType) -> None:
        self._active = value
