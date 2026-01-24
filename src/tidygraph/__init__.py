"""Tidygraph is a package providing a tidy-like experience when working with graphs in Python.

It is heavily inspired by the tidygraph package in R.
"""

from tidygraph import activate, exceptions
from tidygraph.tidygraph import Tidygraph

__all__ = ["Tidygraph", "activate", "exceptions"]
