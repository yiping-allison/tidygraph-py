"""Custom and base exceptions for the tidygraph library."""

__all__ = [
    "TidygraphError",
    "TidygraphValueError",
]


class TidygraphError(Exception):
    """Base class for all Tidygraph errors."""

    pass


class TidygraphValueError(TidygraphError):
    """Exception raised for invalid values in tidygraph operations."""

    pass
