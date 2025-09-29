from typing import Protocol


class NamedResource(Protocol):
    """Protocol for resources that have a name attribute."""

    name: str
