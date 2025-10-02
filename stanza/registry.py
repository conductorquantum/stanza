from typing import Any

from stanza.base.protocols import NamedResource


class ResourceRegistry:
    """Registry for named resources available to routines."""

    def __init__(self, *resources: NamedResource):
        """Initialize with resources that have a 'name' attribute."""
        self._resources = {resource.name: resource for resource in resources}

    def __getattr__(self, name: str) -> Any:
        """Access resources via attribute notation."""
        if name.startswith("_"):
            # Avoid recursion with private attributes
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        if name in self._resources:
            return self._resources[name]

        raise AttributeError(
            f"Resource '{name}' not found. Available: {list(self._resources.keys())}"
        )

    def __getitem__(self, name: str) -> Any:
        """Access resources via dict notation."""
        return self._resources[name]

    def get(self, name: str, default: Any = None) -> Any:
        """Get resource with optional default."""
        return self._resources.get(name, default)

    def add(self, name: str, resource: Any) -> None:
        """Add a new resource."""
        self._resources[name] = resource

    def list_resources(self) -> list[str]:
        """List all available resource names."""
        return list(self._resources.keys())


class ResultsRegistry:
    """Registry for storing and accessing routine results."""

    def __init__(self) -> None:
        """Initialize empty results registry."""
        self._results: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """Access results via attribute notation."""
        if name.startswith("_"):
            # Avoid recursion with private attributes
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        if name in self._results:
            return self._results[name]

        raise AttributeError(
            f"Result '{name}' not found. Available: {list(self._results.keys())}"
        )

    def __getitem__(self, name: str) -> Any:
        """Access results via dict notation."""
        return self._results[name]

    def __setitem__(self, name: str, value: Any) -> None:
        """Store results via dict notation."""
        self._results[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """Get result with optional default."""
        return self._results.get(name, default)

    def store(self, name: str, value: Any) -> None:
        """Store a result."""
        self._results[name] = value

    def list_results(self) -> list[str]:
        """List all stored result names."""
        return list(self._results.keys())

    def clear(self) -> None:
        """Clear all stored results."""
        self._results.clear()
