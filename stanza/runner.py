import logging
from collections.abc import Callable
from typing import Any

from stanza.logger.data_logger import DataLogger
from stanza.models import DeviceConfig
from stanza.registry import ResourceRegistry, ResultsRegistry
from stanza.utils import device_from_config

logger = logging.getLogger(__name__)

# Global registry of routines
_routine_registry: dict[str, Callable[..., Any]] = {}


class RoutineContext:
    """Context object passed to routine functions containing resources and results."""

    def __init__(self, resources: ResourceRegistry, results: ResultsRegistry) -> None:
        """Initialize context with resource and results registries."""
        self.resources = resources
        self.results = results


def routine(
    func: Callable[..., Any] | None = None, *, name: str | None = None
) -> Callable[..., Any]:
    """Decorator to register a function as a routine.

    The decorated function receives:
    - ctx: RoutineContext with ctx.resources and ctx.results
    - **params: Merged config and user parameters

    Usage:
        @routine
        def my_sweep(ctx, gate, voltages, measure_contact):
            device = ctx.resources.device
            return device.sweep_1d(gate, voltages, measure_contact)

        @routine(name="custom_name")
        def analyze_sweep(ctx, **params):
            # Access previous sweep data
            sweep_data = ctx.results.get("my_sweep")
            if sweep_data:
                voltages, currents = sweep_data
                # Do analysis...
            return analysis_result
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        routine_name = name or f.__name__
        _routine_registry[routine_name] = f
        logger.debug(f"Registered routine: {routine_name}")
        return f

    if func is None:
        # Called with arguments: @routine(name="custom_name")
        return decorator
    else:
        # Called without arguments: @routine
        return decorator(func)


def get_registered_routines() -> dict[str, Callable[..., Any]]:
    """Get all registered routines."""
    return _routine_registry.copy()


def clear_routine_registry() -> None:
    """Clear all registred routines"""
    _routine_registry.clear()


class RoutineRunner:
    """Simple runner that executes decorated routine functions with resources and configs.

    Resources can be provided in two ways:
    1. Pass initialized resources directly via `resources` parameter
    2. Pass configuration objects via `configs` parameter (runner instantiates resources)

    When using the `configs` parameter, a DataLogger is automatically created and registered
    with name="logger" for convenient logging within routines.

    Examples:
        # With initialized resources
        >>> device = Device(name="device", ...)
        >>> logger = DataLogger(name="logger", ...)
        >>> runner = RoutineRunner(resources=[device, logger])

        # With configs (runner creates device + logger automatically)
        >>> device_config = DeviceConfig(...)
        >>> runner = RoutineRunner(configs=[device_config])
        >>> # Now ctx.resources.device and ctx.resources.logger are available
    """

    def __init__(
        self,
        resources: list[Any] | None = None,
        configs: list[Any] | None = None,
    ):
        """Initialize runner with resources or configs.

        Args:
            resources: List of resource objects with .name attribute (Device, DataLogger, etc.)
            configs: List of configuration objects (DeviceConfig, etc.) to instantiate resources from

        Raises:
            ValueError: If neither resources nor configs provided, or if both provided
        """
        if resources is None and configs is None:
            raise ValueError("Must provide either 'resources' or 'configs'")

        if resources is not None and configs is not None:
            raise ValueError("Cannot provide both 'resources' and 'configs'")

        if resources is not None:
            self.resources = ResourceRegistry(*resources)
            self.configs: dict[str, dict[str, Any]] = {}
            self._device_configs: list[DeviceConfig] = []

        else:
            assert configs is not None
            instantiated_resources = self._build_resources_from_configs(configs)
            self.resources = ResourceRegistry(*instantiated_resources)
            self.configs = self._extract_routine_configs(configs)
            self._device_configs = [c for c in configs if isinstance(c, DeviceConfig)]

        self.results = ResultsRegistry()
        self.context = RoutineContext(self.resources, self.results)

        logger.info(
            f"Initialized RoutineRunner with {len(self.resources.list_resources())} resources"
        )

    def _build_resources_from_configs(
        self, configs: list[Any]
    ) -> list[DataLogger | Any]:
        """Instantiate resources from configuration objects.

        Args:
            configs: List of configuration objects (e.g., DeviceConfig)

        Returns:
            List of instantiated resource objects
        """

        resources: list[Any] = []

        for config in configs:
            if isinstance(config, DeviceConfig):
                device = device_from_config(config)
                resources.append(device)

                data_logger = DataLogger(
                    name="logger",
                    routine_name=device.name,
                    base_dir="./data",
                )
                resources.append(data_logger)

        return resources

    def _extract_routine_configs(self, configs: list[Any]) -> dict[str, dict[str, Any]]:
        """Extract routine parameters from configuration objects (recursive).

        Args:
            configs: List of configuration objects (e.g., DeviceConfig)

        Returns:
            Dictionary mapping routine names to their parameters
        """
        routine_configs: dict[str, dict[str, Any]] = {}

        for config in configs:
            if isinstance(config, DeviceConfig):
                for routine_config in config.routines:
                    self._extract_from_routine_config(routine_config, routine_configs)

        return routine_configs

    def _extract_from_routine_config(
        self, routine_config: Any, routine_configs: dict[str, dict[str, Any]]
    ) -> None:
        """Recursively extract parameters from routine config and its nested routines.

        Args:
            routine_config: The routine configuration to extract from
            routine_configs: Dictionary to store extracted parameters
        """
        if routine_config.parameters:
            routine_configs[routine_config.name] = routine_config.parameters

        if routine_config.routines:
            for nested_routine in routine_config.routines:
                self._extract_from_routine_config(nested_routine, routine_configs)

    def run(self, routine_name: str, **params: Any) -> Any:
        """Execute a registered routine.

        Args:
            routine_name: Name of the routine to run
            **params: Additional parameters (will override config values)

        Returns:
            Result of the routine
        """
        if routine_name not in _routine_registry:
            available = list(_routine_registry.keys())
            raise ValueError(
                f"Routine '{routine_name}' not registered. Available: {available}"
            )

        # Get config for this routine and merge with user params
        config = self.configs.get(routine_name, {})
        merged_params = {**config, **params}

        # Get the routine function from global registry
        routine_func = _routine_registry[routine_name]

        # Create logger session if logger exists and has create_session method
        data_logger = getattr(self.resources, "logger", None)
        session = None
        if data_logger is not None and hasattr(data_logger, "create_session"):
            session = data_logger.create_session(session_id=routine_name)
            merged_params["session"] = session

        try:
            logger.info(f"Running routine: {routine_name}")
            result = routine_func(self.context, **merged_params)

            # Store result
            self.results.store(routine_name, result)
            logger.info(f"Completed routine: {routine_name}")

            return result

        except Exception as e:
            logger.error(f"Routine {routine_name} failed: {e}")
            raise RuntimeError(f"Routine '{routine_name}' failed: {e}") from e

        finally:
            # Close logger session if it was created
            if session is not None and data_logger is not None:
                data_logger.close_session(session_id=routine_name)

    def run_all(self, parent_routine: str | None = None) -> dict[str, Any]:
        """Execute all routines from config in order.

        Args:
            parent_routine: If specified, run only nested routines under this parent.
                          If None, run all top-level routines.

        Returns:
            Dictionary mapping routine names to their results
        """
        results: dict[str, Any] = {}

        for device_config in self._device_configs:
            for routine_config in device_config.routines:
                if parent_routine is None:
                    self._run_routine_tree(routine_config, results)
                elif routine_config.name == parent_routine and routine_config.routines:
                    for nested_routine in routine_config.routines:
                        self._run_routine_tree(nested_routine, results)

        return results

    def _run_routine_tree(self, routine_config: Any, results: dict[str, Any]) -> None:
        """Recursively run a routine and its nested routines.

        Args:
            routine_config: The routine configuration to execute
            results: Dictionary to store results
        """
        if routine_config.name in _routine_registry:
            result = self.run(routine_config.name)
            results[routine_config.name] = result

        if routine_config.routines:
            for nested_routine in routine_config.routines:
                self._run_routine_tree(nested_routine, results)

    def get_result(self, routine_name: str) -> Any:
        """Get stored result from a routine."""
        return self.results.get(routine_name)

    def list_routines(self) -> list[str]:
        """List all registered routines."""
        return list(_routine_registry.keys())

    def list_results(self) -> list[str]:
        """List all stored results."""
        return self.results.list_results()
