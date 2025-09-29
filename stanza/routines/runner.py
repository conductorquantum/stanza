import logging
from collections.abc import Callable
from typing import Any

from stanza.device import Device
from stanza.models import RoutineConfig
from stanza.routines.datatypes import ResourceRegistry, ResultsRegistry, RoutineContext

logger = logging.getLogger(__name__)

# Global registry of routines
_routine_registry: dict[str, Callable[..., Any]] = {}


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
    """Simple runner that executes decorated routine functions with routine configs and result access."""

    def __init__(self, device: Device, routine_configs: list[RoutineConfig]):
        """Initialize runner with device and routine configurations.

        Args:
            device: The quantum device to run experiments on
            routine_configs: List of RoutineConfig objects from DeviceConfig
        """
        if not device.is_configured():
            raise ValueError(
                "Device must be configured with both control and measurement instruments"
            )

        # Set up resource and results registries
        self.resources = ResourceRegistry(device)
        self.results = ResultsRegistry()
        self.context = RoutineContext(self.resources, self.results)

        # Extract routine configs
        self.configs: dict[str, dict[str, Any]] = {}
        for routine_config in routine_configs:
            if routine_config.parameters:
                self.configs[routine_config.name] = routine_config.parameters

        logger.info(f"Loaded {len(self.configs)} routine configs")

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

    def get_result(self, routine_name: str) -> Any:
        """Get stored result from a routine."""
        return self.results.get(routine_name)

    def list_routines(self) -> list[str]:
        """List all registered routines."""
        return list(_routine_registry.keys())

    def list_results(self) -> list[str]:
        """List all stored results."""
        return self.results.list_results()
