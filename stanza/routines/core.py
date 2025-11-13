import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stanza.context import StanzaSession
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
        logger.debug("Registered routine: %s", routine_name)
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
        base_dir: str | Path | None = None,
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

        self._routine_hierarchy: dict[str, str] = {}
        # _base_dir_override stores an explicit runner-level override (from ctor).
        # When None, the DataLogger should follow the current session/override from
        # StanzaSession. When set, we always use the provided directory.
        self._base_dir_override: Path | None = Path(base_dir) if base_dir else None
        # _manages_logger_base_dir tracks whether the runner created the DataLogger and
        # therefore owns its directory switching. If False (user provided their own logger),
        # we never mutate its base directory unless a base_dir override was supplied.
        self._manages_logger_base_dir = False

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
            "Initialized RoutineRunner with %s resources",
            len(self.resources.list_resources()),
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
                device.name = "device"

                data_logger = DataLogger(
                    name="logger",
                    routine_name=device.name,
                    base_dir=self._resolve_base_dir(),
                )
                resources.append(data_logger)
                # Runner instantiated the logger, so it is safe to retarget it when
                # session overrides change.
                self._manages_logger_base_dir = True

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
        self,
        routine_config: Any,
        routine_configs: dict[str, dict[str, Any]],
        parent_path: str = "",
    ) -> None:
        """Recursively extract parameters from routine config and its nested routines.

        Args:
            routine_config: The routine configuration to extract from
            routine_configs: Dictionary to store extracted parameters
        """
        # Store parameters and group information
        config_data: dict[str, Any] = {}
        if routine_config.parameters:
            config_data.update(routine_config.parameters)
        if hasattr(routine_config, "group") and routine_config.group is not None:
            config_data["group"] = routine_config.group

        if config_data:
            routine_configs[routine_config.name] = config_data

        if routine_config.routines:
            current_path = (
                f"{parent_path}/{routine_config.name}".lower()
                if parent_path
                else routine_config.name.lower()
            )
            for nested_routine in routine_config.routines:
                self._routine_hierarchy[nested_routine.name] = (
                    f"{current_path}/{nested_routine.name}"
                )
                self._extract_from_routine_config(
                    nested_routine, routine_configs, current_path
                )

    def _get_routine_path(self, routine_name: str) -> str:
        """Get the full path for a routine including parent hierarhcy.

        Args:
            routine_name: Name of the routine

        Returns:
            Full path like "health_check/noise_floor_measurement" or just "routine_name" if no parent
        """
        return self._routine_hierarchy.get(routine_name, routine_name)

    def _get_parent_params(self, routine_name: str) -> dict[str, Any]:
        """Get merged parameters from all parent routines in the hierarchy.

        Args:
            routine_name: Name of the routine

        Returns:
            Dictionary of merged parent parameters
        """
        full_path = self._get_routine_path(routine_name)

        if "/" not in full_path:
            return {}

        path_substr = full_path.split("/")
        parent_names = path_substr[:-1]

        merged_params: dict[str, Any] = {}
        for parent_name in parent_names:
            parent_params = self.configs.get(parent_name, {})
            merged_params.update(parent_params)

        return merged_params

    def run(self, routine_name: str, **params: Any) -> Any:
        """Execute a registered routine.

        Args:
            routine_name: Name of the routine to run
            **params: Additional parameters (will override config values), including:
                - session_suffix: Optional suffix to append to session_id for unique save paths
                                  (e.g., "OUT_A", "OUT_B"). Creates paths like "{routine_name}_{suffix}/"
                - group: Filter device to specific gate group (does NOT affect save path)
                - Other routine-specific parameters

        Returns:
            Result of the routine
        """
        if routine_name not in _routine_registry:
            available = list(_routine_registry.keys())
            raise ValueError(
                f"Routine '{routine_name}' not registered. Available: {available}"
            )

        # Get config for this routine and merge with parent and user params
        parent_params = self._get_parent_params(routine_name)
        config = self.configs.get(routine_name, {})
        merged_params = {**parent_params, **config, **params}

        # Extract session_suffix if present (not passed to routine as parameter)
        session_suffix = merged_params.pop("session_suffix", None)

        # Extract group information if present (not passed to routine as parameter)
        group_name = merged_params.pop("group", None)

        # Get the routine function from global registry
        routine_func = _routine_registry[routine_name]

        # Filter device by group if specified
        if group_name is not None:
            device = getattr(self.resources, "device", None)
            if device is not None:
                group = device.filter_by_group(group_name)
                # Add group configs to resources for routine access
                self.resources.add("group", group)
                # Only one group can be available at a time
                logger.info("Filtering device to group: %s", group_name)

        # Create logger session if logger exists and has create_session method
        data_logger = getattr(self.resources, "logger", None)
        session = None
        if data_logger is not None and hasattr(data_logger, "create_session"):
            self._update_logger_base_dir_if_needed()
            session_id = self._get_routine_path(routine_name)
            # Append session_suffix if provided (for unique save paths)
            if session_suffix is not None:
                session_id = f"{session_id}_{session_suffix}"
            session = data_logger.create_session(
                session_id=session_id,
                group_name=group_name,  # stored in metadata, not path
            )
            merged_params["session"] = session

        try:
            logger.info("Running routine: %s", routine_name)
            result = routine_func(self.context, **merged_params)

            # Store result with group suffix if group is specified
            result_key = (
                f"{routine_name}_{group_name}"
                if group_name is not None
                else routine_name
            )
            self.results.store(result_key, result)
            logger.info("Completed routine: %s", routine_name)

            return result

        except Exception as e:
            logger.error("Routine %s failed: %s", routine_name, e)
            raise RuntimeError(f"Routine '{routine_name}' failed: {e}") from e

        finally:
            # Close logger session if it was created
            if session is not None and data_logger is not None:
                data_logger.close_session(session_id=session.session_id)

    def _resolve_base_dir(self) -> Path:
        """Resolve logger base directory in priority order.

        Priority:
            1. Runner-level override (`base_dir` ctor arg)
            2. Session override/active session
            3. Local ./data fallback when nothing else is configured
        """
        if self._base_dir_override is not None:
            return self._base_dir_override

        session_dir = StanzaSession.get_active_session()
        if session_dir is not None:
            return session_dir

        return Path("./data")

    def _update_logger_base_dir_if_needed(self) -> None:
        """Synchronize the logger base directory if we own it.

        We only mutate the logger path when:
            * the runner created the logger (so `_manages_logger_base_dir` is True), or
            * the caller provided an explicit `base_dir` override.

        This prevents clobbering user-provided loggers while still keeping runner-created
        loggers aligned with session overrides.
        """
        if not (self._manages_logger_base_dir or self._base_dir_override is not None):
            return

        data_logger = getattr(self.resources, "logger", None)
        if data_logger is None or not hasattr(data_logger, "set_base_directory"):
            return

        data_logger.set_base_directory(self._resolve_base_dir())

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
                    parent_params = routine_config.parameters or {}
                    for nested_routine in routine_config.routines:
                        self._run_routine_tree(nested_routine, results, parent_params)

        return results

    def _run_routine_tree(
        self,
        routine_config: Any,
        results: dict[str, Any],
        parent_params: dict[str, Any] | None = None,
    ) -> None:
        """Recursively run a routine and its nested routines.

        Args:
            routine_config: The routine configuration to execute
            results: Dictionary to store results
            parent_params: Parameters from parent routine to inherit
        """
        if routine_config.name in _routine_registry:
            # Extract group from routine_config if present
            group_override = {}
            if hasattr(routine_config, "group") and routine_config.group is not None:
                group_override["group"] = routine_config.group

            if parent_params:
                merged = {
                    **parent_params,
                    **(routine_config.parameters or {}),
                    **group_override,
                }
                result = self.run(routine_config.name, **merged)
            else:
                # Merge config parameters with group override
                merged = {**(routine_config.parameters or {}), **group_override}
                if merged:
                    result = self.run(routine_config.name, **merged)
                else:
                    result = self.run(routine_config.name)

            # Store result with unique key if there's a group
            result_key = (
                f"{routine_config.name}_{routine_config.group}"
                if hasattr(routine_config, "group") and routine_config.group
                else routine_config.name
            )
            results[result_key] = result

        if routine_config.routines:
            current_params = routine_config.parameters or {}
            child_params = (
                {**parent_params, **current_params} if parent_params else current_params
            )
            for nested_routine in routine_config.routines:
                self._run_routine_tree(nested_routine, results, child_params)

    def get_result(self, routine_name: str) -> Any:
        """Get stored result from a routine."""
        return self.results.get(routine_name)

    def list_routines(self) -> list[str]:
        """List all registered routines."""
        return list(_routine_registry.keys())

    def list_results(self) -> list[str]:
        """List all stored results."""
        return self.results.list_results()
