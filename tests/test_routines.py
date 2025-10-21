import pytest
import yaml

from stanza.logger.data_logger import DataLogger
from stanza.models import DeviceConfig
from stanza.registry import ResourceRegistry, ResultsRegistry
from stanza.routines import (
    RoutineContext,
    RoutineRunner,
    clear_routine_registry,
    get_registered_routines,
    routine,
)


class MockResource:
    def __init__(self, name: str):
        self.name = name
        self.data = f"data_from_{name}"


@pytest.fixture
def registry_fixture():
    clear_routine_registry()
    yield
    clear_routine_registry()


@pytest.fixture
def resource_registry():
    return ResourceRegistry(MockResource("resource1"), MockResource("resource2"))


@pytest.fixture
def results_registry():
    registry = ResultsRegistry()
    registry.store("result1", "value1")
    registry.store("result2", "value2")
    return registry


class TestResourceRegistry:
    def test_initialization_with_named_resources(self):
        """Test initializing ResourceRegistry with multiple named resources."""
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")
        registry = ResourceRegistry(resource1, resource2)

        assert registry.resource1 is resource1
        assert registry.resource2 is resource2

    def test_getattr_access(self):
        """Test accessing resources via attribute notation."""
        resource = MockResource("test_resource")
        registry = ResourceRegistry(resource)

        assert registry.test_resource is resource
        assert registry.test_resource.data == "data_from_test_resource"

    def test_getitem_access(self, resource_registry):
        """Test accessing resources via dictionary-style indexing."""
        assert resource_registry["resource1"].name == "resource1"

    def test_get_method(self, resource_registry):
        """Test get method with default value fallback."""
        assert resource_registry.get("resource1").name == "resource1"
        assert resource_registry.get("nonexistent", "default") == "default"

    def test_add_resource(self):
        """Test adding a resource to the registry dynamically."""
        registry = ResourceRegistry()
        resource = MockResource("new_resource")
        registry.add("new_resource", resource)

        assert registry.new_resource is resource

    def test_list_resources(self, resource_registry):
        """Test listing all registered resources."""
        resources = resource_registry.list_resources()
        assert "resource1" in resources
        assert "resource2" in resources
        assert len(resources) == 2

    def test_nonexistent_resource_raises_attribute_error(self):
        """Test that accessing nonexistent resource raises AttributeError."""
        registry = ResourceRegistry()
        with pytest.raises(AttributeError, match="Resource 'nonexistent' not found"):
            _ = registry.nonexistent

    def test_private_attribute_access(self):
        """Test that accessing private attributes raises AttributeError."""
        registry = ResourceRegistry()
        with pytest.raises(AttributeError):
            _ = registry._private_attr


class TestResultsRegistry:
    def test_initialization(self):
        """Test initializing an empty ResultsRegistry."""
        registry = ResultsRegistry()
        assert registry.list_results() == []

    def test_store_and_get(self):
        """Test storing and retrieving results from the registry."""
        registry = ResultsRegistry()
        registry.store("test_result", {"data": "test"})

        assert registry.get("test_result") == {"data": "test"}

    def test_getattr_access(self, results_registry):
        """Test accessing results via attribute notation."""
        assert results_registry.result1 == "value1"

    def test_getitem_setitem_access(self):
        """Test accessing and setting results via dictionary-style indexing."""
        registry = ResultsRegistry()
        registry["test_result"] = "test_value"

        assert registry["test_result"] == "test_value"

    def test_get_with_default(self):
        """Test get method with default value for nonexistent results."""
        registry = ResultsRegistry()
        assert registry.get("nonexistent", "default") == "default"

    def test_list_results(self, results_registry):
        """Test listing all stored results."""
        results = results_registry.list_results()
        assert "result1" in results
        assert "result2" in results
        assert len(results) == 2

    def test_clear(self):
        """Test clearing all results from the registry."""
        registry = ResultsRegistry()
        registry.store("result1", "value1")

        assert len(registry.list_results()) == 1
        registry.clear()
        assert len(registry.list_results()) == 0

    def test_nonexistent_result_raises_attribute_error(self):
        """Test that accessing nonexistent result raises AttributeError."""
        registry = ResultsRegistry()
        with pytest.raises(AttributeError, match="Result 'nonexistent' not found"):
            _ = registry.nonexistent


class TestRoutineContext:
    def test_initialization(self):
        """Test initializing RoutineContext with resources and results registries."""
        resources = ResourceRegistry()
        results = ResultsRegistry()
        context = RoutineContext(resources, results)

        assert context.resources is resources
        assert context.results is results


class TestRoutineDecorator:
    def test_routine_decorator_registers_function(self, registry_fixture):
        """Test that routine decorator registers function with its name."""

        @routine
        def test_routine(ctx):
            return "test_result"

        registered = get_registered_routines()
        assert "test_routine" in registered
        assert registered["test_routine"] == test_routine

    def test_routine_decorator_with_custom_name(self, registry_fixture):
        """Test that routine decorator can register function with custom name."""

        @routine(name="custom_name")
        def test_routine(ctx):
            return "test_result"

        registered = get_registered_routines()
        assert "custom_name" in registered
        assert "test_routine" not in registered

    def test_routine_decorator_returns_original_function(self, registry_fixture):
        """Test that routine decorator returns the original function unchanged."""

        def original_func(ctx):
            return "original"

        decorated = routine()(original_func)
        assert decorated is original_func

    def test_multiple_routine_registrations(self, registry_fixture):
        """Test registering multiple routines simultaneously."""

        @routine
        def routine1(ctx):
            pass

        @routine
        def routine2(ctx):
            pass

        registered = get_registered_routines()
        assert "routine1" in registered
        assert "routine2" in registered
        assert len(registered) == 2

    def test_clear_routine_registry(self, registry_fixture):
        """Test clearing all registered routines from the registry."""

        @routine
        def test_routine(ctx):
            pass

        assert len(get_registered_routines()) == 1
        clear_routine_registry()
        assert len(get_registered_routines()) == 0

    def test_get_registered_routines_returns_copy(self, registry_fixture):
        """Test that get_registered_routines returns a copy, not the original registry."""

        @routine
        def test_routine(ctx):
            pass

        registered1 = get_registered_routines()
        registered2 = get_registered_routines()

        assert registered1 == registered2
        assert registered1 is not registered2


class TestRoutineRunner:
    def test_initialization_with_resources(self, registry_fixture):
        """Test initializing RoutineRunner with resource list."""
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")

        runner = RoutineRunner(resources=[resource1, resource2])

        assert runner.resources.resource1 is resource1
        assert runner.resources.resource2 is resource2
        assert runner.results.list_results() == []
        assert runner.configs == {}

    def test_initialization_requires_resources_or_configs(self, registry_fixture):
        """Test that initialization requires either resources or configs parameter."""
        with pytest.raises(
            ValueError, match="Must provide either 'resources' or 'configs'"
        ):
            RoutineRunner()

    def test_initialization_cannot_provide_both(
        self, registry_fixture, simple_routines_yaml
    ):
        """Test that initialization cannot accept both resources and configs."""
        with pytest.raises(
            ValueError, match="Cannot provide both 'resources' and 'configs'"
        ):
            device = MockResource("device")
            device_config = DeviceConfig.model_validate(
                yaml.safe_load(simple_routines_yaml)
            )
            RoutineRunner(resources=[device], configs=[device_config])

    def test_run_routine_not_registered(self, registry_fixture):
        """Test that running unregistered routine raises ValueError."""
        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        with pytest.raises(ValueError, match="Routine 'nonexistent' not registered"):
            runner.run("nonexistent")

    def test_run_routine_basic(self, registry_fixture):
        """Test running a basic routine and storing its result."""

        @routine
        def test_routine(ctx):
            return "success"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])
        result = runner.run("test_routine")

        assert result == "success"
        assert runner.get_result("test_routine") == "success"

    def test_run_routine_with_context_access(self, registry_fixture):
        """Test that routines can access resources through context."""

        @routine
        def test_routine(ctx):
            resource = ctx.resources.test_resource
            return f"data_{resource.data}"

        resource = MockResource("test_resource")
        runner = RoutineRunner(resources=[resource])
        result = runner.run("test_routine")

        assert result == "data_data_from_test_resource"

    def test_run_routine_with_params(self, registry_fixture):
        """Test running routine with custom parameters."""

        @routine
        def test_routine(ctx, param1, param2="default"):
            return f"{param1}-{param2}"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])
        result = runner.run("test_routine", param1="hello", param2="world")

        assert result == "hello-world"

    def test_run_routine_with_config_params(self, registry_fixture):
        """Test running routine with parameters from configuration."""

        @routine
        def test_routine(ctx, param1, param2="default"):
            return f"{param1}-{param2}"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        runner.configs["test_routine"] = {"param1": "value1", "param2": 42}

        result = runner.run("test_routine")

        assert result == "value1-42"

    def test_run_routine_user_params_override_config(self, registry_fixture):
        """Test that user-provided parameters override config parameters."""

        @routine
        def test_routine(ctx, param1, param2="default"):
            return f"{param1}-{param2}"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        runner.configs["test_routine"] = {"param1": "value1", "param2": 42}

        result = runner.run("test_routine", param2="overridden")

        assert result == "value1-overridden"

    def test_run_routine_access_previous_results(self, registry_fixture):
        """Test that routines can access results from previously executed routines."""

        @routine
        def first_routine(ctx):
            return {"data": "first_result"}

        @routine
        def second_routine(ctx):
            first_data = ctx.results.get("first_routine")
            return f"processed_{first_data['data']}"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        runner.run("first_routine")
        result = runner.run("second_routine")

        assert result == "processed_first_result"

    def test_run_routine_exception_handling(self, registry_fixture):
        """Test that routine exceptions are wrapped in RuntimeError with context."""

        @routine
        def failing_routine(ctx):
            raise ValueError("Something went wrong")

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        with pytest.raises(RuntimeError, match="Routine 'failing_routine' failed"):
            runner.run("failing_routine")

    def test_get_result(self, registry_fixture):
        """Test retrieving stored routine results."""

        @routine
        def test_routine(ctx):
            return "test_result"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        runner.run("test_routine")

        assert runner.get_result("test_routine") == "test_result"
        assert runner.get_result("nonexistent") is None

    def test_list_routines(self, registry_fixture):
        """Test listing all registered routines."""

        @routine
        def routine1(ctx):
            pass

        @routine
        def routine2(ctx):
            pass

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])
        routines = runner.list_routines()

        assert "routine1" in routines
        assert "routine2" in routines
        assert len(routines) == 2

    def test_list_results(self, registry_fixture):
        """Test listing all routine execution results."""

        @routine
        def routine1(ctx):
            return "result1"

        @routine
        def routine2(ctx):
            return "result2"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        runner.run("routine1")
        runner.run("routine2")

        results = runner.list_results()
        assert "routine1" in results
        assert "routine2" in results
        assert len(results) == 2

    def test_sequential_routines_building_on_results(self, registry_fixture):
        """Test sequential routines that build on each other's results."""

        @routine
        def collect_data(ctx, data_size=10):
            return list(range(data_size))

        @routine
        def process_data(ctx, multiplier=2):
            raw_data = ctx.results.get("collect_data", [])
            return [x * multiplier for x in raw_data]

        @routine
        def analyze_data(ctx):
            processed_data = ctx.results.get("process_data", [])
            return {"sum": sum(processed_data), "count": len(processed_data)}

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        runner.configs["collect_data"] = {"data_size": 5}
        runner.configs["process_data"] = {"multiplier": 3}

        collect_result = runner.run("collect_data")
        process_result = runner.run("process_data")
        analyze_result = runner.run("analyze_data")

        assert collect_result == [0, 1, 2, 3, 4]
        assert process_result == [0, 3, 6, 9, 12]
        assert analyze_result == {"sum": 30, "count": 5}

    def test_multiple_resources_in_runner(self, registry_fixture):
        """Test accessing multiple resources within a routine."""

        @routine
        def use_multiple_resources(ctx):
            r1 = ctx.resources.resource1
            r2 = ctx.resources.resource2
            return f"{r1.data}+{r2.data}"

        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")
        runner = RoutineRunner(resources=[resource1, resource2])

        result = runner.run("use_multiple_resources")

        assert result == "data_from_resource1+data_from_resource2"

    def test_nested_routine_config_extraction(
        self, registry_fixture, nested_routines_yaml
    ):
        """Test extracting configuration from nested routine hierarchy."""
        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        runner = RoutineRunner(
            resources=[MockResource("device"), MockResource("logger")]
        )
        extracted_configs = runner._extract_routine_configs([device_config])

        assert extracted_configs["health_check"] == {
            "parent_param": "value",
            "charge_carrier_type": "holes",
        }
        assert extracted_configs["leakage_test"] == {
            "leakage_threshold_resistance": 50000000,
            "leakage_threshold_count": 0,
        }
        assert extracted_configs["global_accumulation"] == {"step_size": 0.01}
        assert extracted_configs["reservoir_characterization"] == {"step_size": 0.01}
        assert len(extracted_configs) == 4

    def test_deeply_nested_routine_config_extraction(
        self, registry_fixture, deeply_nested_routines_yaml
    ):
        """Test extracting configuration from deeply nested routine structure."""
        device_config = DeviceConfig.model_validate(
            yaml.safe_load(deeply_nested_routines_yaml)
        )
        runner = RoutineRunner(
            resources=[MockResource("device"), MockResource("logger")]
        )
        extracted_configs = runner._extract_routine_configs([device_config])

        assert extracted_configs["level1"] == {"level": 1}
        assert extracted_configs["level2"] == {"level": 2}
        assert extracted_configs["level3"] == {"level": 3}
        assert len(extracted_configs) == 3

    def test_run_all_nested_routines(self, registry_fixture, nested_routines_yaml):
        """Test running all subroutines within a parent routine."""
        execution_order = []

        @routine
        def leakage_test(ctx, **params):
            execution_order.append("leakage_test")
            return f"leakage_{params['leakage_threshold_resistance']}"

        @routine
        def global_accumulation(ctx, **params):
            execution_order.append("global_accumulation")
            return f"accumulation_{params['step_size']}"

        @routine
        def reservoir_characterization(ctx, **params):
            execution_order.append("reservoir_characterization")
            return f"reservoir_{params['step_size']}"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        device = MockResource("device")
        logger = MockResource("logger")
        runner = RoutineRunner(resources=[device, logger])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]
        results = runner.run_all(parent_routine="health_check")

        assert results["leakage_test"] == "leakage_50000000.0"
        assert results["global_accumulation"] == "accumulation_0.01"
        assert results["reservoir_characterization"] == "reservoir_0.01"
        assert len(results) == 3
        assert execution_order == [
            "leakage_test",
            "global_accumulation",
            "reservoir_characterization",
        ]

    def test_run_all_top_level_routines(self, registry_fixture, simple_routines_yaml):
        """Test running all top-level routines when no parent specified."""
        execution_order = []

        @routine
        def routine1(ctx, param1):
            execution_order.append("routine1")
            return f"result1_{param1}"

        @routine
        def routine2(ctx, param2):
            execution_order.append("routine2")
            return f"result2_{param2}"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(simple_routines_yaml)
        )
        device = MockResource("device")
        logger = MockResource("logger")
        runner = RoutineRunner(resources=[device, logger])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]
        results = runner.run_all()

        assert results["routine1"] == "result1_value1"
        assert results["routine2"] == "result2_value2"
        assert len(results) == 2
        assert execution_order == ["routine1", "routine2"]


class TestRoutineRunnerLoggerIntegration:
    def test_run_routine_creates_and_passes_session(self, registry_fixture, tmp_path):
        """Test that running a routine creates and passes a logging session."""

        @routine
        def test_routine(ctx, session=None):
            assert session is not None
            assert session.session_id == "test_routine"
            assert ctx.resources.logger.current_session is session
            return "success"

        resource = MockResource("resource")
        logger = DataLogger(
            name="logger", routine_name="test", base_dir=tmp_path / "data"
        )
        runner = RoutineRunner(resources=[resource, logger])

        result = runner.run("test_routine")

        assert result == "success"
        assert logger.current_session is None
        assert "test_routine" not in logger._active_sessions

    def test_run_routine_closes_session_on_failure(self, registry_fixture, tmp_path):
        """Test that logging session is properly closed even when routine fails."""

        @routine
        def failing_routine(ctx, session=None):
            assert session is not None
            raise ValueError("Test error")

        resource = MockResource("resource")
        logger = DataLogger(
            name="logger", routine_name="test", base_dir=tmp_path / "data"
        )
        runner = RoutineRunner(resources=[resource, logger])

        with pytest.raises(RuntimeError, match="Routine 'failing_routine' failed"):
            runner.run("failing_routine")

        assert logger.current_session is None
        assert "failing_routine" not in logger._active_sessions

    def test_run_routine_without_logger(self, registry_fixture):
        """Test that routines run normally when no logger is available."""

        @routine
        def test_routine(ctx, session=None):
            assert session is None
            return "success"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        result = runner.run("test_routine")

        assert result == "success"

    def test_multiple_routines_get_separate_sessions(self, registry_fixture, tmp_path):
        """Test that each routine execution gets its own separate logging session."""
        session_ids_seen = []

        @routine
        def first_routine(ctx, session=None):
            session_ids_seen.append(session.session_id)
            return "first"

        @routine
        def second_routine(ctx, session=None):
            session_ids_seen.append(session.session_id)
            return "second"

        resource = MockResource("resource")
        logger = DataLogger(
            name="logger", routine_name="test", base_dir=tmp_path / "data"
        )
        runner = RoutineRunner(resources=[resource, logger])

        runner.run("first_routine")
        runner.run("second_routine")

        assert session_ids_seen == ["first_routine", "second_routine"]
        assert logger.current_session is None

    def test_initialization_with_configs(self, registry_fixture):
        """Test initializing RoutineRunner with device and logger resources."""
        device = MockResource("device")
        logger = MockResource("logger")
        runner = RoutineRunner(resources=[device, logger])

        assert "device" in runner.resources.list_resources()
        assert "logger" in runner.resources.list_resources()
        assert len(runner._device_configs) == 0
        assert runner.configs == {}

    def test_run_routine_tree_with_unregistered_routine(
        self, registry_fixture, nested_routines_yaml
    ):
        """Test that routine tree execution skips unregistered routines gracefully."""
        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        runner = RoutineRunner(resources=[MockResource("device")])
        runner._device_configs = [device_config]

        results = {}
        routine_config = device_config.routines[0]
        runner._run_routine_tree(routine_config, results)

        assert len(results) == 0

    def test_initialization_with_configs_using_mock(
        self, registry_fixture, simple_routines_yaml, tmp_path
    ):
        """Test initializing RoutineRunner from device config with mocked dependencies."""
        from unittest.mock import Mock, patch

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(simple_routines_yaml)
        )

        mock_device = Mock()
        mock_device.name = "test_device"

        with patch("stanza.routines.core.device_from_config", return_value=mock_device):
            with patch("stanza.routines.core.DataLogger") as mock_logger_class:
                with patch(
                    "stanza.routines.core.StanzaSession.get_active_session",
                    return_value=None,
                ):
                    mock_logger = Mock()
                    mock_logger.name = "logger"
                    mock_logger_class.return_value = mock_logger

                    runner = RoutineRunner(configs=[device_config])

                    assert "device" in runner.resources.list_resources()
                    assert "logger" in runner.resources.list_resources()
                    assert len(runner._device_configs) == 1
                    assert runner.configs["routine1"]["param1"] == "value1"
                    assert runner.configs["routine2"]["param2"] == "value2"

                    mock_logger_class.assert_called_once_with(
                        name="logger", routine_name="device", base_dir="./data"
                    )


class TestParentParameterInheritance:
    """Test that parent routine parameters are inherited by subroutines."""

    def test_parent_params_inherited_by_children(
        self, registry_fixture, nested_routines_yaml
    ):
        """Test that child routines inherit parent parameters."""
        received_params = {}

        @routine
        def leakage_test(
            ctx,
            parent_param,
            charge_carrier_type,
            leakage_threshold_resistance,
            leakage_threshold_count,
        ):
            received_params["parent_param"] = parent_param
            received_params["charge_carrier_type"] = charge_carrier_type
            received_params["leakage_threshold_resistance"] = (
                leakage_threshold_resistance
            )
            received_params["leakage_threshold_count"] = leakage_threshold_count
            return "leakage_result"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        runner = RoutineRunner(resources=[MockResource("device")])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        results = runner.run_all(parent_routine="health_check")

        assert received_params["parent_param"] == "value"
        assert received_params["charge_carrier_type"] == "holes"
        assert received_params["leakage_threshold_resistance"] == 50000000.0
        assert received_params["leakage_threshold_count"] == 0
        assert results["leakage_test"] == "leakage_result"

    def test_child_params_override_parent_params(
        self, registry_fixture, param_override_yaml
    ):
        """Test that child parameters override parent parameters with the same name."""
        received_params = {}

        @routine
        def child_routine(ctx, shared_param, parent_only, child_only):
            received_params["shared_param"] = shared_param
            received_params["parent_only"] = parent_only
            received_params["child_only"] = child_only
            return "child_result"

        device_config = DeviceConfig.model_validate(yaml.safe_load(param_override_yaml))
        runner = RoutineRunner(resources=[MockResource("device")])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        results = runner.run_all(parent_routine="parent_routine")

        assert received_params["shared_param"] == "child_value"
        assert received_params["parent_only"] == "parent_data"
        assert received_params["child_only"] == "child_data"
        assert results["child_routine"] == "child_result"

    def test_multiple_children_inherit_same_parent_params(
        self, registry_fixture, nested_routines_yaml
    ):
        """Test that multiple child routines all inherit parent parameters."""
        received_leakage_params = {}
        received_accumulation_params = {}

        @routine
        def leakage_test(ctx, **params):
            received_leakage_params["charge_carrier_type"] = params[
                "charge_carrier_type"
            ]
            received_leakage_params["leakage_threshold_resistance"] = params[
                "leakage_threshold_resistance"
            ]
            return "leakage_result"

        @routine
        def global_accumulation(ctx, **params):
            received_accumulation_params["charge_carrier_type"] = params[
                "charge_carrier_type"
            ]
            received_accumulation_params["step_size"] = params["step_size"]
            return "accumulation_result"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        runner = RoutineRunner(resources=[MockResource("device")])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        runner.run_all(parent_routine="health_check")

        assert received_leakage_params["charge_carrier_type"] == "holes"
        assert received_accumulation_params["charge_carrier_type"] == "holes"
        assert received_leakage_params["leakage_threshold_resistance"] == 50000000.0
        assert received_accumulation_params["step_size"] == 0.01

    def test_deeply_nested_param_inheritance(
        self, registry_fixture, deeply_nested_routines_yaml
    ):
        """Test that parameters are inherited across multiple nesting levels."""
        received_level2_params = {}
        received_level3_params = {}

        @routine
        def level2(ctx, level):
            received_level2_params["level"] = level
            return "level2_result"

        @routine
        def level3(ctx, level):
            received_level3_params["level"] = level
            return "level3_result"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(deeply_nested_routines_yaml)
        )
        runner = RoutineRunner(resources=[MockResource("device")])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        runner.run_all(parent_routine="level1")

        assert received_level2_params["level"] == 2
        assert received_level3_params["level"] == 3

    def test_direct_run_inherits_parent_params(
        self, registry_fixture, nested_routines_yaml
    ):
        """Test that direct routine run inherits parent parameters."""
        received_params = {}

        @routine
        def global_accumulation(ctx, **params):
            received_params.update(params)
            return "accumulation_result"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        runner = RoutineRunner(resources=[MockResource("device")])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        for routine_config in device_config.routines:
            runner._extract_from_routine_config(routine_config, runner.configs)

        runner.run("global_accumulation")

        assert "charge_carrier_type" in received_params
        assert received_params["charge_carrier_type"] == "holes"
        assert received_params["step_size"] == 0.01
