import pytest

from stanza.routines.datatypes import ResourceRegistry, ResultsRegistry, RoutineContext
from stanza.routines.runner import (
    RoutineRunner,
    clear_routine_registry,
    get_registered_routines,
    routine,
)


class MockResource:
    """Mock resource for testing."""

    def __init__(self, name: str):
        self.name = name
        self.data = f"data_from_{name}"


class TestResourceRegistry:
    def test_initialization_with_named_resources(self):
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")
        registry = ResourceRegistry(resource1, resource2)

        assert registry.resource1 is resource1
        assert registry.resource2 is resource2

    def test_getattr_access(self):
        resource = MockResource("test_resource")
        registry = ResourceRegistry(resource)

        assert registry.test_resource is resource
        assert registry.test_resource.data == "data_from_test_resource"

    def test_getitem_access(self):
        resource = MockResource("test_resource")
        registry = ResourceRegistry(resource)

        assert registry["test_resource"] is resource

    def test_get_method(self):
        resource = MockResource("test_resource")
        registry = ResourceRegistry(resource)

        assert registry.get("test_resource") is resource
        assert registry.get("nonexistent", "default") == "default"

    def test_add_resource(self):
        registry = ResourceRegistry()
        resource = MockResource("new_resource")

        registry.add("new_resource", resource)
        assert registry.new_resource is resource

    def test_list_resources(self):
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")
        registry = ResourceRegistry(resource1, resource2)

        resources = registry.list_resources()
        assert "resource1" in resources
        assert "resource2" in resources
        assert len(resources) == 2

    def test_nonexistent_resource_raises_attribute_error(self):
        registry = ResourceRegistry()

        with pytest.raises(AttributeError, match="Resource 'nonexistent' not found"):
            _ = registry.nonexistent

    def test_private_attribute_access(self):
        registry = ResourceRegistry()

        with pytest.raises(AttributeError):
            _ = registry._private_attr


class TestResultsRegistry:
    def test_initialization(self):
        registry = ResultsRegistry()
        assert registry.list_results() == []

    def test_store_and_get(self):
        registry = ResultsRegistry()
        registry.store("test_result", {"data": "test"})

        assert registry.get("test_result") == {"data": "test"}

    def test_getattr_access(self):
        registry = ResultsRegistry()
        registry.store("test_result", "test_value")

        assert registry.test_result == "test_value"

    def test_getitem_setitem_access(self):
        registry = ResultsRegistry()
        registry["test_result"] = "test_value"

        assert registry["test_result"] == "test_value"

    def test_get_with_default(self):
        registry = ResultsRegistry()

        assert registry.get("nonexistent", "default") == "default"

    def test_list_results(self):
        registry = ResultsRegistry()
        registry.store("result1", "value1")
        registry.store("result2", "value2")

        results = registry.list_results()
        assert "result1" in results
        assert "result2" in results
        assert len(results) == 2

    def test_clear(self):
        registry = ResultsRegistry()
        registry.store("result1", "value1")

        assert len(registry.list_results()) == 1
        registry.clear()
        assert len(registry.list_results()) == 0

    def test_nonexistent_result_raises_attribute_error(self):
        registry = ResultsRegistry()

        with pytest.raises(AttributeError, match="Result 'nonexistent' not found"):
            _ = registry.nonexistent


class TestRoutineContext:
    def test_initialization(self):
        resources = ResourceRegistry()
        results = ResultsRegistry()
        context = RoutineContext(resources, results)

        assert context.resources is resources
        assert context.results is results


class TestRoutineDecorator:
    def setup_method(self):
        """Clear registry before each test."""
        clear_routine_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_routine_registry()

    def test_routine_decorator_registers_function(self):
        @routine
        def test_routine(ctx):
            return "test_result"

        registered = get_registered_routines()
        assert "test_routine" in registered
        assert registered["test_routine"] == test_routine

    def test_routine_decorator_with_custom_name(self):
        @routine(name="custom_name")
        def test_routine(ctx):
            return "test_result"

        registered = get_registered_routines()
        assert "custom_name" in registered
        assert "test_routine" not in registered

    def test_routine_decorator_returns_original_function(self):
        def original_func(ctx):
            return "original"

        decorated = routine()(original_func)
        assert decorated is original_func

    def test_multiple_routine_registrations(self):
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

    def test_clear_routine_registry(self):
        @routine
        def test_routine(ctx):
            pass

        assert len(get_registered_routines()) == 1
        clear_routine_registry()
        assert len(get_registered_routines()) == 0

    def test_get_registered_routines_returns_copy(self):
        @routine
        def test_routine(ctx):
            pass

        registered1 = get_registered_routines()
        registered2 = get_registered_routines()

        # Should be equal but not the same object
        assert registered1 == registered2
        assert registered1 is not registered2


class TestRoutineRunner:
    def setup_method(self):
        """Clear registry before each test."""
        clear_routine_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_routine_registry()

    def test_initialization_with_empty_configs(self, device):
        runner = RoutineRunner(device, [])
        assert runner.resources.test_device is device
        assert runner.results.list_results() == []
        assert runner.configs == {}

    def test_initialization_with_routine_configs(self, device, routine_configs):
        runner = RoutineRunner(device, routine_configs)

        assert runner.configs["test_routine"] == {"param1": "value1", "param2": 42}
        assert runner.configs["configured_routine"] == {
            "threshold": 1e-12,
            "multiplier": 2.5,
        }
        # no_params_routine should not be in configs since it has no parameters

    def test_initialization_unconfigured_device(self, device_no_instruments):
        with pytest.raises(ValueError, match="Device must be configured"):
            RoutineRunner(device_no_instruments, [])

    def test_run_routine_not_registered(self, device):
        runner = RoutineRunner(device, [])

        with pytest.raises(ValueError, match="Routine 'nonexistent' not registered"):
            runner.run("nonexistent")

    def test_run_routine_basic(self, device):
        @routine
        def test_routine(ctx):
            return "success"

        runner = RoutineRunner(device, [])
        result = runner.run("test_routine")

        assert result == "success"
        assert runner.get_result("test_routine") == "success"

    def test_run_routine_with_context_access(self, device):
        @routine
        def test_routine(ctx):
            # Test access to device via context
            device = ctx.resources.test_device
            return f"device_name_{device.name}"

        runner = RoutineRunner(device, [])
        result = runner.run("test_routine")

        assert result == "device_name_test_device"

    def test_run_routine_with_params(self, device):
        @routine
        def test_routine(ctx, param1, param2="default"):
            return f"{param1}-{param2}"

        runner = RoutineRunner(device, [])
        result = runner.run("test_routine", param1="hello", param2="world")

        assert result == "hello-world"

    def test_run_routine_with_config_params(self, device, routine_configs):
        @routine
        def test_routine(ctx, param1, param2="default"):
            return f"{param1}-{param2}"

        runner = RoutineRunner(device, routine_configs)
        result = runner.run("test_routine")

        assert result == "value1-42"

    def test_run_routine_user_params_override_config(self, device, routine_configs):
        @routine
        def test_routine(ctx, param1, param2="default"):
            return f"{param1}-{param2}"

        runner = RoutineRunner(device, routine_configs)
        result = runner.run("test_routine", param2="overridden")

        assert result == "value1-overridden"

    def test_run_routine_access_previous_results(self, device):
        @routine
        def first_routine(ctx):
            return {"data": "first_result"}

        @routine
        def second_routine(ctx):
            first_data = ctx.results.get("first_routine")
            return f"processed_{first_data['data']}"

        runner = RoutineRunner(device, [])
        runner.run("first_routine")
        result = runner.run("second_routine")

        assert result == "processed_first_result"

    def test_run_routine_exception_handling(self, device):
        @routine
        def failing_routine(ctx):
            raise ValueError("Something went wrong")

        runner = RoutineRunner(device, [])

        with pytest.raises(RuntimeError, match="Routine 'failing_routine' failed"):
            runner.run("failing_routine")

    def test_get_result(self, device):
        @routine
        def test_routine(ctx):
            return "test_result"

        runner = RoutineRunner(device, [])
        runner.run("test_routine")

        assert runner.get_result("test_routine") == "test_result"
        assert runner.get_result("nonexistent") is None

    def test_list_routines(self, device):
        @routine
        def routine1(ctx):
            pass

        @routine
        def routine2(ctx):
            pass

        runner = RoutineRunner(device, [])
        routines = runner.list_routines()

        assert "routine1" in routines
        assert "routine2" in routines
        assert len(routines) == 2

    def test_list_results(self, device):
        @routine
        def routine1(ctx):
            return "result1"

        @routine
        def routine2(ctx):
            return "result2"

        runner = RoutineRunner(device, [])
        runner.run("routine1")
        runner.run("routine2")

        results = runner.list_results()
        assert "routine1" in results
        assert "routine2" in results
        assert len(results) == 2

    def test_routine_with_no_config_parameters(self, device, routine_configs):
        @routine
        def no_params_routine(ctx):
            return "no_params_result"

        runner = RoutineRunner(device, routine_configs)
        result = runner.run("no_params_routine")

        assert result == "no_params_result"

    def test_sequential_routines_building_on_results(self, device):
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

        from stanza.models import RoutineConfig

        routine_configs = [
            RoutineConfig(name="collect_data", parameters={"data_size": 5}),
            RoutineConfig(name="process_data", parameters={"multiplier": 3}),
        ]
        runner = RoutineRunner(device, routine_configs)

        # Sequential routines
        collect_result = runner.run("collect_data")
        process_result = runner.run("process_data")
        analyze_result = runner.run("analyze_data")

        assert collect_result == [0, 1, 2, 3, 4]
        assert process_result == [0, 3, 6, 9, 12]
        assert analyze_result == {"sum": 30, "count": 5}


class TestIntegrationWithDevice:
    def setup_method(self):
        """Clear registry before each test."""
        clear_routine_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_routine_registry()

    def test_routine_using_device_methods(self, device):
        @routine
        def sweep_routine(ctx, gate, voltages, measure_contact):
            device = ctx.resources.test_device
            return device.sweep_1d(gate, voltages, measure_contact)

        device.measurement_instrument.measurements["contact1"] = 1e-6

        runner = RoutineRunner(device, [])
        voltages, currents = runner.run(
            "sweep_routine",
            gate="gate1",
            voltages=[0.0, 1.0, 2.0],
            measure_contact="contact1",
        )

        assert len(voltages) == 3
        assert len(currents) == 3
        assert all(i == 1e-6 for i in currents)

    def test_routine_using_device_jump_and_check(self, device):
        @routine
        def set_and_verify(ctx, gate, voltage):
            device = ctx.resources.test_device
            device.jump({gate: voltage})
            actual_voltage = device.check(gate)
            return {"requested": voltage, "actual": actual_voltage}

        runner = RoutineRunner(device, [])
        result = runner.run("set_and_verify", gate="gate1", voltage=1.5)

        assert result["requested"] == 1.5
        assert result["actual"] == 1.5

    def test_routine_with_device_properties(self, device):
        @routine
        def device_info(ctx):
            device = ctx.resources.test_device
            return {
                "gates": device.gates,
                "contacts": device.contacts,
                "control_gates": device.control_gates,
                "is_configured": device.is_configured(),
            }

        runner = RoutineRunner(device, [])
        result = runner.run("device_info")

        assert "gate1" in result["gates"]
        assert "contact1" in result["contacts"]
        assert "gate1" in result["control_gates"]
        assert result["is_configured"] is True

    def test_complex_routine_workflow(self, device):
        @routine
        def initialize_device(ctx):
            device = ctx.resources.test_device
            device.jump({"gate1": 0.0})
            return {"status": "initialized"}

        @routine
        def characterize_device(ctx, voltage_range=(-1.0, 1.0), num_points=5):
            device = ctx.resources.test_device
            start_v, end_v = voltage_range
            voltages = [
                start_v + i * (end_v - start_v) / (num_points - 1)
                for i in range(num_points)
            ]

            device.measurement_instrument.measurements["contact1"] = 1e-6

            results = {}
            for gate in device.control_gates:
                v_meas, i_meas = device.sweep_1d(gate, voltages, "contact1")
                results[gate] = {"voltages": v_meas, "currents": i_meas}

            return results

        @routine
        def analyze_characterization(ctx):
            init_result = ctx.results.get("initialize_device")
            char_result = ctx.results.get("characterize_device")

            if not init_result or not char_result:
                raise ValueError("Previous routines must be run first")

            analysis = {
                "initialization_status": init_result["status"],
                "gates_characterized": list(char_result.keys()),
                "total_measurements": sum(
                    len(data["currents"]) for data in char_result.values()
                ),
            }

            return analysis

        runner = RoutineRunner(device, [])

        init_result = runner.run("initialize_device")
        char_result = runner.run("characterize_device", num_points=3)
        analysis_result = runner.run("analyze_characterization")

        assert init_result["status"] == "initialized"
        assert "gate1" in char_result
        assert len(char_result["gate1"]["voltages"]) == 3
        assert analysis_result["initialization_status"] == "initialized"
        assert analysis_result["gates_characterized"] == ["gate1"]
        assert analysis_result["total_measurements"] == 3
