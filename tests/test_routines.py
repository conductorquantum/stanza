import pytest

from stanza.routines.datatypes import ResourceRegistry, ResultsRegistry, RoutineContext
from stanza.routines.runner import (
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

    def test_getitem_access(self, resource_registry):
        assert resource_registry["resource1"].name == "resource1"

    def test_get_method(self, resource_registry):
        assert resource_registry.get("resource1").name == "resource1"
        assert resource_registry.get("nonexistent", "default") == "default"

    def test_add_resource(self):
        registry = ResourceRegistry()
        resource = MockResource("new_resource")
        registry.add("new_resource", resource)

        assert registry.new_resource is resource

    def test_list_resources(self, resource_registry):
        resources = resource_registry.list_resources()
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

    def test_getattr_access(self, results_registry):
        assert results_registry.result1 == "value1"

    def test_getitem_setitem_access(self):
        registry = ResultsRegistry()
        registry["test_result"] = "test_value"

        assert registry["test_result"] == "test_value"

    def test_get_with_default(self):
        registry = ResultsRegistry()
        assert registry.get("nonexistent", "default") == "default"

    def test_list_results(self, results_registry):
        results = results_registry.list_results()
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
    def test_routine_decorator_registers_function(self, registry_fixture):
        @routine
        def test_routine(ctx):
            return "test_result"

        registered = get_registered_routines()
        assert "test_routine" in registered
        assert registered["test_routine"] == test_routine

    def test_routine_decorator_with_custom_name(self, registry_fixture):
        @routine(name="custom_name")
        def test_routine(ctx):
            return "test_result"

        registered = get_registered_routines()
        assert "custom_name" in registered
        assert "test_routine" not in registered

    def test_routine_decorator_returns_original_function(self, registry_fixture):
        def original_func(ctx):
            return "original"

        decorated = routine()(original_func)
        assert decorated is original_func

    def test_multiple_routine_registrations(self, registry_fixture):
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
        @routine
        def test_routine(ctx):
            pass

        assert len(get_registered_routines()) == 1
        clear_routine_registry()
        assert len(get_registered_routines()) == 0

    def test_get_registered_routines_returns_copy(self, registry_fixture):
        @routine
        def test_routine(ctx):
            pass

        registered1 = get_registered_routines()
        registered2 = get_registered_routines()

        assert registered1 == registered2
        assert registered1 is not registered2


class TestRoutineRunner:
    def test_initialization_with_resources(self, registry_fixture):
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")

        runner = RoutineRunner(resources=[resource1, resource2])

        assert runner.resources.resource1 is resource1
        assert runner.resources.resource2 is resource2
        assert runner.results.list_results() == []
        assert runner.configs == {}

    def test_initialization_requires_resources_or_configs(self, registry_fixture):
        with pytest.raises(
            ValueError, match="Must provide either 'resources' or 'configs'"
        ):
            RoutineRunner()

    def test_initialization_cannot_provide_both(self, registry_fixture):
        pass

    def test_run_routine_not_registered(self, registry_fixture):
        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        with pytest.raises(ValueError, match="Routine 'nonexistent' not registered"):
            runner.run("nonexistent")

    def test_run_routine_basic(self, registry_fixture):
        @routine
        def test_routine(ctx):
            return "success"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])
        result = runner.run("test_routine")

        assert result == "success"
        assert runner.get_result("test_routine") == "success"

    def test_run_routine_with_context_access(self, registry_fixture):
        @routine
        def test_routine(ctx):
            resource = ctx.resources.test_resource
            return f"data_{resource.data}"

        resource = MockResource("test_resource")
        runner = RoutineRunner(resources=[resource])
        result = runner.run("test_routine")

        assert result == "data_data_from_test_resource"

    def test_run_routine_with_params(self, registry_fixture):
        @routine
        def test_routine(ctx, param1, param2="default"):
            return f"{param1}-{param2}"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])
        result = runner.run("test_routine", param1="hello", param2="world")

        assert result == "hello-world"

    def test_run_routine_with_config_params(self, registry_fixture):
        @routine
        def test_routine(ctx, param1, param2="default"):
            return f"{param1}-{param2}"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        runner.configs["test_routine"] = {"param1": "value1", "param2": 42}

        result = runner.run("test_routine")

        assert result == "value1-42"

    def test_run_routine_user_params_override_config(self, registry_fixture):
        @routine
        def test_routine(ctx, param1, param2="default"):
            return f"{param1}-{param2}"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        runner.configs["test_routine"] = {"param1": "value1", "param2": 42}

        result = runner.run("test_routine", param2="overridden")

        assert result == "value1-overridden"

    def test_run_routine_access_previous_results(self, registry_fixture):
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
        @routine
        def failing_routine(ctx):
            raise ValueError("Something went wrong")

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        with pytest.raises(RuntimeError, match="Routine 'failing_routine' failed"):
            runner.run("failing_routine")

    def test_get_result(self, registry_fixture):
        @routine
        def test_routine(ctx):
            return "test_result"

        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        runner.run("test_routine")

        assert runner.get_result("test_routine") == "test_result"
        assert runner.get_result("nonexistent") is None

    def test_list_routines(self, registry_fixture):
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
