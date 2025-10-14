"""Tests for routine runner integration with DataLogger, focusing on nested session paths."""

import pytest
import yaml

from stanza.logger.data_logger import DataLogger
from stanza.models import DeviceConfig
from stanza.routines import RoutineRunner, clear_routine_registry, routine


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
def nested_routines_yaml():
    """Fixture providing device YAML configuration with nested routines."""
    return """
name: "Test Device with Nested Routines"
contacts:
  IN:
    type: SOURCE
    control_channel: 1
    measure_channel: 1
    v_lower_bound: -3.0
    v_upper_bound: 3.0
gates:
  G1:
    type: PLUNGER
    control_channel: 2
    measure_channel: 2
    v_lower_bound: -3.0
    v_upper_bound: 3.0
routines:
  - name: health_check
    parameters:
      parent_param: value
    routines:
      - name: leakage_test
        parameters:
          leakage_threshold_resistance: 50000000.0
          leakage_threshold_count: 0
      - name: global_accumulation
        parameters:
          step_size: 0.01
      - name: reservoir_characterization
        parameters:
          step_size: 0.01
instruments:
  - name: test_control
    type: CONTROL
    driver: qdac2
    ip_addr: "127.0.0.1"
    slew_rate: 1.0
  - name: test_measurement
    type: MEASUREMENT
    driver: qdac2
    ip_addr: "127.0.0.1"
    measurement_duration: 0.001
    sample_time: 0.00001
"""


@pytest.fixture
def deeply_nested_routines_yaml():
    """Fixture providing device YAML configuration with deeply nested routines (3+ levels)."""
    return """
name: "Test Device with Deeply Nested Routines"
contacts:
  IN:
    type: SOURCE
    control_channel: 1
    measure_channel: 1
    v_lower_bound: -3.0
    v_upper_bound: 3.0
gates:
  G1:
    type: PLUNGER
    control_channel: 2
    measure_channel: 2
    v_lower_bound: -3.0
    v_upper_bound: 3.0
routines:
  - name: level1
    parameters:
      level: 1
    routines:
      - name: level2
        parameters:
          level: 2
        routines:
          - name: level3
            parameters:
              level: 3
instruments:
  - name: test_control
    type: CONTROL
    driver: qdac2
    ip_addr: "127.0.0.1"
    slew_rate: 1.0
  - name: test_measurement
    type: MEASUREMENT
    driver: qdac2
    ip_addr: "127.0.0.1"
    measurement_duration: 0.001
    sample_time: 0.00001
"""


@pytest.fixture
def simple_routines_yaml():
    """Fixture providing device YAML configuration with simple top-level routines."""
    return """
name: test
gates:
  G1:
    type: PLUNGER
    control_channel: 1
    measure_channel: 1
    v_lower_bound: -1.0
    v_upper_bound: 1.0
contacts:
  C1:
    type: SOURCE
    control_channel: 2
    measure_channel: 2
    v_lower_bound: -1.0
    v_upper_bound: 1.0
routines:
  - name: routine1
    parameters:
      param1: value1
  - name: routine2
    parameters:
      param2: value2
instruments:
  - name: ctrl
    type: CONTROL
    ip_addr: "127.0.0.1"
    slew_rate: 1.0
  - name: meas
    type: MEASUREMENT
    ip_addr: "127.0.0.1"
    measurement_duration: 0.001
    sample_time: 0.0001
"""


class TestRoutinePathGeneration:
    """Tests for _get_routine_path method."""

    def test_get_routine_path_top_level(self, registry_fixture, simple_routines_yaml):
        """Test that top-level routines return just their name."""
        device_config = DeviceConfig.model_validate(
            yaml.safe_load(simple_routines_yaml)
        )
        runner = RoutineRunner(
            resources=[MockResource("device"), MockResource("logger")]
        )
        runner._device_configs = [device_config]
        runner._extract_routine_configs([device_config])

        path = runner._get_routine_path("routine1")

        assert path == "routine1"

    def test_get_routine_path_nested_single_level(
        self, registry_fixture, nested_routines_yaml
    ):
        """Test that single-level nested routines return 'parent/child'."""
        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        runner = RoutineRunner(
            resources=[MockResource("device"), MockResource("logger")]
        )
        runner._device_configs = [device_config]
        runner._extract_routine_configs([device_config])

        leakage_path = runner._get_routine_path("leakage_test")
        accumulation_path = runner._get_routine_path("global_accumulation")
        reservoir_path = runner._get_routine_path("reservoir_characterization")

        assert leakage_path == "health_check/leakage_test"
        assert accumulation_path == "health_check/global_accumulation"
        assert reservoir_path == "health_check/reservoir_characterization"

    def test_get_routine_path_deeply_nested(
        self, registry_fixture, deeply_nested_routines_yaml
    ):
        """Test that deeply nested routines (3+ levels) return correct paths."""
        device_config = DeviceConfig.model_validate(
            yaml.safe_load(deeply_nested_routines_yaml)
        )
        runner = RoutineRunner(
            resources=[MockResource("device"), MockResource("logger")]
        )
        runner._device_configs = [device_config]
        runner._extract_routine_configs([device_config])

        level2_path = runner._get_routine_path("level2")
        level3_path = runner._get_routine_path("level3")

        assert level2_path == "level1/level2"
        assert level3_path == "level1/level2/level3"


class TestNestedSessionCreation:
    """Tests for hierarchical session ID creation and directory structure."""

    def test_nested_routine_creates_hierarchical_session_id(
        self, registry_fixture, nested_routines_yaml, tmp_path
    ):
        """Test that nested routines create sessions with hierarchical IDs."""
        session_ids_created = []

        @routine
        def leakage_test(ctx, session=None, **params):
            if session:
                session_ids_created.append(session.session_id)
            return "leakage_result"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        device = MockResource("device")
        logger = DataLogger(
            name="logger", routine_name="test", base_dir=tmp_path / "data"
        )
        runner = RoutineRunner(resources=[device, logger])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        runner.run("leakage_test")

        assert len(session_ids_created) == 1
        assert session_ids_created[0] == "health_check/leakage_test"

    def test_nested_routines_create_separate_session_directories(
        self, registry_fixture, nested_routines_yaml, tmp_path
    ):
        """Test that nested routines create separate subdirectories."""

        @routine
        def leakage_test(ctx, session=None, **params):
            return "leakage"

        @routine
        def global_accumulation(ctx, session=None, **params):
            return "accumulation"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        device = MockResource("device")
        logger = DataLogger(
            name="logger", routine_name="test", base_dir=tmp_path / "data"
        )
        runner = RoutineRunner(resources=[device, logger])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        runner.run("leakage_test")
        runner.run("global_accumulation")

        base_dir = tmp_path / "data" / "test"
        leakage_dir = base_dir / "health_check" / "leakage_test"
        accumulation_dir = base_dir / "health_check" / "global_accumulation"

        assert leakage_dir.exists()
        assert accumulation_dir.exists()

    def test_deeply_nested_routines_create_deep_directories(
        self, registry_fixture, deeply_nested_routines_yaml, tmp_path
    ):
        """Test that deeply nested routines create properly nested directories."""

        @routine
        def level2(ctx, session=None, **params):
            return "level2"

        @routine
        def level3(ctx, session=None, **params):
            return "level3"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(deeply_nested_routines_yaml)
        )
        device = MockResource("device")
        logger = DataLogger(
            name="logger", routine_name="test", base_dir=tmp_path / "data"
        )
        runner = RoutineRunner(resources=[device, logger])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        runner.run("level2")
        runner.run("level3")

        base_dir = tmp_path / "data" / "test"
        level2_dir = base_dir / "level1" / "level2"
        level3_dir = base_dir / "level1" / "level2" / "level3"

        assert level2_dir.exists()
        assert level3_dir.exists()


class TestSessionLifecycle:
    """Tests for session creation and cleanup with nested paths."""

    def test_session_closes_properly_with_nested_paths(
        self, registry_fixture, nested_routines_yaml, tmp_path
    ):
        """Test that sessions with nested paths close properly."""

        @routine
        def leakage_test(ctx, session=None, **params):
            assert session is not None
            return "success"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        device = MockResource("device")
        logger = DataLogger(
            name="logger", routine_name="test", base_dir=tmp_path / "data"
        )
        runner = RoutineRunner(resources=[device, logger])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        runner.run("leakage_test")

        assert logger.current_session is None
        assert "health_check/leakage_test" not in logger._active_sessions


class TestRunAllWithNesting:
    """Tests for run_all method with nested routines."""

    def test_run_all_with_nested_routines_creates_hierarchical_sessions(
        self, registry_fixture, nested_routines_yaml, tmp_path
    ):
        """Test that run_all with nested routines creates hierarchical session paths."""
        sessions_created = []

        @routine
        def leakage_test(ctx, session=None, **params):
            if session:
                sessions_created.append(session.session_id)
            return "leakage"

        @routine
        def global_accumulation(ctx, session=None, **params):
            if session:
                sessions_created.append(session.session_id)
            return "accumulation"

        @routine
        def reservoir_characterization(ctx, session=None, **params):
            if session:
                sessions_created.append(session.session_id)
            return "reservoir"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        device = MockResource("device")
        logger = DataLogger(
            name="logger", routine_name="test", base_dir=tmp_path / "data"
        )
        runner = RoutineRunner(resources=[device, logger])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        runner.run_all(parent_routine="health_check")

        assert len(sessions_created) == 3
        assert "health_check/leakage_test" in sessions_created
        assert "health_check/global_accumulation" in sessions_created
        assert "health_check/reservoir_characterization" in sessions_created


class TestRoutineHierarchyInitialization:
    """Tests for _routine_hierarchy attribute initialization."""

    def test_routine_hierarchy_initialization_with_resources(self, registry_fixture):
        """Test that _routine_hierarchy is initialized when using resources."""
        resource = MockResource("resource")
        runner = RoutineRunner(resources=[resource])

        assert hasattr(runner, "_routine_hierarchy")
        assert runner._routine_hierarchy == {}

    def test_routine_hierarchy_initialization_with_configs(
        self, registry_fixture, simple_routines_yaml
    ):
        """Test that _routine_hierarchy is initialized when using configs."""
        device_config = DeviceConfig.model_validate(
            yaml.safe_load(simple_routines_yaml)
        )
        device = MockResource("device")
        logger = MockResource("logger")
        runner = RoutineRunner(resources=[device, logger])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        assert hasattr(runner, "_routine_hierarchy")
        assert isinstance(runner._routine_hierarchy, dict)


class TestMixedRoutineTypes:
    """Tests for mixed top-level and nested routines."""

    def test_mixed_top_level_and_nested_routines(
        self, registry_fixture, nested_routines_yaml, tmp_path
    ):
        """Test that mix of top-level and nested routines work correctly."""

        @routine
        def health_check(ctx, session=None, **params):
            return "health_check"

        @routine
        def leakage_test(ctx, session=None, **params):
            return "leakage"

        device_config = DeviceConfig.model_validate(
            yaml.safe_load(nested_routines_yaml)
        )
        device = MockResource("device")
        logger = DataLogger(
            name="logger", routine_name="test", base_dir=tmp_path / "data"
        )
        runner = RoutineRunner(resources=[device, logger])
        runner.configs = runner._extract_routine_configs([device_config])
        runner._device_configs = [device_config]

        runner.run("health_check")
        health_check_path = runner._get_routine_path("health_check")

        runner.run("leakage_test")
        leakage_path = runner._get_routine_path("leakage_test")

        assert health_check_path == "health_check"
        assert leakage_path == "health_check/leakage_test"
