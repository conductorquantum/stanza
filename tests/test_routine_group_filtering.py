"""Tests for RoutineRunner group filtering functionality."""

import pytest

from stanza.exceptions import DeviceError
from stanza.models import DeviceConfig, DeviceGroup, GateType, PadType
from stanza.routines import (
    RoutineContext,
    RoutineRunner,
    clear_routine_registry,
    routine,
)
from tests.conftest import (
    MockControlInstrument,
    MockMeasurementInstrument,
    make_gate,
    standard_instrument_configs,
)


def get_gates_from_configs(configs):
    """Extract gate names from channel configs."""
    return [name for name, config in configs.items() if config.pad_type == PadType.GATE]


@pytest.fixture
def registry_fixture():
    """Clear routine registry before and after each test."""
    clear_routine_registry()
    yield
    clear_routine_registry()


@pytest.fixture
def device_with_groups(create_device):
    """Create a device with multiple groups and shared RESERVOIR gates."""
    device_config = DeviceConfig(
        name="device",
        gates={
            "G1": make_gate(GateType.PLUNGER, control_channel=1),
            "G2": make_gate(GateType.BARRIER, control_channel=2),
            "G3": make_gate(GateType.PLUNGER, control_channel=3),
            "G4": make_gate(GateType.BARRIER, control_channel=4),
            "RES1": make_gate(GateType.RESERVOIR, control_channel=5),
            "RES2": make_gate(GateType.RESERVOIR, control_channel=6),
        },
        contacts={},
        groups={
            "control": DeviceGroup(gates=["G1", "G2", "RES1", "RES2"]),
            "sensor": DeviceGroup(gates=["G3", "G4", "RES1", "RES2"]),
        },
        routines=[],
        instruments=standard_instrument_configs(),
    )

    control_inst = MockControlInstrument()
    measure_inst = MockMeasurementInstrument()

    device = create_device(device_config, control_inst, measure_inst)

    return device, control_inst, measure_inst


@pytest.fixture
def routine_runner_with_grouped_device(device_with_groups):
    """Create a RoutineRunner with a device that has groups."""
    device, control_inst, measure_inst = device_with_groups

    runner = RoutineRunner(resources=[device])

    return runner, device, control_inst, measure_inst


class TestRoutineDeviceFiltering:
    """Test that RoutineRunner correctly filters device by group."""

    @pytest.mark.parametrize(
        "group,expected_gates",
        [
            ("control", {"G1", "G2", "RES1", "RES2"}),
            ("sensor", {"G3", "G4", "RES1", "RES2"}),
        ],
    )
    def test_routine_receives_filtered_device_with_only_group_gates(
        self,
        registry_fixture,
        routine_runner_with_grouped_device,
        group,
        expected_gates,
    ):
        """Test that routine receives full device plus group_configs with only the group's gates."""
        runner, original_device, control_inst, measure_inst = (
            routine_runner_with_grouped_device
        )

        # Track what device and group_configs the routine receives
        received_device = None
        received_group_configs = None

        @routine(name="test_routine")
        def capture_device_routine(ctx: RoutineContext) -> dict:
            nonlocal received_device, received_group_configs
            received_device = ctx.resources.device
            received_group_configs = getattr(ctx.resources, "group_configs", None)
            return {"gates": list(ctx.resources.device.gates)}

        # Run with specified group
        runner.run("test_routine", group=group)

        # Verify routine received the original device (not filtered)
        assert received_device is not None
        assert received_device is original_device
        assert set(received_device.gates) == set(original_device.gates)

        # Verify group_configs were added and contain only group gates
        assert received_group_configs is not None
        assert set(get_gates_from_configs(received_group_configs)) == expected_gates

    def test_device_name_unchanged_with_group(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that device name remains unchanged when group is specified."""
        runner, original_device, _, _ = routine_runner_with_grouped_device

        received_device_name = None

        @routine(name="test_routine")
        def capture_name_routine(ctx: RoutineContext) -> dict:
            nonlocal received_device_name
            received_device_name = ctx.resources.device.name
            return {}

        runner.run("test_routine", group="control")

        # Device name should still be "device" (not "device_control")
        assert received_device_name == "device"
        assert received_device_name == original_device.name

    def test_filter_unknown_group_raises_error(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that filtering by unknown group raises DeviceError."""
        runner, _, _, _ = routine_runner_with_grouped_device

        @routine(name="test_routine")
        def simple_routine(ctx: RoutineContext, **kwargs) -> dict:
            return {}

        with pytest.raises(DeviceError, match="Group 'unknown' not found"):
            runner.run("test_routine", group="unknown")

    def test_group_parameter_works(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that 'group' parameter filters the device correctly."""
        runner, original_device, _, _ = routine_runner_with_grouped_device

        received_device = None
        received_group_configs = None

        @routine(name="test_routine")
        def capture_device_routine(ctx: RoutineContext) -> dict:
            nonlocal received_device, received_group_configs
            received_device = ctx.resources.device
            received_group_configs = getattr(ctx.resources, "group_configs", None)
            return {"gates": list(ctx.resources.device.gates)}

        # Run with 'group' parameter
        runner.run("test_routine", group="control")

        # Verify routine received original device and group_configs
        assert received_device is not None
        assert received_device is original_device
        assert received_group_configs is not None
        assert set(get_gates_from_configs(received_group_configs)) == {
            "G1",
            "G2",
            "RES1",
            "RES2",
        }


class TestDeviceRestoration:
    """Test that original device is properly restored after routine execution."""

    def test_original_device_restored_after_routine(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that original device is restored after successful routine."""
        runner, original_device, _, _ = routine_runner_with_grouped_device

        @routine(name="test_routine")
        def simple_routine(ctx: RoutineContext, **kwargs) -> dict:
            return {}

        # Run routine with group filtering
        runner.run("test_routine", group="control")

        # Original device should be restored in resources
        restored_device = runner.resources.device
        assert restored_device is original_device
        assert set(restored_device.gates) == {"G1", "G2", "G3", "G4", "RES1", "RES2"}

    def test_original_device_restored_on_routine_failure(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that original device is restored even if routine raises exception."""
        runner, original_device, _, _ = routine_runner_with_grouped_device

        @routine(name="failing_routine")
        def failing_routine(ctx: RoutineContext) -> dict:
            raise RuntimeError("Routine failed")

        # Run routine with group filtering - should raise error
        with pytest.raises(RuntimeError, match="Routine failed"):
            runner.run("failing_routine", group="control")

        # Original device should STILL be restored despite exception
        restored_device = runner.resources.device
        assert restored_device is original_device
        assert set(restored_device.gates) == {"G1", "G2", "G3", "G4", "RES1", "RES2"}

    def test_sequential_routines_with_different_groups(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that sequential routines with different groups get correct devices."""
        runner, original_device, _, _ = routine_runner_with_grouped_device

        devices_received = []

        @routine(name="test_routine")
        def capture_device_routine(ctx: RoutineContext) -> dict:
            devices_received.append(
                {
                    "name": ctx.resources.device.name,
                    "gates": set(ctx.resources.device.gates),
                }
            )
            return {}

        # Run with control group
        runner.run("test_routine", group="control")

        # Run with sensor group
        runner.run("test_routine", group="sensor")

        # Run without group (should get original device)
        runner.run("test_routine")

        # Verify each routine got the correct device
        assert len(devices_received) == 3

        # All routines should receive the original device (name stays "device")
        assert devices_received[0]["name"] == "device"
        assert devices_received[1]["name"] == "device"
        assert devices_received[2]["name"] == "device"

        # All routines see the same gates (full device gates)
        all_gates = {"G1", "G2", "G3", "G4", "RES1", "RES2"}
        assert devices_received[0]["gates"] == all_gates
        assert devices_received[1]["gates"] == all_gates
        assert devices_received[2]["gates"] == all_gates


class TestGroupFilteringInstrumentSharing:
    """Test that filtered devices share instrument instances."""

    def test_filtered_device_shares_instruments(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that filtered device shares the same instrument instances."""
        runner, original_device, control_inst, measure_inst = (
            routine_runner_with_grouped_device
        )

        received_instruments = {}

        @routine(name="test_routine")
        def capture_instruments_routine(ctx: RoutineContext) -> dict:
            nonlocal received_instruments
            received_instruments = {
                "control": ctx.resources.device.control_instrument,
                "measurement": ctx.resources.device.measurement_instrument,
            }
            return {}

        runner.run("test_routine", group="control")

        # Filtered device should share the same instrument instances
        assert received_instruments["control"] is control_inst
        assert received_instruments["measurement"] is measure_inst
        assert received_instruments["control"] is original_device.control_instrument


class TestGroupFilteringWithLogger:
    """Tests for group filtering integration with data logger."""

    def test_group_name_included_in_logger_session_path(
        self, registry_fixture, routine_runner_with_grouped_device, tmp_path
    ):
        """Test that group name is included in logger session directory path."""
        import tempfile

        from stanza.logger.data_logger import DataLogger

        runner, _, _, _ = routine_runner_with_grouped_device

        # Create a data logger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )
            runner.context.resources.add("logger", logger)

            @routine(name="test_routine")
            def test_routine(ctx: RoutineContext, session=None) -> dict:
                # Session should have group in its ID
                if session:
                    assert session.session_id == "test_routine_control"
                    assert session.metadata.group_name == "control"
                return {}

            runner.run("test_routine", group="control")

            # Verify directory with group suffix was created
            session_dir = logger.base_directory / "test_routine_control"
            assert session_dir.exists()

    def test_different_groups_create_separate_directories(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that different groups create separate output directories."""
        import tempfile

        from stanza.logger.data_logger import DataLogger

        runner, _, _, _ = routine_runner_with_grouped_device

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )
            runner.context.resources.add("logger", logger)

            @routine(name="test_routine")
            def test_routine(ctx: RoutineContext, session=None) -> dict:
                if session:
                    session.log_measurement("value", {"data": 1})
                return {}

            # Run for control group
            runner.run("test_routine", group="control")

            # Run for sensor group
            runner.run("test_routine", group="sensor")

            # Verify separate directories exist
            control_dir = logger.base_directory / "test_routine_control"
            sensor_dir = logger.base_directory / "test_routine_sensor"

            assert control_dir.exists()
            assert sensor_dir.exists()

            # Verify both have their own data files
            assert (control_dir / "measurement.jsonl").exists()
            assert (sensor_dir / "measurement.jsonl").exists()

    def test_routine_without_group_creates_path_without_suffix(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test routines without groups don't get suffix."""
        import tempfile

        from stanza.logger.data_logger import DataLogger

        runner, _, _, _ = routine_runner_with_grouped_device

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )
            runner.context.resources.add("logger", logger)

            @routine(name="test_routine")
            def test_routine(ctx: RoutineContext, session=None) -> dict:
                if session:
                    # No group specified, so no suffix
                    assert session.session_id == "test_routine"
                    assert session.metadata.group_name is None
                return {}

            # Run without group
            runner.run("test_routine")

            # Verify directory without group suffix
            session_dir = logger.base_directory / "test_routine"
            assert session_dir.exists()

            # Verify no group-suffixed directory was created
            assert not (logger.base_directory / "test_routine_control").exists()
            assert not (logger.base_directory / "test_routine_sensor").exists()
