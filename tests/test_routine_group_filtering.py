"""Tests for RoutineRunner group filtering and zero_other_groups functionality."""

import pytest

from stanza.device import Device
from stanza.exceptions import DeviceError
from stanza.models import ContactType, DeviceConfig, DeviceGroup, GateType
from stanza.routines import (
    RoutineContext,
    RoutineRunner,
    clear_routine_registry,
    routine,
)
from stanza.utils import generate_channel_configs
from tests.conftest import (
    MockControlInstrument,
    MockMeasurementInstrument,
    make_contact,
    make_gate,
    standard_instrument_configs,
)


@pytest.fixture
def registry_fixture():
    """Clear routine registry before and after each test."""
    clear_routine_registry()
    yield
    clear_routine_registry()


@pytest.fixture
def device_with_groups():
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

    channel_configs = generate_channel_configs(device_config)

    control_inst = MockControlInstrument()
    measure_inst = MockMeasurementInstrument()

    device = Device(
        name="device",
        device_config=device_config,
        channel_configs=channel_configs,
        control_instrument=control_inst,
        measurement_instrument=measure_inst,
    )

    return device, control_inst, measure_inst


@pytest.fixture
def routine_runner_with_grouped_device(device_with_groups):
    """Create a RoutineRunner with a device that has groups."""
    device, control_inst, measure_inst = device_with_groups

    runner = RoutineRunner(resources=[device])

    return runner, device, control_inst, measure_inst


class TestRoutineDeviceFiltering:
    """Test that RoutineRunner correctly filters device by group."""

    def test_routine_receives_filtered_device_with_only_group_gates(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that routine receives a filtered device with only the group's gates."""
        runner, original_device, control_inst, measure_inst = (
            routine_runner_with_grouped_device
        )

        # Track what device the routine receives
        received_device = None

        @routine(name="test_routine")
        def capture_device_routine(ctx: RoutineContext) -> dict:
            nonlocal received_device
            received_device = ctx.resources.device
            return {"gates": list(ctx.resources.device.gates)}

        # Run with control group
        runner.run("test_routine", __group__="control")

        # Verify routine received filtered device
        assert received_device is not None
        assert set(received_device.gates) == {"G1", "G2", "RES1", "RES2"}
        assert set(received_device.gates) != set(original_device.gates)

        # Run with sensor group
        received_device = None
        runner.run("test_routine", __group__="sensor")

        assert received_device is not None
        assert set(received_device.gates) == {"G3", "G4", "RES1", "RES2"}

    def test_filtered_device_has_correct_name(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that filtered device name includes group name."""
        runner, original_device, _, _ = routine_runner_with_grouped_device

        received_device_name = None

        @routine(name="test_routine")
        def capture_name_routine(ctx: RoutineContext) -> dict:
            nonlocal received_device_name
            received_device_name = ctx.resources.device.name
            return {}

        runner.run("test_routine", __group__="control")

        assert received_device_name == "device_control"

    def test_filter_unknown_group_raises_error(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that filtering by unknown group raises DeviceError."""
        runner, _, _, _ = routine_runner_with_grouped_device

        @routine(name="test_routine")
        def simple_routine(ctx: RoutineContext, **kwargs) -> dict:
            return {}

        with pytest.raises(DeviceError, match="Group 'unknown' not found"):
            runner.run("test_routine", __group__="unknown")


class TestZeroOtherGroupsParameter:
    """Test zero_other_groups parameter functionality."""

    def test_zero_other_groups_zeros_non_group_gates(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that zero_other_groups=True zeros gates from other groups."""
        runner, original_device, control_inst, _ = routine_runner_with_grouped_device

        # Set initial voltages on all gates
        original_device.jump(
            {
                "G1": 0.5,
                "G2": 0.5,
                "G3": 0.5,
                "G4": 0.5,
                "RES1": 0.5,
                "RES2": 0.5,
            }
        )

        @routine(name="test_routine")
        def check_voltages_routine(ctx: RoutineContext, **kwargs) -> dict:
            # Check voltages during routine execution
            return {
                "G1": control_inst.voltages.get("G1"),
                "G3": control_inst.voltages.get("G3"),
            }

        # Run with control group and zero_other_groups=True
        runner.run("test_routine", __group__="control", zero_other_groups=True)

        # G3 and G4 (sensor group gates) should be zeroed
        assert control_inst.voltages["G3"] == 0.0
        assert control_inst.voltages["G4"] == 0.0

        # G1 and G2 (control group gates) should still have their values
        assert control_inst.voltages["G1"] == 0.5
        assert control_inst.voltages["G2"] == 0.5

    def test_zero_other_groups_does_not_zero_shared_gates(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that zero_other_groups=True does NOT zero shared RESERVOIR gates."""
        runner, original_device, control_inst, _ = routine_runner_with_grouped_device

        # Set initial voltages including shared reservoirs
        original_device.jump(
            {
                "G1": 0.5,
                "G3": 0.5,
                "RES1": 0.7,
                "RES2": 0.8,
            }
        )

        @routine(name="test_routine")
        def simple_routine(ctx: RoutineContext, **kwargs) -> dict:
            return {}

        # Run with control group and zero_other_groups=True
        runner.run("test_routine", __group__="control", zero_other_groups=True)

        # Shared RESERVOIR gates should NOT be zeroed
        assert control_inst.voltages["RES1"] == 0.7
        assert control_inst.voltages["RES2"] == 0.8

        # Non-shared gate from other group should be zeroed
        assert control_inst.voltages["G3"] == 0.0

    def test_zero_other_groups_false_skips_zeroing(
        self, registry_fixture, routine_runner_with_grouped_device
    ):
        """Test that zero_other_groups=False (default) skips zeroing."""
        runner, original_device, control_inst, _ = routine_runner_with_grouped_device

        # Set initial voltages
        original_device.jump({"G1": 0.5, "G3": 0.5})

        @routine(name="test_routine")
        def simple_routine(ctx: RoutineContext, **kwargs) -> dict:
            return {}

        # Run with control group but zero_other_groups=False (default)
        runner.run("test_routine", __group__="control", zero_other_groups=False)

        # G3 should NOT be zeroed
        assert control_inst.voltages["G3"] == 0.5

    def test_zero_other_groups_error_logged_but_routine_continues(
        self, registry_fixture, routine_runner_with_grouped_device, caplog
    ):
        """Test that zeroing errors are logged as warnings but routine continues."""
        runner, original_device, control_inst, _ = routine_runner_with_grouped_device

        # Make control instrument fail on set_voltage
        control_inst.should_fail = True

        routine_executed = False

        @routine(name="test_routine")
        def track_execution_routine(ctx: RoutineContext, **kwargs) -> dict:
            nonlocal routine_executed
            routine_executed = True
            return {}

        # This should log a warning but continue
        runner.run("test_routine", __group__="control", zero_other_groups=True)

        # Routine should still execute despite zeroing failure
        assert routine_executed

        # Check that warning was logged
        assert any(
            "Failed to zero other group gates" in record.message
            for record in caplog.records
        )

    def test_zero_other_groups_respects_conditional_filtering(
        self, registry_fixture
    ):
        """Test that zero_other_groups respects conditional filtering logic.

        When GPIOs/contacts are omitted from group definition, they are accessible
        to all groups (conditional filtering). Therefore, they should NOT be zeroed
        by zero_other_groups.
        """
        from stanza.models import GPIO, GPIOType

        # Create device with gates, contacts, and GPIOs
        device_config = DeviceConfig(
            name="device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
                "G2": make_gate(GateType.BARRIER, control_channel=2),
                "G3": make_gate(GateType.PLUNGER, control_channel=3),
                "G4": make_gate(GateType.BARRIER, control_channel=4),
            },
            contacts={
                "IN": make_contact(ContactType.SOURCE, measure_channel=1),
                "OUT_A": make_contact(ContactType.DRAIN, measure_channel=2),
                "OUT_B": make_contact(ContactType.DRAIN, measure_channel=3),
            },
            gpios={
                "VDD": GPIO(
                    type=GPIOType.INPUT,
                    control_channel=10,
                    v_lower_bound=-3.0,
                    v_upper_bound=3.0,
                ),
                "VSS": GPIO(
                    type=GPIOType.INPUT,
                    control_channel=11,
                    v_lower_bound=-3.0,
                    v_upper_bound=3.0,
                ),
            },
            groups={
                # control group: omits gpios and contacts (conditional filtering)
                "control": DeviceGroup(gates=["G1", "G2"]),
                # sensor group: omits gpios and contacts (conditional filtering)
                "sensor": DeviceGroup(gates=["G3", "G4"]),
            },
            routines=[],
            instruments=standard_instrument_configs(),
        )

        channel_configs = generate_channel_configs(device_config)
        control_inst = MockControlInstrument()
        measure_inst = MockMeasurementInstrument()

        device = Device(
            name="device",
            device_config=device_config,
            channel_configs=channel_configs,
            control_instrument=control_inst,
            measurement_instrument=measure_inst,
        )

        runner = RoutineRunner(resources=[device])

        # Set initial voltages on all pads
        device.jump(
            {
                "G1": 0.5,
                "G2": 0.5,
                "G3": 0.5,
                "G4": 0.5,
                "VDD": 1.5,
                "VSS": -1.5,
            }
        )

        @routine(name="test_routine")
        def simple_routine(ctx: RoutineContext, **kwargs) -> dict:
            return {}

        # Run with control group and zero_other_groups=True
        runner.run("test_routine", __group__="control", zero_other_groups=True)

        # Gates from sensor group should be zeroed
        assert control_inst.voltages["G3"] == 0.0
        assert control_inst.voltages["G4"] == 0.0

        # Gates from control group should NOT be zeroed
        assert control_inst.voltages["G1"] == 0.5
        assert control_inst.voltages["G2"] == 0.5

        # GPIOs should NOT be zeroed (accessible to control group via conditional filtering)
        assert control_inst.voltages["VDD"] == 1.5
        assert control_inst.voltages["VSS"] == -1.5

        # Note: Contacts don't have control channels in this test, so they can't be zeroed


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
        runner.run("test_routine", __group__="control")

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
            runner.run("failing_routine", __group__="control")

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
        runner.run("test_routine", __group__="control")

        # Run with sensor group
        runner.run("test_routine", __group__="sensor")

        # Run without group (should get original device)
        runner.run("test_routine")

        # Verify each routine got the correct device
        assert len(devices_received) == 3

        # First routine: control group
        assert devices_received[0]["name"] == "device_control"
        assert devices_received[0]["gates"] == {"G1", "G2", "RES1", "RES2"}

        # Second routine: sensor group
        assert devices_received[1]["name"] == "device_sensor"
        assert devices_received[1]["gates"] == {"G3", "G4", "RES1", "RES2"}

        # Third routine: original device (no group)
        assert devices_received[2]["name"] == "device"
        assert devices_received[2]["gates"] == {"G1", "G2", "G3", "G4", "RES1", "RES2"}


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

        runner.run("test_routine", __group__="control")

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

            runner.run("test_routine", __group__="control")

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
            runner.run("test_routine", __group__="control")

            # Run for sensor group
            runner.run("test_routine", __group__="sensor")

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
        """Test backward compatibility: routines without groups don't get suffix."""
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
