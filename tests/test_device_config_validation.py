import pytest

from stanza.models import (
    Contact,
    ContactType,
    ControlInstrumentConfig,
    DeviceConfig,
    Electrode,
    Gate,
    GateType,
    InstrumentType,
    MeasurementInstrumentConfig,
    RoutineConfig,
)


def test_electrode_requires_control_channel_when_no_measure_channel():
    with pytest.raises(
        ValueError,
        match="Either `control_channel` or `measure_channel` must be specified",
    ):
        Electrode(
            control_channel=None,
            measure_channel=None,
            v_lower_bound=0.0,
            v_upper_bound=1.0,
        )


def test_electrode_control_channel_requires_voltage_bounds():
    with pytest.raises(
        ValueError,
        match="`v_lower_bound` must be specified when control_channel is set",
    ):
        Electrode(control_channel=1, v_lower_bound=None, v_upper_bound=1.0)

    with pytest.raises(
        ValueError,
        match="`v_upper_bound` must be specified when control_channel is set",
    ):
        Electrode(control_channel=1, v_lower_bound=0.0, v_upper_bound=None)


def test_base_instrument_config_communication_validation():
    with pytest.raises(
        ValueError, match="Either 'ip_addr' or 'serial_addr' must be provided"
    ):
        ControlInstrumentConfig(
            name="test",
            type=InstrumentType.CONTROL,
            slew_rate=1.0,
            ip_addr=None,
            serial_addr=None,
        )


def test_measurement_instrument_timing_validation():
    with pytest.raises(
        ValueError, match="sample_time .* cannot be larger than measurement_duration"
    ):
        MeasurementInstrumentConfig(
            name="test",
            type=InstrumentType.MEASUREMENT,
            ip_addr="192.168.1.1",
            measurement_duration=1.0,
            sample_time=2.0,
        )


def test_device_config_unique_channels():
    gate1 = Gate(
        type=GateType.PLUNGER,
        control_channel=1,
        v_lower_bound=0.0,
        v_upper_bound=1.0,
    )
    gate2 = Gate(
        type=GateType.BARRIER,
        control_channel=1,
        v_lower_bound=0.0,
        v_upper_bound=1.0,
    )

    control_instrument = ControlInstrumentConfig(
        name="control",
        type=InstrumentType.CONTROL,
        ip_addr="192.168.1.1",
        slew_rate=1.0,
    )
    measurement_instrument = MeasurementInstrumentConfig(
        name="measurement",
        type=InstrumentType.MEASUREMENT,
        ip_addr="192.168.1.2",
        measurement_duration=1.0,
        sample_time=0.5,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate channels found: gate 'gate1' control_channel 1, gate 'gate2' control_channel 1",
    ):
        DeviceConfig(
            name="test_device",
            gates={"gate1": gate1, "gate2": gate2},
            contacts={},
            routines=[RoutineConfig(name="test_exp")],
            instruments=[control_instrument, measurement_instrument],
        )


def test_device_config_required_instruments():
    gate = Gate(
        type=GateType.PLUNGER,
        measure_channel=1,
        v_lower_bound=0.0,
        v_upper_bound=1.0,
    )

    with pytest.raises(
        ValueError, match="At least one MEASUREMENT or GENERAL instrument is required"
    ):
        DeviceConfig(
            name="test_device",
            gates={"gate1": gate},
            contacts={},
            routines=[RoutineConfig(name="test_exp")],
            instruments=[
                ControlInstrumentConfig(
                    name="control",
                    type=InstrumentType.CONTROL,
                    ip_addr="192.168.1.1",
                    slew_rate=1.0,
                )
            ],
        )

    with pytest.raises(
        ValueError, match="At least one CONTROL or GENERAL instrument is required"
    ):
        DeviceConfig(
            name="test_device",
            gates={"gate1": gate},
            contacts={},
            routines=[RoutineConfig(name="test_exp")],
            instruments=[
                MeasurementInstrumentConfig(
                    name="measurement",
                    type=InstrumentType.MEASUREMENT,
                    ip_addr="192.168.1.2",
                    measurement_duration=1.0,
                    sample_time=0.5,
                )
            ],
        )


def test_valid_device_config():
    gate = Gate(
        type=GateType.PLUNGER,
        measure_channel=1,
        v_lower_bound=0.0,
        v_upper_bound=1.0,
    )
    contact = Contact(
        type=ContactType.SOURCE,
        measure_channel=2,
        v_lower_bound=0.0,
        v_upper_bound=1.0,
    )

    control_instrument = ControlInstrumentConfig(
        name="control",
        type=InstrumentType.CONTROL,
        ip_addr="192.168.1.1",
        slew_rate=1.0,
    )
    measurement_instrument = MeasurementInstrumentConfig(
        name="measurement",
        type=InstrumentType.MEASUREMENT,
        serial_addr="/dev/ttyUSB0",
        measurement_duration=1.0,
        sample_time=0.5,
    )

    device = DeviceConfig(
        name="test_device",
        gates={"gate1": gate},
        contacts={"contact1": contact},
        routines=[RoutineConfig(name="test_exp")],
        instruments=[control_instrument, measurement_instrument],
    )

    assert device.name == "test_device"
    assert len(device.gates) == 1
    assert len(device.contacts) == 1
    assert len(device.instruments) == 2


def test_gate_type_str():
    gate_type = GateType.PLUNGER
    assert str(gate_type) == "GateType.PLUNGER"


def test_device_config_duplicate_measure_channels():
    with pytest.raises(ValueError, match="Duplicate channels found"):
        DeviceConfig(
            name="test",
            gates={
                "gate1": Gate(
                    name="gate1",
                    type=GateType.PLUNGER,
                    v_lower_bound=-2.0,
                    v_upper_bound=2.0,
                    control_channel=1,
                    measure_channel=1,
                ),
                "gate2": Gate(
                    name="gate2",
                    type=GateType.PLUNGER,
                    v_lower_bound=-2.0,
                    v_upper_bound=2.0,
                    control_channel=2,
                    measure_channel=1,
                ),
            },
            contacts={},
            routines=[],
            instruments=[],
        )


def test_routine_config_converts_float_to_int():
    """Test that floats with no fractional part are converted to ints."""
    routine = RoutineConfig(
        name="test_routine",
        parameters={
            "frequency": 50e6,
            "amplitude": 0.5,
            "count": 100.0,
            "enabled": True,
            "name": "test",
        },
    )

    assert routine.parameters["frequency"] == 50000000
    assert isinstance(routine.parameters["frequency"], int)

    assert routine.parameters["amplitude"] == 0.5
    assert isinstance(routine.parameters["amplitude"], float)

    assert routine.parameters["count"] == 100
    assert isinstance(routine.parameters["count"], int)

    assert routine.parameters["enabled"] is True
    assert isinstance(routine.parameters["enabled"], bool)

    assert routine.parameters["name"] == "test"
    assert isinstance(routine.parameters["name"], str)


def test_routine_config_with_no_parameters():
    """Test that RoutineConfig works with no parameters."""
    routine = RoutineConfig(name="test_routine")
    assert routine.parameters is None


def test_routine_config_with_nested_routines():
    """Test that nested routines also convert numbers correctly."""
    routine = RoutineConfig(
        name="parent_routine",
        parameters={"value": 42.0},
        routines=[
            RoutineConfig(
                name="child_routine",
                parameters={"frequency": 1e9},
            )
        ],
    )

    assert routine.parameters["value"] == 42
    assert isinstance(routine.parameters["value"], int)

    assert routine.routines[0].parameters["frequency"] == 1000000000
    assert isinstance(routine.routines[0].parameters["frequency"], int)
