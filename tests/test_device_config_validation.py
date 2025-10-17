import pytest

from stanza.models import (
    GPIO,
    Contact,
    ContactType,
    ControlInstrumentConfig,
    DeviceConfig,
    Electrode,
    Gate,
    GateType,
    GPIOType,
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

    gpio1 = GPIO(
        type=GPIOType.OUTPUT,
        control_channel=2,
        v_lower_bound=0.0,
        v_upper_bound=3.3,
    )
    gpio2 = GPIO(
        type=GPIOType.INPUT,
        control_channel=2,
        v_lower_bound=0.0,
        v_upper_bound=3.3,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate channels found: gpio 'gpio1' control_channel 2, gpio 'gpio2' control_channel 2",
    ):
        DeviceConfig(
            name="test_device",
            gates={},
            contacts={},
            gpios={"gpio1": gpio1, "gpio2": gpio2},
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
    gpio = GPIO(
        type=GPIOType.OUTPUT,
        control_channel=3,
        v_lower_bound=0.0,
        v_upper_bound=3.3,
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
        gpios={"gpio1": gpio},
        routines=[RoutineConfig(name="test_exp")],
        instruments=[control_instrument, measurement_instrument],
    )

    assert device.name == "test_device"
    assert len(device.gates) == 1
    assert len(device.contacts) == 1
    assert len(device.gpios) == 1
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


def test_routine_config_with_no_parameters():
    """Test that RoutineConfig works with no parameters."""
    routine = RoutineConfig(name="test_routine")
    assert routine.parameters is None


def test_routine_config_with_nested_routines():
    """Test that nested routines work correctly."""
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

    assert routine.parameters["value"] == 42.0
    assert routine.routines[0].parameters["frequency"] == 1e9


def test_routine_config_converts_string_integers_to_int():
    """Test that string values representing integers are converted to int."""
    routine = RoutineConfig(
        name="test_routine",
        parameters={
            "count": "100",
            "frequency": "50000000",
        },
    )

    assert routine.parameters["count"] == 100
    assert isinstance(routine.parameters["count"], int)
    assert routine.parameters["frequency"] == 50000000
    assert isinstance(routine.parameters["frequency"], int)


def test_routine_config_converts_string_floats_to_float():
    """Test that string values representing floats are converted to float."""
    routine = RoutineConfig(
        name="test_routine",
        parameters={
            "amplitude": "0.5",
            "threshold": "1.23456",
        },
    )

    assert routine.parameters["amplitude"] == 0.5
    assert isinstance(routine.parameters["amplitude"], float)
    assert routine.parameters["threshold"] == 1.23456
    assert isinstance(routine.parameters["threshold"], float)


def test_routine_config_converts_string_float_integers_to_int():
    """Test that string values like '100.0' are converted to int."""
    routine = RoutineConfig(
        name="test_routine",
        parameters={
            "count": "100.0",
            "iterations": "42.0",
        },
    )

    assert routine.parameters["count"] == 100
    assert isinstance(routine.parameters["count"], int)
    assert routine.parameters["iterations"] == 42
    assert isinstance(routine.parameters["iterations"], int)


def test_routine_config_preserves_non_numeric_strings():
    """Test that non-numeric strings are left as-is."""
    routine = RoutineConfig(
        name="test_routine",
        parameters={
            "name": "test_name",
            "mode": "fast",
            "invalid_number": "not_a_number",
        },
    )

    assert routine.parameters["name"] == "test_name"
    assert isinstance(routine.parameters["name"], str)
    assert routine.parameters["mode"] == "fast"
    assert isinstance(routine.parameters["mode"], str)
    assert routine.parameters["invalid_number"] == "not_a_number"
    assert isinstance(routine.parameters["invalid_number"], str)


def test_routine_config_preserves_native_types():
    """Test that native types (bool, int, float) are preserved."""
    routine = RoutineConfig(
        name="test_routine",
        parameters={
            "enabled": True,
            "count": 100,
            "amplitude": 0.5,
            "frequency": 50e6,
        },
    )

    assert routine.parameters["enabled"] is True
    assert isinstance(routine.parameters["enabled"], bool)
    assert routine.parameters["count"] == 100
    assert isinstance(routine.parameters["count"], int)
    assert routine.parameters["amplitude"] == 0.5
    assert isinstance(routine.parameters["amplitude"], float)
    # Note: floats are NOT automatically converted to int
    assert routine.parameters["frequency"] == 50000000.0
    assert isinstance(routine.parameters["frequency"], float)


def test_routine_config_handles_scientific_notation_strings():
    """Test that scientific notation strings are converted properly."""
    routine = RoutineConfig(
        name="test_routine",
        parameters={
            "frequency": "50e6",
            "small_value": "1e-9",
            "integer_scientific": "1e3",
        },
    )

    # "50e6" becomes 50000000.0 (float), which has no fractional part, so it's converted to int
    assert routine.parameters["frequency"] == 50000000
    assert isinstance(routine.parameters["frequency"], int)
    # "1e-9" has fractional part, so it stays as float
    assert routine.parameters["small_value"] == 1e-9
    assert isinstance(routine.parameters["small_value"], float)
    # "1e3" becomes 1000.0, which has no fractional part, so it's converted to int
    assert routine.parameters["integer_scientific"] == 1000
    assert isinstance(routine.parameters["integer_scientific"], int)
