import pytest

from operator_.models import (
    Contact,
    ContactType,
    ControlInstrumentConfig,
    DeviceConfig,
    Electrode,
    ExperimentConfig,
    Gate,
    GateType,
    InstrumentType,
    MeasurementInstrumentConfig,
)


def test_electrode_readout_requires_measure_channel():
    """Test that readout=True requires measure_channel to be specified."""
    with pytest.raises(ValueError, match="`measure_channel` must be specified when readout=True"):
        Electrode(readout=True, measure_channel=None, v_lower_bound=0.0, v_upper_bound=1.0)


def test_electrode_control_requires_measure_channel():
    """Test that readout=False requires control_channel to be specified."""
    with pytest.raises(ValueError, match="`control_channel` must be specified when readout=False"):
        Electrode(readout=False, control_channel=None, v_lower_bound=0.0, v_upper_bound=1.0)


def test_electrode_control_channel_requires_voltage_bounds():
    """Test that control_channel requires v_lower_bound and v_upper_bound."""
    with pytest.raises(ValueError, match="`v_lower_bound` must be specified when control_channel is set"):
        Electrode(readout=False, control_channel=1, v_lower_bound=None, v_upper_bound=1.0)

    with pytest.raises(ValueError, match="`v_upper_bound` must be specified when control_channel is set"):
        Electrode(readout=False, control_channel=1, v_lower_bound=0.0, v_upper_bound=None)


def test_base_instrument_config_communication_validation():
    """Test that either ip_addr or serial_addr must be provided."""
    with pytest.raises(ValueError, match="Either 'ip_addr' or 'serial_addr' must be provided"):
        ControlInstrumentConfig(
            name="test",
            type=InstrumentType.CONTROL,
            slew_rate=1.0,
            ip_addr=None,
            serial_addr=None
        )


def test_measurement_instrument_timing_validation():
    """Test that sample_time cannot be larger than measurement_duration."""
    with pytest.raises(ValueError, match="sample_time .* cannot be larger than measurement_duration"):
        MeasurementInstrumentConfig(
            name="test",
            type=InstrumentType.MEASUREMENT,
            ip_addr="192.168.1.1",
            measurement_duration=1.0,
            sample_time=2.0
        )


def test_device_config_unique_channels():
    """Test that duplicate channels raise ValueError."""
    gate1 = Gate(type=GateType.PLUNGER, readout=False, control_channel=1, v_lower_bound=0.0, v_upper_bound=1.0)
    gate2 = Gate(type=GateType.BARRIER, readout=False, control_channel=1, v_lower_bound=0.0, v_upper_bound=1.0)

    control_instrument = ControlInstrumentConfig(
        name="control", type=InstrumentType.CONTROL, ip_addr="192.168.1.1", slew_rate=1.0
    )
    measurement_instrument = MeasurementInstrumentConfig(
        name="measurement", type=InstrumentType.MEASUREMENT, ip_addr="192.168.1.2",
        measurement_duration=1.0, sample_time=0.5
    )

    with pytest.raises(ValueError, match="Duplicate channels found: gate 'gate1' control_channel 1, gate 'gate2' control_channel 1"):
        DeviceConfig(
            name="test_device",
            gates={"gate1": gate1, "gate2": gate2},
            contacts={},
            experiments=[ExperimentConfig(name="test_exp")],
            instruments=[control_instrument, measurement_instrument]
        )


def test_device_config_required_instruments():
    """Test that at least one control and one measurement instrument are required."""
    gate = Gate(type=GateType.PLUNGER, readout=True, measure_channel=1, v_lower_bound=0.0, v_upper_bound=1.0)

    # Missing measurement instrument
    with pytest.raises(ValueError, match="At least one measurement instrument is required"):
        DeviceConfig(
            name="test_device",
            gates={"gate1": gate},
            contacts={},
            experiments=[ExperimentConfig(name="test_exp")],
            instruments=[ControlInstrumentConfig(
                name="control", type=InstrumentType.CONTROL, ip_addr="192.168.1.1", slew_rate=1.0
            )]
        )

    # Missing control instrument
    with pytest.raises(ValueError, match="At least one control instrument is required"):
        DeviceConfig(
            name="test_device",
            gates={"gate1": gate},
            contacts={},
            experiments=[ExperimentConfig(name="test_exp")],
            instruments=[MeasurementInstrumentConfig(
                name="measurement", type=InstrumentType.MEASUREMENT, ip_addr="192.168.1.2",
                measurement_duration=1.0, sample_time=0.5
            )]
        )


def test_valid_device_config():
    """Test that valid configuration passes validation."""
    gate = Gate(type=GateType.PLUNGER, readout=True, measure_channel=1, v_lower_bound=0.0, v_upper_bound=1.0)
    contact = Contact(type=ContactType.SOURCE, readout=True, measure_channel=2, v_lower_bound=0.0, v_upper_bound=1.0)

    control_instrument = ControlInstrumentConfig(
        name="control", type=InstrumentType.CONTROL, ip_addr="192.168.1.1", slew_rate=1.0
    )
    measurement_instrument = MeasurementInstrumentConfig(
        name="measurement", type=InstrumentType.MEASUREMENT, serial_addr="/dev/ttyUSB0",
        measurement_duration=1.0, sample_time=0.5
    )

    device = DeviceConfig(
        name="test_device",
        gates={"gate1": gate},
        contacts={"contact1": contact},
        experiments=[ExperimentConfig(name="test_exp")],
        instruments=[control_instrument, measurement_instrument]
    )

    assert device.name == "test_device"
    assert len(device.gates) == 1
    assert len(device.contacts) == 1
    assert len(device.instruments) == 2