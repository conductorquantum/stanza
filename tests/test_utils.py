import pytest

from stanza.models import (
    Contact,
    ContactType,
    ControlInstrumentConfig,
    DeviceConfig,
    Gate,
    GateType,
    InstrumentType,
    MeasurementInstrumentConfig,
    PadType,
)
from stanza.utils import (
    device_from_config,
    generate_channel_configs,
    load_device_config,
)


class TestLoadDeviceConfig:
    def test_loads_sample_device_config(self):
        """Test that sample device config is loaded correctly."""
        result = load_device_config("devices/device.sample.yaml", is_stanza_config=True)

        assert result.name == "Sample Device"
        assert len(result.gates) == 3
        assert len(result.contacts) == 2
        assert len(result.instruments) == 2
        assert "G1" in result.gates
        assert "IN" in result.contacts
        assert result.gates["G1"].control_channel == 3
        assert result.gates["G1"].measure_channel == 3
        assert result.gates["G1"].breakout_channel == 3
        assert result.contacts["IN"].control_channel == 1
        assert result.contacts["IN"].measure_channel == 1
        assert result.contacts["IN"].breakout_channel == 1

    def test_loads_sample_device__with_groups_config(self):
        result = load_device_config(
            "devices/device.sample.groups.yaml", is_stanza_config=True
        )

        assert result.name == "Sample Device"
        assert len(result.gates) == 10
        assert len(result.contacts) == 2
        assert len(result.gpios) == 3
        assert len(result.instruments) == 2
        assert "G1" in result.gates
        assert "IN" in result.contacts
        assert result.gates["G1"].control_channel == 3
        assert result.gates["G1"].measure_channel == 3
        assert result.gates["G1"].breakout_channel == 3
        assert result.contacts["IN"].control_channel == 1
        assert result.contacts["IN"].measure_channel == 1
        assert result.contacts["IN"].breakout_channel == 1
        assert set(result.groups.keys()) == {"control", "sensor"}
        assert result.groups["control"].gates == [
            "G1",
            "G2",
            "G3",
            "G4",
            "G5",
            "G9",
            "G10",
        ]
        assert result.groups["control"].contacts == ["IN", "OUT"]
        assert result.groups["sensor"].contacts == ["IN", "OUT"]
        assert result.groups["control"].gpios == ["MUX1"]
        assert result.groups["sensor"].gpios == ["MUX2", "SENSOR_ENABLE"]

    def test_loads_external_yaml_config(self, valid_device_yaml, tmp_path):
        config_file = tmp_path / "device.yaml"
        config_file.write_text(valid_device_yaml)

        result = load_device_config(str(config_file), is_stanza_config=False)

        assert result.name == "test_device"
        assert "G1" in result.gates
        assert result.gates["G1"].control_channel == 1
        assert "C1" in result.contacts
        assert result.contacts["C1"].measure_channel == 3
        assert "GPIO1" in result.gpios
        assert result.gpios["GPIO1"].control_channel == 4

    def test_raises_error_for_nonexistent_stanza_config(self):
        with pytest.raises(ValueError, match="Failed to load device config"):
            load_device_config("nonexistent/path.yaml", is_stanza_config=True)

    def test_raises_error_for_nonexistent_external_file(self):
        with pytest.raises(ValueError, match="Failed to load device config"):
            load_device_config("/nonexistent/path.yaml", is_stanza_config=False)


class TestGenerateChannelConfigs:
    def test_generates_gpio_channel_configs(self, valid_device_yaml, tmp_path):
        config_file = tmp_path / "device.yaml"
        config_file.write_text(valid_device_yaml)
        device_config = load_device_config(str(config_file), is_stanza_config=False)

        channel_configs = generate_channel_configs(device_config)

        assert "GPIO1" in channel_configs
        assert channel_configs["GPIO1"].pad_type == PadType.GPIO
        assert channel_configs["GPIO1"].output_mode == "digital"
        assert channel_configs["GPIO1"].control_channel == 4


class TestDeviceFromConfig:
    """Test device_from_config utility function with new breakout box support."""

    def test_device_from_config_missing_driver_field(self):
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": Gate(
                    name="G1",
                    type=GateType.PLUNGER,
                    v_lower_bound=-1.0,
                    v_upper_bound=1.0,
                    control_channel=1,
                )
            },
            contacts={
                "C1": Contact(
                    name="C1",
                    type=ContactType.SOURCE,
                    v_lower_bound=-1.0,
                    v_upper_bound=1.0,
                    measure_channel=1,
                )
            },
            instruments=[
                ControlInstrumentConfig(
                    name="ctrl",
                    type=InstrumentType.CONTROL,
                    ip_addr="127.0.0.1",
                    slew_rate=1.0,
                    driver=None,
                ),
                MeasurementInstrumentConfig(
                    name="meas",
                    type=InstrumentType.MEASUREMENT,
                    ip_addr="127.0.0.1",
                    measurement_duration=1.0,
                    sample_time=0.1,
                ),
            ],
        )

        with pytest.raises(ValueError, match="missing driver field"):
            device_from_config(device_config)
