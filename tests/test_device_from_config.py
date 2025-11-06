import importlib.resources
from importlib.resources import as_file

import pytest
import yaml

from stanza.exceptions import DeviceError
from stanza.models import DeviceConfig
from stanza.utils import device_from_config, device_from_yaml


def test_device_from_yaml():
    """Test that device can be loaded from yaml file."""
    with as_file(
        importlib.resources.files("tests.test_qdac2_pyvisa_sim").joinpath(
            "qdac2_pyvisa_sim.yaml"
        )
    ) as sim_file:
        device = device_from_yaml(
            "devices/device.sample.groups.yaml",
            is_stanza_config=True,
            is_simulation=True,
            sim_file=str(sim_file),
        )

        assert device.name == "Sample Device"
        assert device.control_instrument is not None
        assert device.measurement_instrument is not None
        assert len(device.gates) == 10
        assert len(device.contacts) == 2
        assert len(device.gpios) == 3
        assert set(device.group_names()) == {"control", "sensor"}
        assert set(device.group_gates("control")) == {
            "G1",
            "G2",
            "G3",
            "G4",
            "G5",
            "G9",
            "G10",
        }
        assert set(device.group_contacts("control")) == {"IN", "OUT"}
        assert set(device.group_contacts("sensor")) == {"IN", "OUT"}
        assert set(device.group_gpios("control")) == {"MUX1"}
        assert set(device.group_gpios("sensor")) == {"MUX2", "SENSOR_ENABLE"}

        with pytest.raises(DeviceError, match="Group 'unknown' not found"):
            device.group_gates("unknown")


def test_device_from_config_with_qswitch(device_yaml_with_qswitch):
    """Test that qswitch breakout box can be loaded with device_from_config."""
    config = DeviceConfig.model_validate(yaml.safe_load(device_yaml_with_qswitch))

    with as_file(
        importlib.resources.files("tests.test_qdac2_pyvisa_sim").joinpath(
            "qdac2_pyvisa_sim.yaml"
        )
    ) as sim_file:
        device = device_from_config(config, is_simulation=True, sim_file=str(sim_file))

        assert device.breakout_box_instrument is not None
        assert {"G1", "C1"} <= device.breakout_box_instrument.channels.keys()
