import importlib.resources
from importlib.resources import as_file

import yaml

from stanza.models import DeviceConfig
from stanza.utils import device_from_config, device_from_yaml


def test_device_from_yaml():
    with as_file(
        importlib.resources.files("tests.test_qdac2_pyvisa_sim").joinpath(
            "qdac2_pyvisa_sim.yaml"
        )
    ) as sim_file:
        device = device_from_yaml(
            "devices/device.sample.yaml",
            is_stanza_config=True,
            is_simulation=True,
            sim_file=str(sim_file),
        )

        assert device.name == "Sample Device"
        assert device.control_instrument is not None
        assert device.measurement_instrument is not None
        assert len(device.gates) == 3
        assert len(device.contacts) == 2


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
