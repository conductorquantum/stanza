import importlib.resources
from importlib.resources import as_file

from stanza.utils import device_from_yaml


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
