import importlib.resources
from importlib.resources import as_file

import pytest

from stanza.base.channels import ChannelConfig
from stanza.drivers.qswitch import QSwitch, QSwitchChannel
from stanza.models import BaseInstrumentConfig, GateType, InstrumentType, PadType


@pytest.fixture
def qswitch_sim():
    instrument_config = BaseInstrumentConfig(
        name="qswitch_sim",
        type=InstrumentType.BREAKOUT_BOX,
        serial_addr="192.168.1.100",
        port=5025,
    )

    channel_configs = {
        "ch1": ChannelConfig(
            name="ch1",
            voltage_range=(0.0, 5.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            breakout_channel=1,
        ),
    }

    with as_file(
        importlib.resources.files(__package__).joinpath("qswitch_pyvisa_sim.yaml")
    ) as sim_file:
        return QSwitch(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            is_simulation=True,
            sim_file=str(sim_file),
        )


class TestQSwitchPyVisaSim:
    def test_idn_query(self, qswitch_sim):
        idn = qswitch_sim.driver.query("*IDN?")
        assert "QDevil,QSwitch" in idn
        assert "QSWSIM1234" in idn

    def test_channel_initialization(self, qswitch_sim):
        assert "ch1" in qswitch_sim.channels
        assert isinstance(qswitch_sim.channels["ch1"], QSwitchChannel)

    def test_parameter_access(self, qswitch_sim):
        channel = qswitch_sim.channels["ch1"]
        assert channel.get_parameter("connect_relay") is not None
        assert channel.get_parameter("disconnect_relay") is not None

    def test_get_grounded(self, qswitch_sim):
        assert qswitch_sim.get_grounded("ch1") is True

    def test_get_ungrounded(self, qswitch_sim):
        assert qswitch_sim.get_ungrounded("ch1") is False

    def test_get_connected(self, qswitch_sim):
        assert qswitch_sim.get_connected("ch1", 0) is True
        assert qswitch_sim.get_connected("ch1", 1) is False
        assert qswitch_sim.get_connected("ch1", 5) is False

    def test_get_disconnected(self, qswitch_sim):
        assert qswitch_sim.get_disconnected("ch1", 0) is False
        assert qswitch_sim.get_disconnected("ch1", 1) is True
        assert qswitch_sim.get_disconnected("ch1", 5) is True

    def test_set_operations(self, qswitch_sim):
        qswitch_sim.set_grounded("ch1")
        qswitch_sim.set_ungrounded("ch1")
        qswitch_sim.set_connected("ch1", 3)
        qswitch_sim.set_disconnected("ch1", 5)

    def test_close(self, qswitch_sim):
        qswitch_sim.close()
