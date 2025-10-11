import importlib.resources
from importlib.resources import as_file

import pytest
import pyvisa

from stanza.base.channels import ChannelConfig
from stanza.drivers.qdac2 import QDAC2, QDAC2CurrentRange
from stanza.models import (
    BaseInstrumentConfig,
    ContactType,
    GateType,
    InstrumentType,
    PadType,
)


@pytest.fixture
def sim_resource_manager():
    """Create a pyvisa-sim resource manager using the QDAC2 simulation file."""
    with as_file(
        importlib.resources.files(__package__).joinpath("qdac2_pyvisa_sim.yaml")
    ) as sim_file:
        rm = pyvisa.ResourceManager(f"{sim_file}@sim")
        yield rm
        rm.close()


@pytest.fixture
def qdac2_sim():
    """Create a QDAC2 instance using simulation."""
    instrument_config = BaseInstrumentConfig(
        name="qdac2_sim",
        type=InstrumentType.GENERAL,
        serial_addr="192.168.1.1",
        port=5025,
    )

    channel_configs = {
        "gate1": ChannelConfig(
            name="gate1",
            voltage_range=(-10.0, 10.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        ),
        "gate2": ChannelConfig(
            name="gate2",
            voltage_range=(-10.0, 10.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.BARRIER,
            control_channel=2,
        ),
        "sense1": ChannelConfig(
            name="sense1",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.CONTACT,
            electrode_type=ContactType.SOURCE,
            measure_channel=1,
        ),
    }

    with as_file(
        importlib.resources.files(__package__).joinpath("qdac2_pyvisa_sim.yaml")
    ) as sim_file:
        return QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.HIGH,
            channel_configs=channel_configs,
            is_simulation=True,
            sim_file=str(sim_file),
        )


class TestQDAC2PyVisaSim:
    """Test QDAC2 driver using pyvisa-sim simulation."""

    def test_idn_query(self, qdac2_sim):
        """Test IDN query returns simulated device identification."""
        idn = qdac2_sim.idn
        assert "QDevil,QDAC-II" in idn
        assert "QD2SIM1234" in idn
        assert "v0.20.2-42069-g53da8fc7" in idn

    def test_set_and_get_voltage(self, qdac2_sim):
        """Test setting and getting voltage on control channels."""
        # Test gate1
        qdac2_sim.set_voltage("gate1", 2.5)
        voltage = qdac2_sim.get_voltage("gate1")
        assert voltage == pytest.approx(2.5, rel=1e-6)

        # Test gate2
        qdac2_sim.set_voltage("gate2", -1.8)
        voltage = qdac2_sim.get_voltage("gate2")
        assert voltage == pytest.approx(-1.8, rel=1e-6)

    def test_voltage_bounds(self, qdac2_sim):
        """Test voltage setting within bounds defined in simulation."""
        # Test maximum voltage
        qdac2_sim.set_voltage("gate1", 10.0)
        voltage = qdac2_sim.get_voltage("gate1")
        assert voltage == pytest.approx(10.0, rel=1e-6)

        # Test minimum voltage
        qdac2_sim.set_voltage("gate1", -10.0)
        voltage = qdac2_sim.get_voltage("gate1")
        assert voltage == pytest.approx(-10.0, rel=1e-6)

    def test_slew_rate(self, qdac2_sim):
        """Test slew rate parameter access."""
        channel = qdac2_sim.channels["control_gate1"]
        slew_param = channel.get_parameter("slew_rate")

        # Set slew rate
        slew_param.set(100.0)
        slew_rate = slew_param.get()
        assert slew_rate == pytest.approx(100.0, rel=1e-3)

    def test_prepare_measurement(self, qdac2_sim):
        """Test measurement preparation commands."""
        # This should not raise an exception
        qdac2_sim.prepare_measurement()

        # Verify current range was set
        assert qdac2_sim.current_range == QDAC2CurrentRange.HIGH

    def test_current_measurement(self, qdac2_sim):
        """Test current measurement on measurement channels."""
        current = qdac2_sim.measure("sense1")
        assert current == pytest.approx(0.000001, rel=1e-9)

    def test_multiple_channels(self, qdac2_sim):
        """Test operations on multiple channels."""
        # Set voltages on both gates
        qdac2_sim.set_voltage("gate1", 1.0)
        qdac2_sim.set_voltage("gate2", 2.0)

        # Verify both voltages
        voltage1 = qdac2_sim.get_voltage("gate1")
        voltage2 = qdac2_sim.get_voltage("gate2")

        assert voltage1 == pytest.approx(1.0, rel=1e-6)
        assert voltage2 == pytest.approx(2.0, rel=1e-6)

    def test_current_range_setting(self, qdac2_sim):
        """Test current range configuration."""
        qdac2_sim.set_current_range("sense1", "LOW")
        current_range = qdac2_sim.get_current_range("sense1")
        assert current_range == "LOW"

        qdac2_sim.set_current_range("sense1", QDAC2CurrentRange.HIGH)
        current_range = qdac2_sim.get_current_range("sense1")
        assert current_range == "HIGH"

    def test_channel_initialization(self, qdac2_sim):
        """Test that channels are properly initialized."""
        # Check control channels
        assert "control_gate1" in qdac2_sim.channels
        assert "control_gate2" in qdac2_sim.channels

        # Check measurement channels
        assert "measure_sense1" in qdac2_sim.channels

        # Verify channel types
        from stanza.drivers.qdac2 import QDAC2ControlChannel, QDAC2MeasurementChannel

        assert isinstance(qdac2_sim.channels["control_gate1"], QDAC2ControlChannel)
        assert isinstance(qdac2_sim.channels["measure_sense1"], QDAC2MeasurementChannel)

    def test_driver_direct_access(self, qdac2_sim):
        """Test direct driver access for SCPI commands."""
        # Test direct SCPI query
        channel_count = qdac2_sim.driver.query(":CHANnel:COUNt?")
        assert channel_count == "24"

        # Test system information
        model = qdac2_sim.driver.query(":SYSTem:INFOrmation:MODEL?")
        assert model == "QDAC-II"


class TestPyVisaSimIntegration:
    """Test pyvisa-sim integration directly."""

    def test_resource_manager_creation(self, sim_resource_manager):
        """Test that simulation resource manager is created correctly."""
        resources = sim_resource_manager.list_resources()
        assert "ASRL2::INSTR" in resources
