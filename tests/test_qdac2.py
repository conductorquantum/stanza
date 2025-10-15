from unittest.mock import Mock, patch

import pytest

from stanza.base.channels import ChannelConfig
from stanza.drivers.qdac2 import (
    QDAC2,
    QDAC2ControlChannel,
    QDAC2CurrentRange,
    QDAC2MeasurementChannel,
)
from stanza.models import (
    BaseInstrumentConfig,
    ContactType,
    GateType,
    InstrumentType,
    PadType,
)


@pytest.fixture
def instrument_config():
    return BaseInstrumentConfig(
        name="qdac2",
        type=InstrumentType.GENERAL,
        serial_addr="192.168.1.1",
        port=5025,
    )


@pytest.fixture
def control_channel_config():
    return ChannelConfig(
        name="gate1",
        voltage_range=(-2.0, 2.0),
        pad_type=PadType.GATE,
        electrode_type=GateType.PLUNGER,
        control_channel=1,
    )


@pytest.fixture
def measurement_channel_config():
    return ChannelConfig(
        name="sense1",
        voltage_range=(-1.0, 1.0),
        pad_type=PadType.CONTACT,
        electrode_type=ContactType.SOURCE,
        measure_channel=2,
    )


class TestQDAC2CurrentRange:
    def test_enum_values(self):
        """Test QDAC2 current range enum values."""
        assert QDAC2CurrentRange.LOW == "LOW"
        assert QDAC2CurrentRange.HIGH == "HIGH"
        assert str(QDAC2CurrentRange.LOW) == "LOW"


@patch("stanza.drivers.qdac2.PyVisaDriver")
class TestQDAC2:
    def test_initialization_with_simulation(
        self, mock_driver_class, instrument_config, control_channel_config
    ):
        """Test QDAC2 initialization with simulation mode."""
        mock_driver_class.return_value = Mock()
        channel_configs = {"gate1": control_channel_config}

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.HIGH,
            channel_configs=channel_configs,
            is_simulation=True,
        )

        assert qdac.name == "qdac2"
        assert qdac.address == "192.168.1.1"
        assert qdac.port == 5025
        assert qdac.current_range == QDAC2CurrentRange.HIGH
        mock_driver_class.assert_called_once_with("ASRL2::INSTR", sim_file=None)

    def test_initialization_with_tcp(
        self,
        mock_driver_class,
        instrument_config,
        control_channel_config,
        measurement_channel_config,
    ):
        """Test QDAC2 initialization with TCP connection."""
        mock_driver_class.return_value = Mock()
        channel_configs = {
            "gate1": control_channel_config,
            "sense1": measurement_channel_config,
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        assert qdac.name == "qdac2"
        assert qdac.current_range == QDAC2CurrentRange.LOW
        assert qdac.control_channels == [("gate1", 1)]
        assert qdac.measurement_channels == [("sense1", 2)]
        mock_driver_class.assert_called_once_with(
            "TCPIP::192.168.1.1::5025::SOCKET", sim_file=None
        )

    def test_set_and_get_current_range(
        self, mock_driver_class, instrument_config, measurement_channel_config
    ):
        """Test setting and getting current range on measurement channels."""
        mock_driver = Mock()
        mock_driver.query.return_value = "LOW"
        mock_driver_class.return_value = mock_driver

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={"sense1": measurement_channel_config},
        )

        qdac.set_current_range("sense1", "HIGH")
        mock_driver.write.assert_called_with("sens:rang HIGH,(@2)")

        current_range = qdac.get_current_range("sense1")
        assert current_range == "LOW"

    def test_set_and_get_measurement_aperature_s(
        self, mock_driver_class, instrument_config, measurement_channel_config
    ):
        """Test setting and getting measurement aperture in seconds."""
        mock_driver = Mock()
        mock_driver.query.return_value = "0.002"
        mock_driver_class.return_value = mock_driver

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={"sense1": measurement_channel_config},
        )

        qdac.set_measurement_aperature_s("sense1", 0.005)
        mock_driver.write.assert_called_with("sens:aper 0.005,(@2)")

        aperature = qdac.get_measurement_aperature_s("sense1")
        assert aperature == 0.002

    def test_set_and_get_nplc_cycles(
        self, mock_driver_class, instrument_config, measurement_channel_config
    ):
        """Test setting and getting NPLC (number of power line cycles) for measurements."""
        mock_driver = Mock()
        mock_driver.query.return_value = "20"
        mock_driver_class.return_value = mock_driver

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={"sense1": measurement_channel_config},
        )

        qdac.set_nplc_cycles("sense1", 15)
        mock_driver.write.assert_called_with("sens:nplc 15,(@2)")

        nplc = qdac.get_nplc_cycles("sense1")
        assert nplc == 20.0

    def test_prepare_measurement(
        self, mock_driver_class, instrument_config, measurement_channel_config
    ):
        """Test preparing measurement channels with correct current range configuration."""
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver
        channel_configs = {
            "sense1": measurement_channel_config,
            "sense2": ChannelConfig(
                name="sense2",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.DRAIN,
                measure_channel=2,
            ),
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.HIGH,
            channel_configs=channel_configs,
        )

        qdac.prepare_measurement()

        expected_calls = [
            ("sens:rang high,(@2,2)",),
        ]
        actual_calls = [call.args for call in mock_driver.write.call_args_list]
        assert actual_calls == expected_calls

    def test_idn_property(self, mock_driver_class, instrument_config):
        """Test querying and caching of instrument identification string."""
        mock_driver = Mock()
        mock_driver.query.return_value = "QDevil,QDAC-II,12345,1.0.0"
        mock_driver_class.return_value = mock_driver

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={},
        )

        idn = qdac.idn
        assert idn == "QDevil,QDAC-II,12345,1.0.0"
        mock_driver.query.assert_called_once_with("*IDN?")

        idn2 = qdac.idn
        assert idn2 == idn
        assert mock_driver.query.call_count == 1

    def test_initialize_channels(
        self,
        mock_driver_class,
        instrument_config,
        control_channel_config,
        measurement_channel_config,
    ):
        """Test initialization and registration of control and measurement channels."""
        mock_driver_class.return_value = Mock()
        channel_configs = {
            "gate1": control_channel_config,
            "sense1": measurement_channel_config,
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        assert "control_gate1" in qdac.channels
        assert "measure_sense1" in qdac.channels
        assert isinstance(qdac.channels["control_gate1"], QDAC2ControlChannel)
        assert isinstance(qdac.channels["measure_sense1"], QDAC2MeasurementChannel)

    def test_convenience_methods(
        self,
        mock_driver_class,
        instrument_config,
        control_channel_config,
        measurement_channel_config,
    ):
        """Test convenience methods for setting voltage, getting slew rate, and measuring current."""
        mock_driver = Mock()
        mock_driver.query.side_effect = ["1.5", "0.01", "0.001"]
        mock_driver_class.return_value = mock_driver
        channel_configs = {
            "gate1": control_channel_config,
            "sense1": measurement_channel_config,
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        qdac.set_voltage("gate1", 1.5)
        mock_driver.write.assert_called_with("sour1:volt 1.5")

        voltage = qdac.get_voltage("gate1")
        assert voltage == 1.5

        slew_rate = qdac.get_slew_rate("gate1")
        assert slew_rate == 0.01

        current = qdac.measure("sense1")
        assert current == 0.001

    def test_str_method(self, mock_driver_class, instrument_config):
        """Test string representation of QDAC2 instrument."""
        mock_driver = Mock()
        mock_driver.query.return_value = "QDevil,QDAC-II,12345,1.0.0"
        mock_driver_class.return_value = mock_driver

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={},
        )

        str_repr = str(qdac)
        expected = "QDAC2(name=qdac2, address=192.168.1.1, port=5025, idn=QDevil,QDAC-II,12345,1.0.0)"
        assert str_repr == expected

    def test_set_slew_rate(
        self, mock_driver_class, instrument_config, control_channel_config
    ):
        """Test setting slew rate for voltage changes on control channels."""
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={"gate1": control_channel_config},
        )

        result = qdac.set_slew_rate("gate1", 0.5)
        assert result is None
        mock_driver.write.assert_called_with("sour1:volt:slew 0.5")

    def test_set_current_ranges(
        self, mock_driver_class, instrument_config, measurement_channel_config
    ):
        """Test setting current ranges for all measurement channels at once."""
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        channel_config2 = ChannelConfig(
            name="sense2",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.CONTACT,
            electrode_type=ContactType.DRAIN,
            measure_channel=3,
        )

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={
                "sense1": measurement_channel_config,
                "sense2": channel_config2,
            },
        )

        qdac.set_current_ranges("HIGH")
        assert mock_driver.write.call_count >= 2

    def test_set_measurement_aperatures_s(
        self, mock_driver_class, instrument_config, measurement_channel_config
    ):
        """Test setting measurement apertures for all measurement channels at once."""
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        channel_config2 = ChannelConfig(
            name="sense2",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.CONTACT,
            electrode_type=ContactType.DRAIN,
            measure_channel=3,
        )

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={
                "sense1": measurement_channel_config,
                "sense2": channel_config2,
            },
        )

        qdac.set_measurement_aperatures_s(0.002)
        assert mock_driver.write.call_count >= 2

    def test_set_all_nplc_cycles(
        self, mock_driver_class, instrument_config, measurement_channel_config
    ):
        """Test setting NPLC cycles for all measurement channels at once."""
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        channel_config2 = ChannelConfig(
            name="sense2",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.CONTACT,
            electrode_type=ContactType.DRAIN,
            measure_channel=3,
        )

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={
                "sense1": measurement_channel_config,
                "sense2": channel_config2,
            },
        )

        qdac.set_all_nplc_cycles(10)
        assert mock_driver.write.call_count >= 2

    def test_close(self, mock_driver_class, instrument_config):
        """Test closing the instrument connection."""
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={},
        )

        qdac.close()
        mock_driver.close.assert_called_once()

    def test_measure_single_channel(
        self, mock_driver_class, instrument_config, measurement_channel_config
    ):
        """Test measuring current from a single channel."""
        mock_driver = Mock()
        mock_driver.query.return_value = "0.002"
        mock_driver_class.return_value = mock_driver

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs={"sense1": measurement_channel_config},
        )

        current = qdac.measure("sense1")
        assert isinstance(current, float)
        assert current == 0.002

    def test_measure_multiple_channels(self, mock_driver_class, instrument_config):
        """Test measuring current from multiple channels simultaneously."""
        mock_driver = Mock()
        mock_driver.query.return_value = "0.001, 0.002, 0.003"
        mock_driver_class.return_value = mock_driver

        channel_configs = {
            "sense1": ChannelConfig(
                name="sense1",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.SOURCE,
                measure_channel=2,
            ),
            "sense2": ChannelConfig(
                name="sense2",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.DRAIN,
                measure_channel=3,
            ),
            "sense3": ChannelConfig(
                name="sense3",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.SOURCE,
                measure_channel=4,
            ),
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        currents = qdac.measure(["sense1", "sense2", "sense3"])

        mock_driver.query.assert_called_with("read? ,(@2,3,4)")
        assert isinstance(currents, list)
        assert len(currents) == 3
        assert currents == [0.001, 0.002, 0.003]

    def test_measure_two_channels(self, mock_driver_class, instrument_config):
        """Test measuring current from two channels and verifying result accuracy."""
        mock_driver = Mock()
        mock_driver.query.return_value = "0.0015, 0.0025"
        mock_driver_class.return_value = mock_driver

        channel_configs = {
            "sense1": ChannelConfig(
                name="sense1",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.SOURCE,
                measure_channel=5,
            ),
            "sense2": ChannelConfig(
                name="sense2",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.DRAIN,
                measure_channel=6,
            ),
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.HIGH,
            channel_configs=channel_configs,
        )

        currents = qdac.measure(["sense1", "sense2"])

        mock_driver.query.assert_called_with("read? ,(@5,6)")
        assert isinstance(currents, list)
        assert len(currents) == 2
        assert currents == pytest.approx([0.0015, 0.0025])

    def test_measure_channels_with_whitespace(
        self, mock_driver_class, instrument_config
    ):
        """Test parsing measurement results with irregular whitespace in response."""
        mock_driver = Mock()
        mock_driver.query.return_value = "  0.001  ,  0.002  ,0.003"
        mock_driver_class.return_value = mock_driver

        channel_configs = {
            "sense1": ChannelConfig(
                name="sense1",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.SOURCE,
                measure_channel=1,
            ),
            "sense2": ChannelConfig(
                name="sense2",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.DRAIN,
                measure_channel=2,
            ),
            "sense3": ChannelConfig(
                name="sense3",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.SOURCE,
                measure_channel=3,
            ),
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        currents = qdac.measure(["sense1", "sense2", "sense3"])

        assert currents == [0.001, 0.002, 0.003]

    def test_initialization_with_aperature_s(
        self, mock_driver_class, measurement_channel_config
    ):
        """Test initializing QDAC2 with predefined measurement aperture setting."""
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        instrument_config = BaseInstrumentConfig(
            name="qdac2",
            type=InstrumentType.GENERAL,
            serial_addr="192.168.1.1",
            port=5025,
        )
        instrument_config.aperature_s = 0.002

        channel_configs = {
            "sense1": measurement_channel_config,
            "sense2": ChannelConfig(
                name="sense2",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.DRAIN,
                measure_channel=3,
            ),
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.HIGH,
            channel_configs=channel_configs,
        )

        write_calls = [call.args[0] for call in mock_driver.write.call_args_list]
        assert "sens:aper 0.002,(@2)" in write_calls
        assert "sens:aper 0.002,(@3)" in write_calls
        assert qdac.measurement_aperature_s == 0.002

    def test_initialization_with_nplc(
        self, mock_driver_class, measurement_channel_config
    ):
        """Test initializing QDAC2 with predefined NPLC setting."""
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        instrument_config = BaseInstrumentConfig(
            name="qdac2",
            type=InstrumentType.GENERAL,
            serial_addr="192.168.1.1",
            port=5025,
        )
        instrument_config.nplc = 10

        channel_configs = {
            "sense1": measurement_channel_config,
            "sense2": ChannelConfig(
                name="sense2",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.CONTACT,
                electrode_type=ContactType.DRAIN,
                measure_channel=3,
            ),
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        write_calls = [call.args[0] for call in mock_driver.write.call_args_list]
        assert "sens:nplc 10,(@2)" in write_calls
        assert "sens:nplc 10,(@3)" in write_calls
        assert qdac.measurement_nplc == 10


class TestQDAC2ControlChannel:
    @patch("stanza.drivers.qdac2.PyVisaDriver")
    def test_initialization(self, mock_driver_class, control_channel_config):
        """Test initialization of QDAC2ControlChannel with basic properties."""
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        channel = QDAC2ControlChannel("gate1", 1, control_channel_config, mock_driver)

        assert channel.name == "gate1"
        assert channel.channel_id == 1
        assert channel.driver == mock_driver

    @patch("stanza.drivers.qdac2.PyVisaDriver")
    def test_parameter_setup(self, mock_driver_class, control_channel_config):
        """Test parameter setup and getter/setter functionality for control channel."""
        mock_driver = Mock()
        mock_driver.query.side_effect = ["1.5", "0.01"]
        mock_driver_class.return_value = mock_driver
        control_channel_config.slew_rate = 0.005

        channel = QDAC2ControlChannel("gate1", 1, control_channel_config, mock_driver)

        voltage_param = channel.get_parameter("voltage")
        voltage_param.setter(1.5)
        mock_driver.write.assert_called_with("sour1:volt 1.5")

        voltage = voltage_param.getter()
        assert voltage == 1.5

        slew_rate_param = channel.get_parameter("slew_rate")
        slew_rate_param.setter(0.01)
        mock_driver.write.assert_called_with("sour1:volt:slew 0.01")

        slew_rate = slew_rate_param.getter()
        assert slew_rate == 0.01

    @patch("stanza.drivers.qdac2.PyVisaDriver")
    def test_parameter_setup_with_exception(
        self, mock_driver_class, control_channel_config
    ):
        """Test graceful handling of communication errors during parameter setup."""
        mock_driver = Mock()
        mock_driver.write.side_effect = Exception("Communication error")
        mock_driver_class.return_value = mock_driver
        control_channel_config.slew_rate = 0.005

        channel = QDAC2ControlChannel("gate1", 1, control_channel_config, mock_driver)

        assert channel.name == "gate1"
        assert channel.channel_id == 1


class TestQDAC2MeasurementChannel:
    @patch("stanza.drivers.qdac2.PyVisaDriver")
    def test_initialization(self, mock_driver_class, measurement_channel_config):
        """Test initialization of QDAC2MeasurementChannel with basic properties."""
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        channel = QDAC2MeasurementChannel(
            "sense1", 2, measurement_channel_config, mock_driver
        )

        assert channel.name == "sense1"
        assert channel.channel_id == 2
        assert channel.driver == mock_driver

    @patch("stanza.drivers.qdac2.PyVisaDriver")
    def test_parameter_setup(self, mock_driver_class, measurement_channel_config):
        """Test parameter setup and read-only current measurement functionality."""
        mock_driver = Mock()
        mock_driver.query.return_value = "0.001"
        mock_driver_class.return_value = mock_driver

        channel = QDAC2MeasurementChannel(
            "sense1", 2, measurement_channel_config, mock_driver
        )

        current_param = channel.get_parameter("current")
        current = current_param.getter()
        assert current == 0.001
        mock_driver.query.assert_called_with("read? (@2)")

        assert current_param.setter is None
