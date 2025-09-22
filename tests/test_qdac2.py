from unittest.mock import Mock, patch

from stanza.drivers.qdac2 import (
    QDAC2,
    QDAC2ControlChannel,
    QDAC2CurrentRange,
    QDAC2MeasurementChannel,
)
from stanza.instruments.channels import ChannelConfig
from stanza.models import BaseInstrumentConfig, InstrumentType


class TestQDAC2CurrentRange:
    def test_enum_values(self):
        assert QDAC2CurrentRange.LOW == "LOW"
        assert QDAC2CurrentRange.HIGH == "HIGH"
        assert str(QDAC2CurrentRange.LOW) == "LOW"


@patch("stanza.drivers.qdac2.PyVisaDriver")
class TestQDAC2:
    def test_initialization_with_simulation(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        instrument_config = BaseInstrumentConfig(
            name="qdac2_sim",
            type=InstrumentType.GENERAL,
            serial_addr="192.168.1.1",
            port=5025,
        )
        channel_configs = {
            "gate1": ChannelConfig(
                name="gate1", voltage_range=(-2.0, 2.0), control_channel=1
            )
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.HIGH,
            channel_configs=channel_configs,
            is_simulation=True,
        )

        assert qdac.name == "qdac2_sim"
        assert qdac.address == "192.168.1.1"
        assert qdac.port == 5025
        assert qdac.current_range == QDAC2CurrentRange.HIGH
        mock_driver_class.assert_called_once_with("ASRL2::INSTR")

    def test_initialization_with_tcp(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        instrument_config = BaseInstrumentConfig(
            name="qdac2_tcp",
            type=InstrumentType.GENERAL,
            serial_addr="192.168.1.1",
            port=5025,
        )
        channel_configs = {
            "gate1": ChannelConfig(
                name="gate1", voltage_range=(-2.0, 2.0), control_channel=1
            ),
            "sense1": ChannelConfig(
                name="sense1", voltage_range=(-1.0, 1.0), measure_channel=2
            ),
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        assert qdac.name == "qdac2_tcp"
        assert qdac.current_range == QDAC2CurrentRange.LOW
        assert qdac.control_channels == [1]
        assert qdac.measurement_channels == [2]
        mock_driver_class.assert_called_once_with("TCPIP::192.168.1.1::5025::SOCKET")

    def test_set_current_range(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        instrument_config = BaseInstrumentConfig(
            name="qdac2",
            type=InstrumentType.GENERAL,
            serial_addr="192.168.1.1",
            port=5025,
        )
        channel_configs = {}

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        qdac.set_current_range("HIGH")
        assert qdac.current_range == QDAC2CurrentRange.HIGH

        qdac.set_current_range(QDAC2CurrentRange.LOW)
        assert qdac.current_range == QDAC2CurrentRange.LOW

    def test_prepare_measurement(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        instrument_config = BaseInstrumentConfig(
            name="qdac2",
            type=InstrumentType.GENERAL,
            serial_addr="192.168.1.1",
            port=5025,
        )
        channel_configs = {
            "sense1": ChannelConfig(
                name="sense1", voltage_range=(-1.0, 1.0), measure_channel=1
            ),
            "sense2": ChannelConfig(
                name="sense2", voltage_range=(-1.0, 1.0), measure_channel=2
            ),
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.HIGH,
            channel_configs=channel_configs,
        )

        qdac.prepare_measurement()

        expected_calls = [("sens:rang high,(@1,2)",), ("sens:aper 0.001,(@1,2)",)]
        actual_calls = [call.args for call in mock_driver.write.call_args_list]
        assert actual_calls == expected_calls

    def test_idn_property(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver.query.return_value = "QDevil,QDAC-II,12345,1.0.0"
        mock_driver_class.return_value = mock_driver

        instrument_config = BaseInstrumentConfig(
            name="qdac2",
            type=InstrumentType.GENERAL,
            serial_addr="192.168.1.1",
            port=5025,
        )
        channel_configs = {}

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        idn = qdac.idn
        assert idn == "QDevil,QDAC-II,12345,1.0.0"
        mock_driver.query.assert_called_once_with("*IDN?")

        # Test cached property
        idn2 = qdac.idn
        assert idn2 == idn
        # Should still be called only once due to caching
        assert mock_driver.query.call_count == 1

    def test_initialize_channels(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        instrument_config = BaseInstrumentConfig(
            name="qdac2",
            type=InstrumentType.GENERAL,
            serial_addr="192.168.1.1",
            port=5025,
        )
        channel_configs = {
            "gate1": ChannelConfig(
                name="gate1", voltage_range=(-2.0, 2.0), control_channel=1
            ),
            "sense1": ChannelConfig(
                name="sense1", voltage_range=(-1.0, 1.0), measure_channel=2
            ),
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        # Verify channels were initialized
        assert "control_gate1" in qdac.channels
        assert "measure_sense1" in qdac.channels
        assert isinstance(qdac.channels["control_gate1"], QDAC2ControlChannel)
        assert isinstance(qdac.channels["measure_sense1"], QDAC2MeasurementChannel)

    def test_convenience_methods(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver.query.side_effect = ["1.5", "0.01", "0.001"]
        mock_driver_class.return_value = mock_driver

        instrument_config = BaseInstrumentConfig(
            name="qdac2",
            type=InstrumentType.GENERAL,
            serial_addr="192.168.1.1",
            port=5025,
        )
        channel_configs = {
            "gate1": ChannelConfig(
                name="gate1", voltage_range=(-2.0, 2.0), control_channel=1
            ),
            "sense1": ChannelConfig(
                name="sense1", voltage_range=(-1.0, 1.0), measure_channel=2
            ),
        }

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        # Test set_voltage
        qdac.set_voltage("gate1", 1.5)
        mock_driver.write.assert_called_with("sour:volt 1.5,@1")

        # Test get_voltage
        voltage = qdac.get_voltage("gate1")
        assert voltage == 1.5

        # Test get_slew_rate
        slew_rate = qdac.get_slew_rate("gate1")
        assert slew_rate == 0.01

        # Test measure
        current = qdac.measure("sense1")
        assert current == 0.001

    def test_str_method(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver.query.return_value = "QDevil,QDAC-II,12345,1.0.0"
        mock_driver_class.return_value = mock_driver

        instrument_config = BaseInstrumentConfig(
            name="test_qdac",
            type=InstrumentType.GENERAL,
            serial_addr="192.168.1.1",
            port=5025,
        )
        channel_configs = {}

        qdac = QDAC2(
            instrument_config=instrument_config,
            current_range=QDAC2CurrentRange.LOW,
            channel_configs=channel_configs,
        )

        str_repr = str(qdac)
        expected = "QDAC2(name=test_qdac, address=192.168.1.1, port=5025, idn=QDevil,QDAC-II,12345,1.0.0)"
        assert str_repr == expected


class TestQDAC2ControlChannel:
    @patch("stanza.drivers.qdac2.PyVisaDriver")
    def test_initialization(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        config = ChannelConfig(
            name="gate1", voltage_range=(-2.0, 2.0), control_channel=1
        )

        channel = QDAC2ControlChannel("gate1", 1, config, mock_driver)

        assert channel.name == "gate1"
        assert channel.channel_id == 1
        assert channel.driver == mock_driver

    @patch("stanza.drivers.qdac2.PyVisaDriver")
    def test_parameter_setup(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver.query.side_effect = ["1.5", "0.01"]
        mock_driver_class.return_value = mock_driver

        config = ChannelConfig(
            name="gate1", voltage_range=(-2.0, 2.0), control_channel=1
        )
        # Add slew_rate to config manually to test the feature
        config.slew_rate = 0.005

        channel = QDAC2ControlChannel("gate1", 1, config, mock_driver)

        # Test voltage parameter
        voltage_param = channel.get_parameter("voltage")
        voltage_param.setter(1.5)
        mock_driver.write.assert_called_with("sour:volt 1.5,@1")

        voltage = voltage_param.getter()
        assert voltage == 1.5

        # Test slew rate parameter
        slew_rate_param = channel.get_parameter("slew_rate")
        slew_rate_param.setter(0.01)
        mock_driver.write.assert_called_with("sour1:volt:slew 0.01")

        slew_rate = slew_rate_param.getter()
        assert slew_rate == 0.01

    @patch("stanza.drivers.qdac2.PyVisaDriver")
    def test_parameter_setup_with_exception(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver.write.side_effect = Exception("Communication error")
        mock_driver_class.return_value = mock_driver

        config = ChannelConfig(
            name="gate1", voltage_range=(-2.0, 2.0), control_channel=1
        )
        # Add slew_rate to config manually to test the exception handling
        config.slew_rate = 0.005

        # This should not raise an exception due to the try/except block
        channel = QDAC2ControlChannel("gate1", 1, config, mock_driver)

        # Verify the channel was still created
        assert channel.name == "gate1"
        assert channel.channel_id == 1


class TestQDAC2MeasurementChannel:
    @patch("stanza.drivers.qdac2.PyVisaDriver")
    def test_initialization(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver_class.return_value = mock_driver

        config = ChannelConfig(
            name="sense1", voltage_range=(-1.0, 1.0), measure_channel=2
        )

        channel = QDAC2MeasurementChannel("sense1", 2, config, mock_driver)

        assert channel.name == "sense1"
        assert channel.channel_id == 2
        assert channel.driver == mock_driver

    @patch("stanza.drivers.qdac2.PyVisaDriver")
    def test_parameter_setup(self, mock_driver_class):
        mock_driver = Mock()
        mock_driver.query.return_value = "0.001"
        mock_driver_class.return_value = mock_driver

        config = ChannelConfig(
            name="sense1", voltage_range=(-1.0, 1.0), measure_channel=2
        )

        channel = QDAC2MeasurementChannel("sense1", 2, config, mock_driver)

        # Test current parameter
        current_param = channel.get_parameter("current")
        current = current_param.getter()
        assert current == 0.001
        mock_driver.query.assert_called_with("READ2?")

        # Verify setter is None (read-only parameter)
        assert current_param.setter is None
