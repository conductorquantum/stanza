from stanza.base.protocols import (
    BreakoutBoxInstrument,
    ControlInstrument,
    MeasurementInstrument,
)
from tests.conftest import (
    MockBreakoutBoxInstrument,
    MockControlInstrument,
    MockMeasurementInstrument,
)


class TestControlInstrumentProtocol:
    def test_protocol_implementation(self):
        mock_instrument = MockControlInstrument()

        assert isinstance(mock_instrument, ControlInstrument)

        mock_instrument.set_voltage("gate1", 1.5)
        voltage = mock_instrument.get_voltage("gate1")

        assert voltage == 1.5

    def test_protocol_methods_exist(self):
        mock_instrument = MockControlInstrument()

        assert hasattr(mock_instrument, "set_voltage")
        assert hasattr(mock_instrument, "get_voltage")
        assert hasattr(mock_instrument, "get_slew_rate")


class TestMeasurementInstrumentProtocol:
    def test_protocol_implementation(self):
        mock_instrument = MockMeasurementInstrument()

        assert isinstance(mock_instrument, MeasurementInstrument)

        mock_instrument.measurements["sense1"] = 1e-6
        current = mock_instrument.measure("sense1")

        assert current == 1e-6

    def test_protocol_methods_exist(self):
        mock_instrument = MockMeasurementInstrument()

        assert hasattr(mock_instrument, "measure")


class TestBreakoutBoxInstrumentProtocol:
    def test_protocol_implementation(self):
        mock_instrument = MockBreakoutBoxInstrument()
        assert isinstance(mock_instrument, BreakoutBoxInstrument)

    def test_protocol_methods_exist(self):
        mock_instrument = MockBreakoutBoxInstrument()
        assert hasattr(mock_instrument, "get_grounded")
        assert hasattr(mock_instrument, "set_grounded")
        assert hasattr(mock_instrument, "get_ungrounded")
        assert hasattr(mock_instrument, "set_ungrounded")
        assert hasattr(mock_instrument, "get_connected")
        assert hasattr(mock_instrument, "set_connected")
        assert hasattr(mock_instrument, "get_disconnected")
        assert hasattr(mock_instrument, "set_disconnected")

    def test_grounding_functionality(self):
        mock_instrument = MockBreakoutBoxInstrument()

        assert mock_instrument.get_ungrounded("ch1") is True
        assert mock_instrument.get_grounded("ch1") is False

        mock_instrument.set_grounded("ch1")
        assert mock_instrument.get_grounded("ch1") is True
        assert mock_instrument.get_ungrounded("ch1") is False

        mock_instrument.set_ungrounded("ch1")
        assert mock_instrument.get_ungrounded("ch1") is True
        assert mock_instrument.get_grounded("ch1") is False

    def test_connection_functionality(self):
        mock_instrument = MockBreakoutBoxInstrument()

        assert mock_instrument.get_disconnected("ch1", 5) is True
        assert mock_instrument.get_connected("ch1", 5) is False

        mock_instrument.set_connected("ch1", 5)
        assert mock_instrument.get_connected("ch1", 5) is True
        assert mock_instrument.get_disconnected("ch1", 5) is False
        assert ("ch1", 5) in mock_instrument.connected_calls

        mock_instrument.set_disconnected("ch1", 5)
        assert ("ch1", 5) in mock_instrument.disconnected_calls
