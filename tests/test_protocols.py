from stanza.instruments.protocols import Controlinstrument, MeasurementInstrument


class MockControlInstrument:
    """Mock implementation of Controlinstrument protocol."""

    def __init__(self):
        self.voltages = {}
        self.slew_rates = {}

    def set_voltage(self, channel_name: str, voltage: float) -> None:
        self.voltages[channel_name] = voltage

    def get_voltage(self, channel_name: str) -> float:
        return self.voltages.get(channel_name, 0.0)

    def get_slew_rate(self, channel_name: str) -> float:
        return self.slew_rates.get(channel_name, 1.0)


class MockMeasurementInstrument:
    """Mock implementation of MeasurementInstrument protocol."""

    def __init__(self):
        self.measurements = {}

    def measure(self, channel_name: str) -> float:
        return self.measurements.get(channel_name, 0.0)


class TestControlInstrumentProtocol:
    def test_protocol_implementation(self):
        mock_instrument = MockControlInstrument()

        assert isinstance(mock_instrument, Controlinstrument)

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
