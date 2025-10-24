import pytest

from stanza.base.instruments import BaseControlInstrument, BaseInstrument
from stanza.base.mixins import InstrumentChannelMixin
from stanza.base.registry import load_driver_class, validate_driver_protocols
from stanza.models import InstrumentType


def test_load_driver_class_qdac2():
    driver = load_driver_class("qdac2")
    assert driver.__name__ == "QDAC2"
    assert issubclass(driver, BaseInstrument)


def test_load_driver_class_opx():
    driver = load_driver_class("opx")
    assert driver.__name__ == "OPXInstrument"


def test_load_driver_class_invalid():
    with pytest.raises(ImportError):
        load_driver_class("nonexistent")


def test_validate_driver_protocols_qdac2_general():
    driver = load_driver_class("qdac2")
    validate_driver_protocols(driver, InstrumentType.GENERAL)


def test_validate_driver_protocols_qdac2_control():
    driver = load_driver_class("qdac2")
    validate_driver_protocols(driver, InstrumentType.CONTROL)


def test_validate_driver_protocols_qdac2_measurement():
    driver = load_driver_class("qdac2")
    validate_driver_protocols(driver, InstrumentType.MEASUREMENT)


def test_load_driver_class_no_instrument_found():
    with pytest.raises(ImportError, match="No instrument class found"):
        load_driver_class("utils")


def test_load_driver_class_qswitch():
    driver = load_driver_class("qswitch")
    assert driver.__name__ == "QSwitch"
    assert issubclass(driver, BaseControlInstrument)


def test_validate_driver_protocols_qswitch():
    driver = load_driver_class("qswitch")
    validate_driver_protocols(driver, InstrumentType.BREAKOUT_BOX)


def test_validate_driver_protocols_control_invalid():
    class InvalidControlInstrument(InstrumentChannelMixin):
        pass

    with pytest.raises(TypeError, match="must implement ControlInstrument"):
        validate_driver_protocols(InvalidControlInstrument, InstrumentType.CONTROL)


def test_validate_driver_protocols_measurement_invalid():
    class InvalidMeasurementInstrument(InstrumentChannelMixin):
        pass

    with pytest.raises(TypeError, match="must implement MeasurementInstrument"):
        validate_driver_protocols(
            InvalidMeasurementInstrument, InstrumentType.MEASUREMENT
        )


def test_validate_driver_protocols_general_missing_control():
    class InvalidGeneralInstrument(InstrumentChannelMixin):
        def measure(self, channel_name: str) -> float:
            return 0.0

    with pytest.raises(TypeError, match="must implement ControlInstrument"):
        validate_driver_protocols(InvalidGeneralInstrument, InstrumentType.GENERAL)


def test_validate_driver_protocols_general_missing_measurement():
    class InvalidGeneralInstrument(InstrumentChannelMixin):
        def set_voltage(self, channel_name: str, voltage: float) -> None:
            pass

        def get_voltage(self, channel_name: str) -> float:
            return 0.0

        def get_slew_rate(self, channel_name: str) -> float:
            return 1.0

    with pytest.raises(TypeError, match="must implement MeasurementInstrument"):
        validate_driver_protocols(InvalidGeneralInstrument, InstrumentType.GENERAL)


def test_validate_driver_protocols_breakout_box_invalid():
    class InvalidBreakoutBoxInstrument(InstrumentChannelMixin):
        pass

    with pytest.raises(TypeError, match="must implement BreakoutBoxInstrument"):
        validate_driver_protocols(
            InvalidBreakoutBoxInstrument, InstrumentType.BREAKOUT_BOX
        )
