import pytest

from stanza.base.instruments import BaseInstrument
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
