import importlib

from stanza.base.mixins import InstrumentChannelMixin
from stanza.base.protocols import (
    BreakoutBoxInstrument,
    ControlInstrument,
    MeasurementInstrument,
)
from stanza.models import InstrumentType


def load_driver_class(driver_name: str) -> type:
    module = importlib.import_module(f"stanza.drivers.{driver_name.lower()}")

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, InstrumentChannelMixin)
            and attr.__module__ == module.__name__
        ):
            return attr

    raise ImportError(f"No instrument class found in stanza.drivers.{driver_name}")


def validate_driver_protocols(
    driver_class: type, instrument_type: InstrumentType
) -> None:
    dummy: object = object.__new__(driver_class)

    match instrument_type:
        case InstrumentType.CONTROL:
            if not isinstance(dummy, ControlInstrument):
                raise TypeError(
                    f"{driver_class.__name__} must implement ControlInstrument"
                )
        case InstrumentType.MEASUREMENT:
            if not isinstance(dummy, MeasurementInstrument):
                raise TypeError(
                    f"{driver_class.__name__} must implement MeasurementInstrument"
                )
        case InstrumentType.GENERAL:
            if not isinstance(dummy, ControlInstrument):
                raise TypeError(
                    f"{driver_class.__name__} must implement ControlInstrument"
                )
            if not isinstance(dummy, MeasurementInstrument):
                raise TypeError(
                    f"{driver_class.__name__} must implement MeasurementInstrument"
                )
        case InstrumentType.BREAKOUT_BOX:
            if not isinstance(dummy, BreakoutBoxInstrument):
                raise TypeError(
                    f"{driver_class.__name__} must implement BreakoutBoxInstrument"
                )
