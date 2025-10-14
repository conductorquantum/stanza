from functools import cached_property
from typing import Any

from stanza.base.mixins import ControlInstrumentMixin, MeasurementInstrumentMixin
from stanza.models import (
    BaseInstrumentConfig,
    ControlInstrumentConfig,
    MeasurementInstrumentConfig,
)


class BaseMeasurementInstrument(MeasurementInstrumentMixin):
    """Base class for measurement instruments."""

    def __init__(self, instrument_config: MeasurementInstrumentConfig) -> None:
        super().__init__()
        self.instrument_config = instrument_config
        self.name = instrument_config.name

        self.measurement_duration = instrument_config.measurement_duration
        self.sample_time = instrument_config.sample_time

    @cached_property
    def instrument_info(self) -> dict[str, Any]:
        """Get the instrument information."""
        return {
            "name": self.name,
            "timing": {
                "measurement_duration": self.measurement_duration,
                "sample_time": self.sample_time,
            },
            "channels": {
                channel_name: channel.channel_id
                for channel_name, channel in self.channels.items()
            },
            "instrument_config": self.instrument_config.model_dump(),
        }


class BaseControlInstrument(ControlInstrumentMixin):
    """Base class for control instruments."""

    def __init__(self, instrument_config: ControlInstrumentConfig) -> None:
        super().__init__()
        self.instrument_config = instrument_config
        self.name = instrument_config.name

    @cached_property
    def instrument_info(self) -> dict[str, Any]:
        """Get the instrument information."""
        return {
            "name": self.name,
            "slew_rate": self.instrument_config.slew_rate,
            "channels": {
                channel_name: channel.channel_id
                for channel_name, channel in self.channels.items()
            },
            "instrument_config": self.instrument_config.model_dump(),
        }


class BaseInstrument(MeasurementInstrumentMixin, ControlInstrumentMixin):
    """Base class for instruments."""

    def __init__(self, instrument_config: BaseInstrumentConfig) -> None:
        super().__init__()
        self.instrument_config = instrument_config
        self.name = instrument_config.name

    @cached_property
    def instrument_info(self) -> dict[str, Any]:
        """Get the instrument information."""
        return {
            "name": self.name,
            "channels": {
                channel_name: channel.channel_id
                for channel_name, channel in self.channels.items()
            },
            "instrument_config": self.instrument_config.model_dump(),
        }
