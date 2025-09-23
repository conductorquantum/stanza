from __future__ import annotations

import logging
from enum import Enum
from functools import cached_property

from stanza.instruments.base import BaseInstrument
from stanza.instruments.channels import (
    ChannelConfig,
    ControlChannel,
    MeasurementChannel,
)
from stanza.models import BaseInstrumentConfig
from stanza.pyvisa import PyVisaDriver

logger = logging.getLogger(__name__)


class QDAC2CurrentRange(str, Enum):
    """Current measurment ranges for QDAC2."""

    LOW = "LOW"
    HIGH = "HIGH"

    def __str__(self) -> str:
        return self.value


class QDAC2ControlChannel(ControlChannel):
    def __init__(
        self, name: str, channel_id: int, config: ChannelConfig, driver: PyVisaDriver
    ):
        self.name = name
        self.channel_id = channel_id
        self.driver = driver
        super().__init__(config)

    def _setup_parameters(self) -> None:
        """Setup QDAC2-specific controlparameters with hardware integration."""
        super()._setup_parameters()

        voltage_param = self.get_parameter("voltage")
        voltage_param.setter = lambda v: self.driver.write(
            f"sour{self.channel_id}:volt {v}"
        )
        voltage_param.getter = lambda: float(
            self.driver.query(f"sour{self.channel_id}:volt?")
        )

        slew_rate_param = self.get_parameter("slew_rate")
        slew_rate_param.setter = lambda s: self.driver.write(
            f"sour{self.channel_id}:volt:slew {s}"
        )
        slew_rate_param.getter = lambda: float(
            self.driver.query(f"sour{self.channel_id}:volt:slew?")
        )

        # Set default slew rate if available in config
        try:
            slew_rate = getattr(self.config, "slew_rate", None)
            if slew_rate is not None:
                slew_rate_param.set(slew_rate)
        except Exception as e:
            logger.warning(f"Could not set initial slew rate: {e}")


class QDAC2MeasurementChannel(MeasurementChannel):
    def __init__(
        self, name: str, channel_id: int, config: ChannelConfig, driver: PyVisaDriver
    ):
        self.name = name
        self.channel_id = channel_id
        self.driver = driver
        super().__init__(config)

    def _setup_parameters(self) -> None:
        """Setup QDAC2-specific measurement parameters with hardware integration."""
        super()._setup_parameters()

        current_param = self.get_parameter("current")
        current_param.getter = lambda: float(
            self.driver.query(f"READ{self.channel_id}?")
        )
        current_param.setter = None


class QDAC2(BaseInstrument):
    def __init__(
        self,
        instrument_config: BaseInstrumentConfig,
        current_range: QDAC2CurrentRange,
        channel_configs: dict[str, ChannelConfig],
        is_simulation: bool = False,
        sim_file: str | None = None,
    ):
        self.name = instrument_config.name
        self.address = instrument_config.ip_addr
        self.port = instrument_config.port
        # Extract QDAC2-specific configuration from instrument config or defaults
        self.sample_time = getattr(
            instrument_config, "sample_time", 0.001
        )  # default 1ms
        self.current_range = QDAC2CurrentRange(current_range)

        # Extract channel lists from channel_configs
        self.control_channels = [
            cfg.control_channel
            for cfg in channel_configs.values()
            if cfg.control_channel is not None
        ]
        self.measurement_channels = [
            cfg.measure_channel
            for cfg in channel_configs.values()
            if cfg.measure_channel is not None
        ]

        if is_simulation:
            visa_addr = "ASRL2::INSTR"
            logger.info("Using simulation mode for QDAC2")
        else:
            visa_addr = f"TCPIP::{self.address}::{self.port}::SOCKET"

        self.driver = PyVisaDriver(visa_addr, sim_file=sim_file)
        super().__init__(instrument_config)
        self._initialize_channels(channel_configs)

    def _initialize_channels(self, channel_configs: dict[str, ChannelConfig]) -> None:
        for channel_config in channel_configs.values():
            if (
                channel_config.control_channel is not None
                and channel_config.control_channel in self.control_channels
            ):
                self.add_channel(
                    f"control_{channel_config.name}",
                    QDAC2ControlChannel(
                        channel_config.name,
                        channel_config.control_channel,
                        channel_config,
                        self.driver,
                    ),
                )
            if (
                channel_config.measure_channel is not None
                and channel_config.measure_channel in self.measurement_channels
            ):
                self.add_channel(
                    f"measure_{channel_config.name}",
                    QDAC2MeasurementChannel(
                        channel_config.name,
                        channel_config.measure_channel,
                        channel_config,
                        self.driver,
                    ),
                )

    def set_current_range(self, current_range: str | QDAC2CurrentRange) -> None:
        self.current_range = QDAC2CurrentRange(current_range)

    def prepare_measurement(self) -> None:
        """Prepare the measurement."""
        # Set the current measurment range
        channels_str = ",".join(str(ch) for ch in self.measurement_channels)
        self.driver.write(
            f"sens:rang {str(self.current_range).lower()},(@{channels_str})"
        )
        # Set the integration time
        self.driver.write(f"sens:aper {self.sample_time},(@{channels_str})")

    def set_voltage(self, channel_name: str, voltage: float) -> None:
        """Set the voltage on a specific channel."""
        return super().set_voltage(f"control_{channel_name}", voltage)

    def get_voltage(self, channel_name: str) -> float:
        """Get the voltage on a specific channel."""
        return super().get_voltage(f"control_{channel_name}")

    def set_slew_rate(self, channel_name: str, slew_rate: float) -> None:
        """Set the slew rate on a specific channel."""
        return super().set_slew_rate(f"control_{channel_name}", slew_rate)

    def get_slew_rate(self, channel_name: str) -> float:
        """Get the slew rate on a specific channel."""
        return super().get_slew_rate(f"control_{channel_name}")

    def measure(self, channel_name: str) -> float:
        """Measure the current on a specific channel."""
        return super().measure(f"measure_{channel_name}")

    @cached_property
    def idn(self) -> str:
        return self.driver.query("*IDN?")

    def __str__(self) -> str:
        return f"QDAC2(name={self.name}, address={self.address}, port={self.port}, idn={self.idn})"
