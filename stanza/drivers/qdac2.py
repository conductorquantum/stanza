from __future__ import annotations

import logging
from enum import Enum
from functools import cached_property
from typing import overload

from stanza.base.channels import (
    ChannelConfig,
    ControlChannel,
    MeasurementChannel,
    Parameter,
)
from stanza.base.instruments import BaseInstrument
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
        """Setup QDAC2-specific control parameters with hardware integration."""
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

        # Set default slew rate if available
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
            self.driver.query(f"read? (@{self.channel_id})")
        )
        current_param.setter = None

        current_range_param = Parameter(
            name="current_range",
            value=None,
            unit="",
            getter=lambda: str(self.driver.query(f"sens:rang? (@{self.channel_id})")),
            setter=lambda r: self.driver.write(f"sens:rang {r},(@{self.channel_id})"),
        )
        self.add_parameter(current_range_param)

        measurement_aperature_s = Parameter(
            name="measurement_aperature_s",
            value=None,
            unit="s",
            getter=lambda: float(self.driver.query(f"sens:aper? (@{self.channel_id})")),
            setter=lambda a: self.driver.write(f"sens:aper {a},(@{self.channel_id})"),
        )
        self.add_parameter(measurement_aperature_s)

        measurement_nplc = Parameter(
            name="measurement_nplc",
            value=None,
            unit="cycles",
            getter=lambda: float(self.driver.query(f"sens:nplc? (@{self.channel_id})")),
            setter=lambda n: self.driver.write(f"sens:nplc {n},(@{self.channel_id})"),
        )
        self.add_parameter(measurement_nplc)


class QDAC2(BaseInstrument):
    def __init__(
        self,
        instrument_config: BaseInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        current_range: QDAC2CurrentRange = QDAC2CurrentRange.LOW,
        is_simulation: bool = False,
        sim_file: str | None = None,
    ):
        self.name = instrument_config.name
        self.address = instrument_config.ip_addr or instrument_config.serial_addr
        self.port = instrument_config.port
        self.measurement_aperature_s = getattr(instrument_config, "aperature_s", None)
        self.measurement_nplc = getattr(instrument_config, "nplc", None)
        self.current_range = QDAC2CurrentRange(current_range)

        self.control_channels = [
            (cfg.name, cfg.control_channel)
            for cfg in channel_configs.values()
            if cfg.control_channel is not None
        ]
        self.measurement_channels = [
            (cfg.name, cfg.measure_channel)
            for cfg in channel_configs.values()
            if cfg.measure_channel is not None
        ]

        if is_simulation:
            visa_addr = "ASRL2::INSTR"
            logger.info("Using simulation mode for QDAC2")
        else:
            visa_addr = f"TCPIP::{self.address}::{self.port}::SOCKET"

        self.driver = PyVisaDriver(visa_addr, sim_file=sim_file)
        self.channel_configs = channel_configs
        super().__init__(instrument_config)
        self._initialize_channels(channel_configs)

        # Set current measurement range
        channels_str = ",".join(str(ch) for _, ch in self.measurement_channels)
        self.driver.write(
            f"sens:rang {str(self.current_range).lower()},(@{channels_str})"
        )
        # Set the integration time
        if self.measurement_aperature_s:
            self.set_measurement_aperatures_s(self.measurement_aperature_s)
        if self.measurement_nplc:
            self.set_all_nplc_cycles(self.measurement_nplc)

    def _initialize_channels(self, channel_configs: dict[str, ChannelConfig]) -> None:
        for channel_config in channel_configs.values():
            if (
                channel_config.control_channel is not None
                and (channel_config.name, channel_config.control_channel)
                in self.control_channels
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
                and (channel_config.name, channel_config.measure_channel)
                in self.measurement_channels
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

    def get_current_range(self, channel_name: str) -> str:
        """Get the current range on a specific channel."""
        return str(
            self.get_channel(f"measure_{channel_name}").get_parameter_value(
                "current_range"
            )
        )

    def set_current_range(
        self, channel_name: str, current_range: str | QDAC2CurrentRange
    ) -> None:
        """Set the current range on a specific channel."""
        self.get_channel(f"measure_{channel_name}").set_parameter(
            "current_range", current_range
        )

    def set_current_ranges(self, current_range: str | QDAC2CurrentRange) -> None:
        """Set the current range on all measurement channels."""
        for ch_name, _ in self.measurement_channels:
            self.get_channel(f"measure_{ch_name}").set_parameter(
                "current_range", current_range
            )

    def get_measurement_aperature_s(self, channel_name: str) -> float:
        """Get the measurement aperature on a specific channel."""
        value = self.get_channel(f"measure_{channel_name}").get_parameter_value(
            "measurement_aperature_s"
        )
        return float(value)

    def set_measurement_aperature_s(
        self, channel_name: str, measurement_aperature_s: float
    ) -> None:
        """Set the measurement aperature on a specific channel."""
        self.get_channel(f"measure_{channel_name}").set_parameter(
            "measurement_aperature_s", measurement_aperature_s
        )

    def set_measurement_aperatures_s(self, measurement_aperature_s: float) -> None:
        """Set the measurement aperature on all measurement channels."""
        for ch_name, _ in self.measurement_channels:
            self.get_channel(f"measure_{ch_name}").set_parameter(
                "measurement_aperature_s", measurement_aperature_s
            )

    def get_nplc_cycles(self, channel_name: str) -> float:
        """Get the number of cycles on a specific channel."""
        value = self.get_channel(f"measure_{channel_name}").get_parameter_value(
            "measurement_nplc"
        )
        return float(value)

    def set_nplc_cycles(self, channel_name: str, nplc_cycles: float) -> None:
        """Set the number of cycles on a specific channel."""
        self.get_channel(f"measure_{channel_name}").set_parameter(
            "measurement_nplc", nplc_cycles
        )

    def set_all_nplc_cycles(self, nplc_cycles: float) -> None:
        """Set the number of cycles on all measurement channels."""
        for ch_name, _ in self.measurement_channels:
            self.get_channel(f"measure_{ch_name}").set_parameter(
                "measurement_nplc", nplc_cycles
            )

    @overload
    def measure(self, channel_name: str) -> float: ...

    @overload
    def measure(self, channel_name: list[str]) -> list[float]: ...

    def measure(self, channel_name: str | list[str]) -> float | list[float]:
        """Measure the current on a specific channel."""
        if isinstance(channel_name, str):
            return super().measure(f"measure_{channel_name}")
        else:
            channel_numbers = [
                self.channel_configs[ch].measure_channel for ch in channel_name
            ]
            channel_str = ",".join(str(ch) for ch in channel_numbers)
            channels_suffix = f",(@{channel_str})"

            currents_str = self.driver.query(f"read? {channels_suffix}")
            currents = [float(current.strip()) for current in currents_str.split(",")]
            return currents

    def close(self) -> None:
        """Close the QDAC2 driver."""
        self.driver.close()

    @cached_property
    def idn(self) -> str:
        return self.driver.query("*IDN?")

    def __str__(self) -> str:
        return f"QDAC2(name={self.name}, address={self.address}, port={self.port}, idn={self.idn})"
