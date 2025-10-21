from __future__ import annotations

import logging

from stanza.base.channels import ChannelConfig, InstrumentChannel, Parameter
from stanza.base.instruments import BaseControlInstrument
from stanza.models import ControlInstrumentConfig
from stanza.pyvisa import PyVisaDriver

logger = logging.getLogger(__name__)


class QSwitchChannel(InstrumentChannel):
    def __init__(
        self, name: str, channel_id: int, config: ChannelConfig, driver: PyVisaDriver
    ):
        self.name = name
        self.driver = driver
        super().__init__(config)
        self.channel_id = channel_id

    def _setup_parameters(self) -> None:
        """Setup parameters after channel_id is properly set."""
        connect_param = Parameter(
            name="connect",
            value=None,
            unit="bool",
            getter=lambda: [
                int(self.driver.query(f"close? (@{self.channel_id}!{i})"))
                for i in range(0, 10)
            ],
            setter=lambda r: self.driver.write(f"close (@{self.channel_id}!{r})"),
        )
        self.add_parameter(connect_param)

        disconnect_param = Parameter(
            name="disconnect",
            value=None,
            unit="bool",
            getter=lambda: [
                int(self.driver.query(f"open? (@{self.channel_id}!{i})"))
                for i in range(0, 10)
            ],
            setter=lambda r: self.driver.write(f"open (@{self.channel_id}!{r})"),
        )
        self.add_parameter(disconnect_param)


class QSwitch(BaseControlInstrument):
    def __init__(
        self,
        instrument_config: ControlInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        is_simulation: bool = False,
        sim_file: str | None = None,
    ):
        self.name = instrument_config.name
        self.address = instrument_config.ip_addr or instrument_config.serial_addr
        self.port = instrument_config.port
        self.channel_configs = channel_configs

        if is_simulation:
            visa_addr = "ASRL2::INSTR"
            logger.info("Using simulation mode for QSwitch")
        else:
            visa_addr = f"TCPIP::{self.address}::{self.port}::SOCKET"

        self.driver = PyVisaDriver(visa_addr, sim_file=sim_file)
        super().__init__(instrument_config)
        self._initialize_channels(channel_configs)

    def _initialize_channels(self, channel_configs: dict[str, ChannelConfig]) -> None:
        for channel_config in channel_configs.values():
            if channel_config.breakout_channel is not None:
                channel = QSwitchChannel(
                    channel_config.name,
                    channel_config.breakout_channel,
                    channel_config,
                    self.driver,
                )
                self.add_channel(channel_config.name, channel)

    def get_grounded(self, channel_name: str) -> bool:
        """Get if the channel is grounded."""
        return bool(self.get_channel(channel_name).get_parameter_value("connect")[0])

    def set_grounded(self, channel_name: str) -> None:
        """Set the channel to grounded."""
        return self.get_channel(channel_name).set_parameter("connect", 0)

    def get_ungrounded(self, channel_name: str) -> bool:
        """Get if the channel is ungrounded."""
        return bool(self.get_channel(channel_name).get_parameter_value("disconnect")[0])

    def set_ungrounded(self, channel_name: str) -> None:
        """Set the channel to ungrounded."""
        return self.get_channel(channel_name).set_parameter("disconnect", 0)

    def get_connected(self, channel_name: str, line_number: int) -> bool:
        """Get if the channel is connected to the line number."""
        connected_lines = self.get_channel(channel_name).get_parameter_value("connect")
        if line_number < 0 or line_number >= len(connected_lines):
            raise ValueError(f"Line number {line_number} is out of range")
        return bool(connected_lines[line_number])

    def set_connected(self, channel_name: str, line_number: int) -> None:
        """Set the channel to connected to the line number."""
        return self.get_channel(channel_name).set_parameter("connect", line_number)

    def get_disconnected(self, channel_name: str, line_number: int) -> bool:
        """Get if the channel is disconnected from the line number."""
        disconnected_lines = self.get_channel(channel_name).get_parameter_value(
            "disconnect"
        )
        if line_number < 0 or line_number >= len(disconnected_lines):
            raise ValueError(f"Line number {line_number} is out of range")
        return bool(disconnected_lines[line_number])

    def set_disconnected(self, channel_name: str, line_number: int) -> None:
        """Set the channel to disconnected from the line number."""
        return self.get_channel(channel_name).set_parameter("disconnect", line_number)
