from __future__ import annotations

import logging
from typing import overload

from stanza.base.channels import ChannelConfig, InstrumentChannel, Parameter
from stanza.base.instruments import BaseControlInstrument
from stanza.models import ControlInstrumentConfig
from stanza.pyvisa import PyVisaDriver

logger = logging.getLogger(__name__)


class QSwitchChannel(InstrumentChannel):
    """QSwitch channel for setting and getting the state of the lines and relays."""

    def __init__(
        self, name: str, channel_id: int, config: ChannelConfig, driver: PyVisaDriver
    ):
        self.name = name
        self.driver = driver
        super().__init__(config)
        self.channel_id = channel_id

    def _query_all_relays(self, command: str) -> list[int]:
        """Query all 10 relays and return their states."""
        relays = ",".join(f"{self.channel_id}!{i}" for i in range(10))
        response = self.driver.query(f"{command}? (@{relays})")
        return [int(v) for v in response.split(",")]

    def _set_relay(self, command: str, relay: int | str) -> None:
        """Set a single relay or multiple relays."""
        if isinstance(relay, int):
            self.driver.write(f"{command} (@{self.channel_id}!{relay})")
        else:
            relays = ",".join(f"(@{self.channel_id}!{r})" for r in relay.split(","))
            self.driver.write(f"{command} (@{relays})")

    def _get_connect(self) -> list[int]:
        return self._query_all_relays("close")

    def _set_connect(self, relay: int | str) -> None:
        self._set_relay("close", relay)

    def _get_disconnect(self) -> list[int]:
        return self._query_all_relays("open")

    def _set_disconnect(self, relay: int | str) -> None:
        self._set_relay("open", relay)

    def _setup_parameters(self) -> None:
        """Setup parameters after channel_id is properly set."""
        self.add_parameter(
            Parameter(
                name="connect_relay",
                value=None,
                unit="bool",
                getter=self._get_connect,
                setter=self._set_connect,
            )
        )

        self.add_parameter(
            Parameter(
                name="disconnect_relay",
                value=None,
                unit="bool",
                getter=self._get_disconnect,
                setter=self._set_disconnect,
            )
        )


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
        for cfg in channel_configs.values():
            if cfg.breakout_channel is not None:
                self.add_channel(
                    cfg.name,
                    QSwitchChannel(cfg.name, cfg.breakout_channel, cfg, self.driver),
                )

    def _channels_suffix(self, channel_names: list[str], tap: int | str) -> str:
        """Get the channels suffix for the driver."""
        channels = ",".join(
            f"{self.channel_configs[ch].breakout_channel}!{tap}" for ch in channel_names
        )
        return f"(@{channels})"

    @overload
    def get_grounded(self, channel_name: list[str]) -> list[bool]: ...

    @overload
    def get_grounded(self, channel_name: str) -> bool: ...

    def get_grounded(self, channel_name: str | list[str]) -> bool | list[bool]:
        """Get if the channel is grounded."""
        return self.get_connected(channel_name, 0)

    @overload
    def set_grounded(self, channel_name: list[str]) -> None: ...

    @overload
    def set_grounded(self, channel_name: str) -> None: ...

    def set_grounded(self, channel_name: str | list[str]) -> None:
        """Set the channel to grounded."""
        self.set_connected(channel_name, 0)

    @overload
    def get_ungrounded(self, channel_name: list[str]) -> list[bool]: ...

    @overload
    def get_ungrounded(self, channel_name: str) -> bool: ...

    def get_ungrounded(self, channel_name: str | list[str]) -> bool | list[bool]:
        """Get if the channel is ungrounded."""
        return self.get_disconnected(channel_name, 0)

    @overload
    def set_ungrounded(self, channel_name: list[str]) -> None: ...

    @overload
    def set_ungrounded(self, channel_name: str) -> None: ...

    def set_ungrounded(self, channel_name: str | list[str]) -> None:
        """Set the channel to ungrounded."""
        self.set_disconnected(channel_name, 0)

    @overload
    def get_connected(
        self, channel_name: list[str], line_number: int
    ) -> list[bool]: ...

    @overload
    def get_connected(self, channel_name: str, line_number: int) -> bool: ...

    def get_connected(
        self, channel_name: str | list[str], line_number: int
    ) -> bool | list[bool]:
        """Get if the channel is connected to the line number."""
        if isinstance(channel_name, str):
            return bool(
                self.get_channel(channel_name).get_parameter_value("connect_relay")[
                    line_number
                ]
            )

        response = self.driver.query(
            f"close? {self._channels_suffix(channel_name, line_number)}"
        )
        return [bool(int(v)) for v in response.split(",")]

    @overload
    def set_connected(self, channel_name: list[str], line_number: int) -> None: ...

    @overload
    def set_connected(self, channel_name: str, line_number: int) -> None: ...

    def set_connected(self, channel_name: str | list[str], line_number: int) -> None:
        """Set the channel to connected to the line number."""
        if isinstance(channel_name, str):
            self.get_channel(channel_name).set_parameter("connect_relay", line_number)
        else:
            self.driver.write(
                f"close {self._channels_suffix(channel_name, line_number)}"
            )

    @overload
    def get_disconnected(
        self, channel_name: list[str], line_number: int
    ) -> list[bool]: ...

    @overload
    def get_disconnected(self, channel_name: str, line_number: int) -> bool: ...

    def get_disconnected(
        self, channel_name: str | list[str], line_number: int
    ) -> bool | list[bool]:
        """Get if the channel is disconnected from the line number."""
        if isinstance(channel_name, str):
            return bool(
                self.get_channel(channel_name).get_parameter_value("disconnect_relay")[
                    line_number
                ]
            )

        response = self.driver.query(
            f"open? {self._channels_suffix(channel_name, line_number)}"
        )
        return [bool(int(v)) for v in response.split(",")]

    @overload
    def set_disconnected(self, channel_name: list[str], line_number: int) -> None: ...

    @overload
    def set_disconnected(self, channel_name: str, line_number: int) -> None: ...

    def set_disconnected(self, channel_name: str | list[str], line_number: int) -> None:
        """Set the channel to disconnected from the line number."""
        if isinstance(channel_name, str):
            self.get_channel(channel_name).set_parameter(
                "disconnect_relay", line_number
            )
        else:
            self.driver.write(
                f"open {self._channels_suffix(channel_name, line_number)}"
            )

    def close(self) -> None:
        """Close the QSwitch driver."""
        self.driver.close()
