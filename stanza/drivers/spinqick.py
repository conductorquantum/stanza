from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING

from stanza.base.channels import ChannelConfig, ControlChannel, Parameter
from stanza.base.instruments import BaseInstrument
from stanza.models import BaseInstrumentConfig

if TYPE_CHECKING:
    from spinqick.helper_functions.hardware_manager import DCSource, VoltageSource

try:
    from spinqick.helper_functions.hardware_manager import DCSource, VoltageSource

    HAS_SPINQICK = True
except ImportError:
    HAS_SPINQICK = False

logger = logging.getLogger(__name__)


class SpinQickControlChannel(ControlChannel):
    """Control channel for SpinQICK gates.

    This channel interfaces with SpinQICK's DCSource voltage control system,
    which provides gate-level voltage control with cross-coupling compensation.
    """

    def __init__(
        self,
        name: str,
        gate_name: str,
        config: ChannelConfig,
        dc_source: "DCSource",
    ):
        """Initialize SpinQICK control channel.

        Args:
            name: Channel name in Stanza
            gate_name: Gate name in SpinQICK (e.g., 'P1', 'M1', 'B1')
            config: Channel configuration
            dc_source: SpinQICK DCSource instance for voltage control
        """
        self.name = name
        self.gate_name = gate_name
        self.dc_source = dc_source
        self.channel_id = None  # SpinQICK uses gate names, not numeric IDs
        super().__init__(config)

    def _setup_parameters(self) -> None:
        """Setup SpinQICK-specific control parameters with hardware integration."""
        super()._setup_parameters()

        # Override voltage parameter with SpinQICK-specific getter/setter
        # Note: DCSource.set_dc_voltage expects (volts, gate) signature
        # Note: DCSource.get_dc_voltage expects (gate) signature
        voltage_param = self.get_parameter("voltage")
        voltage_param.setter = lambda v: self.dc_source.set_dc_voltage(volts=v, gate=self.gate_name)
        voltage_param.getter = lambda: self.dc_source.get_dc_voltage(gate=self.gate_name)

        # Add SpinQICK-specific parameters

        # Compensated voltage setting - allows setting voltage while compensating other gates
        compensate_gates_param = Parameter(
            name="compensate_gates",
            value=[],
            unit="",
            metadata={
                "description": "List of gate names to keep constant during voltage changes"
            },
        )
        self.add_parameter(compensate_gates_param)

    def set_voltage_compensated(
        self, voltage: float, iso_gates: list[str] | str
    ) -> None:
        """Set gate voltage with cross-coupling compensation.

        This uses SpinQICK's set_dc_voltage_compensate to adjust other gates
        to keep iso_gates at constant potential.

        Args:
            voltage: Target voltage at this gate (V)
            iso_gates: Gate name(s) to keep constant (e.g., ['M1'] for charge sensor)
        """
        if not HAS_SPINQICK:
            raise ImportError("SpinQICK is not installed")

        self.dc_source.set_dc_voltage_compensate(
            volts=voltage, gates=self.gate_name, iso_gates=iso_gates
        )
        self.parameters["voltage"].value = voltage


class StanzaVoltageSourceAdapter:
    """Adapter that allows Stanza instrument to act as a SpinQICK VoltageSource.

    This enables SpinQICK's DCSource to use any Stanza-compatible voltage
    source (QDAC2, etc.) through the VoltageSource protocol.
    """

    def __init__(self, stanza_instrument: BaseInstrument, gate_to_channel: dict[str, str]):
        """Initialize adapter.

        Args:
            stanza_instrument: Stanza instrument (e.g., QDAC2) providing voltage control
            gate_to_channel: Mapping from SpinQICK gate names to Stanza channel names
                            e.g., {'P1': 'plunger1', 'M1': 'sensor'}
        """
        self.instrument = stanza_instrument
        self.gate_to_channel = gate_to_channel

    def open(self, address: str) -> None:
        """Open connection (no-op for Stanza, connection already managed)."""
        pass

    def close(self) -> None:
        """Close connection (no-op for Stanza, connection already managed)."""
        pass

    def get_voltage(self, ch: int) -> float:
        """Get voltage from channel (SpinQICK uses channel numbers)."""
        # For SpinQICK compatibility, we need to support numeric channels
        # This requires reverse mapping from channel number to gate name
        raise NotImplementedError(
            "Direct channel number access not supported. Use gate names instead."
        )

    def set_voltage(self, ch: int, volts: float) -> None:
        """Set voltage on channel (SpinQICK uses channel numbers)."""
        raise NotImplementedError(
            "Direct channel number access not supported. Use gate names instead."
        )

    def get_voltage_by_gate(self, gate: str) -> float:
        """Get voltage for a specific gate."""
        channel_name = self.gate_to_channel[gate]
        return self.instrument.get_voltage(channel_name)

    def set_voltage_by_gate(self, gate: str, volts: float) -> None:
        """Set voltage for a specific gate."""
        channel_name = self.gate_to_channel[gate]
        self.instrument.set_voltage(channel_name, volts)

    def set_sweep(
        self, ch: int, start: float, stop: float, length: float, num_steps: int
    ) -> None:
        """Program voltage sweep (if supported by underlying instrument)."""
        raise NotImplementedError("Sweep functionality not yet implemented")

    def trigger(self, ch: int) -> None:
        """Trigger programmed sweep."""
        raise NotImplementedError("Trigger functionality not yet implemented")

    def arm_sweep(self, ch: int) -> None:
        """Arm sweep for trigger."""
        raise NotImplementedError("Arm sweep functionality not yet implemented")


class SpinQick(BaseInstrument):
    """SpinQICK instrument driver for Stanza.

    This driver provides integration between Stanza's instrument framework
    and SpinQICK's DCSource voltage control system. It allows using SpinQICK's
    advanced features like cross-coupling compensation, voltage ramps, and
    gate-level abstraction within Stanza experiments.

    SpinQICK uses a gate-centric approach where voltages are controlled by
    gate names (P1, P2, M1, B1, etc.) defined in the hardware_config.json.
    This driver maps Stanza channels to SpinQICK gate names.

    Example:
        >>> from stanza.drivers.spinqick import SpinQick
        >>> from spinqick.helper_functions.hardware_manager import DCSource
        >>>
        >>> # Create SpinQICK instrument
        >>> spinqick = SpinQick(
        ...     instrument_config=config,
        ...     channel_configs=channels,
        ...     voltage_source=my_qdac  # Any SpinQICK-compatible voltage source
        ... )
        >>>
        >>> # Set voltages using Stanza API
        >>> spinqick.set_voltage('P1', 0.5)
        >>>
        >>> # Use SpinQICK-specific features via channels
        >>> channel = spinqick.get_channel('control_P1')
        >>> channel.set_voltage_compensated(0.6, iso_gates=['M1'])
    """

    def __init__(
        self,
        instrument_config: BaseInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        voltage_source: "VoltageSource",
    ):
        """Initialize SpinQICK instrument.

        Args:
            instrument_config: Instrument configuration
            channel_configs: Channel configurations mapping Stanza channel names
                           to gate configurations
            voltage_source: SpinQICK-compatible voltage source (QDAC, Basel LNHR, etc.)
                          This should implement the VoltageSource protocol from
                          spinqick.helper_functions.hardware_manager

        Note:
            This requires:
            1. SpinQICK to be installed (pip install spinqick)
            2. A valid hardware_config.json file configured for SpinQICK
            3. The voltage_source to be initialized and connected
        """
        if not HAS_SPINQICK:
            raise ImportError(
                "SpinQICK is not installed. Install with: pip install spinqick"
            )

        self.name = instrument_config.name
        self.channel_configs = channel_configs

        # Initialize SpinQICK's DCSource
        # This loads the hardware_config.json and sets up gate mappings
        self.dc_source = DCSource(voltage_source=voltage_source)

        # Extract control channels
        self.control_channels = [
            cfg.name
            for cfg in channel_configs.values()
            if cfg.control_channel is not None
        ]

        super().__init__(instrument_config)
        self._initialize_channels(channel_configs)

    def _initialize_channels(self, channel_configs: dict[str, ChannelConfig]) -> None:
        """Initialize SpinQICK control channels.

        Maps Stanza channel names to SpinQICK gate names. By convention,
        we use the channel name as the gate name unless specified otherwise.
        """
        for channel_config in channel_configs.values():
            if channel_config.control_channel is not None:
                # Use the channel name as the SpinQICK gate name
                gate_name = channel_config.name

                self.add_channel(
                    f"control_{channel_config.name}",
                    SpinQickControlChannel(
                        name=channel_config.name,
                        gate_name=gate_name,
                        config=channel_config,
                        dc_source=self.dc_source,
                    ),
                )

    def set_voltage(self, channel_name: str, voltage: float) -> None:
        """Set voltage on a specific gate.

        Args:
            channel_name: Stanza channel name (maps to SpinQICK gate name)
            voltage: Target voltage in volts
        """
        return super().set_voltage(f"control_{channel_name}", voltage)

    def get_voltage(self, channel_name: str) -> float:
        """Get voltage on a specific gate.

        Args:
            channel_name: Stanza channel name (maps to SpinQICK gate name)

        Returns:
            Current voltage in volts
        """
        return super().get_voltage(f"control_{channel_name}")

    def set_voltage_compensated(
        self, channel_name: str, voltage: float, iso_gates: list[str] | str
    ) -> None:
        """Set voltage with cross-coupling compensation.

        Uses SpinQICK's compensated voltage setting to adjust other gates
        and keep iso_gates at constant potential.

        Args:
            channel_name: Gate to change
            voltage: Target voltage
            iso_gates: Gate(s) to keep constant (e.g., ['M1'] for charge sensor)
        """
        channel = self.get_channel(f"control_{channel_name}")
        if isinstance(channel, SpinQickControlChannel):
            channel.set_voltage_compensated(voltage, iso_gates)
        else:
            raise TypeError(f"Channel {channel_name} is not a SpinQickControlChannel")

    def program_ramp(
        self,
        channel_name: str,
        vstart: float,
        vstop: float,
        tstep: float,
        nsteps: int,
    ) -> None:
        """Program a voltage ramp on a gate.

        Must be followed by arm_sweep() and trigger() to execute.

        Args:
            channel_name: Gate to ramp
            vstart: Start voltage (V)
            vstop: End voltage (V)
            tstep: Time per step (seconds)
            nsteps: Number of steps
        """
        channel = self.get_channel(f"control_{channel_name}")
        if isinstance(channel, SpinQickControlChannel):
            self.dc_source.program_ramp(
                vstart=vstart,
                vstop=vstop,
                tstep=tstep,
                nsteps=nsteps,
                gate=channel.gate_name,
            )
        else:
            raise TypeError(f"Channel {channel_name} is not a SpinQickControlChannel")

    def arm_sweep(self, channel_name: str) -> None:
        """Arm the programmed sweep on a gate."""
        channel = self.get_channel(f"control_{channel_name}")
        if isinstance(channel, SpinQickControlChannel):
            self.dc_source.arm_sweep(channel.gate_name)
        else:
            raise TypeError(f"Channel {channel_name} is not a SpinQickControlChannel")

    def trigger_sweep(self, channel_name: str) -> None:
        """Trigger the programmed sweep on a gate."""
        channel = self.get_channel(f"control_{channel_name}")
        if isinstance(channel, SpinQickControlChannel):
            self.dc_source.digital_trigger(channel.gate_name)
        else:
            raise TypeError(f"Channel {channel_name} is not a SpinQickControlChannel")

    def get_all_voltages(self) -> dict[str, float]:
        """Get voltages on all gates.

        Returns:
            Dictionary mapping gate names to voltages
        """
        return self.dc_source.all_voltages

    def save_voltage_state(self, file_path: str | None = None) -> None:
        """Save current voltage state of all gates to YAML file.

        Args:
            file_path: Optional path to save to. If None, uses SpinQICK's
                      default timestamped filename.
        """
        self.dc_source.save_voltage_state(file_path)

    @cached_property
    def idn(self) -> str:
        """Get instrument identification."""
        return f"SpinQICK DCSource (voltage_source={self.dc_source.source_type})"

    def __str__(self) -> str:
        return f"SpinQick(name={self.name}, gates={list(self.control_channels)})"
