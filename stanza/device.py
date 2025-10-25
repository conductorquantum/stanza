import time
from collections.abc import Callable
from typing import Any, cast, overload

import numpy as np

from stanza.base.channels import ChannelConfig
from stanza.base.instruments import BaseControlInstrument, BaseMeasurementInstrument
from stanza.base.protocols import (
    BreakoutBoxInstrument,
    ControlInstrument,
    MeasurementInstrument,
)
from stanza.exceptions import DeviceError
from stanza.logger.session import LoggerSession
from stanza.models import ContactType, DeviceConfig, GateType, PadType


class Device:
    """Interface for controlling quantum devices through voltage sweeps and current measurements.

    The Device class provides a high-level abstraction for interacting with quantum
    devices by coordinating between control instruments (for setting voltages) and
    measurement instruments (for reading currents). It supports various sweep patterns
    (1D, 2D, N-dimensional) and manages the mapping between logical pad names and
    physical instrument channels.

    Attributes:
        name: Human-readable name of the device
        device_config: Configuration object containing device specifications
        channel_configs: Dictionary mapping pad names to their channel configurations
        control_instrument: Instrument used for setting voltages (must implement ControlInstrument protocol)
        measurement_instrument: Instrument used for measuring currents (must implement MeasurementInstrument protocol)

    The class distinguishes between different pad types (gates vs contacts) and
    electrode types (e.g., BARRIER, PLUNGER for gates; SOURCE, DRAIN for contacts),
    providing convenient properties and methods for filtering and accessing specific
    electrodes based on their characteristics.
    """

    def __init__(
        self,
        name: str,
        device_config: DeviceConfig,
        channel_configs: dict[str, ChannelConfig],
        control_instrument: Any | None,
        measurement_instrument: Any | None,
        breakout_box_instrument: Any | None = None,
    ):
        self.name = name
        self.device_config = device_config

        if control_instrument and not isinstance(control_instrument, ControlInstrument):
            raise DeviceError(
                "Control instrument must implement the `ControlInstrument` protocol"
            )

        if measurement_instrument and not isinstance(
            measurement_instrument, MeasurementInstrument
        ):
            raise DeviceError(
                "Measurement instrument must implement the `MeasurementInstrument` protocol"
            )

        if breakout_box_instrument and not isinstance(
            breakout_box_instrument, BreakoutBoxInstrument
        ):
            raise DeviceError(
                "Breakout Box instrument must implement the `BreakoutBoxInstrument` protocol"
            )

        self.control_instrument = cast(ControlInstrument | None, control_instrument)
        self.measurement_instrument = cast(
            MeasurementInstrument | None, measurement_instrument
        )
        self.breakout_box_instrument = cast(
            BreakoutBoxInstrument | None, breakout_box_instrument
        )
        self.channel_configs = channel_configs

    @property
    def breakout_lines(self) -> list[str]:
        """List of all breakout box line names in the device."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.breakout_channel is not None
        ]

    @property
    def gates(self) -> list[str]:
        """List of all gate pad names in the device."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.GATE
        ]

    @property
    def contacts(self) -> list[str]:
        """List of all contact pad names in the device."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.CONTACT
        ]

    @property
    def gpios(self) -> list[str]:
        """List of all gpio pad names in the device."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.GPIO
        ]

    @property
    def control_gates(self) -> list[str]:
        """List of gate pads that have a control channel configured."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.GATE and channel.control_channel is not None
        ]

    @property
    def control_contacts(self) -> list[str]:
        """List of contact pads that have a control channel configured."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.CONTACT
            and channel.control_channel is not None
        ]

    @property
    def control_gpios(self) -> list[str]:
        """List of gpio pads that have a control channel configured."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.GPIO and channel.control_channel is not None
        ]

    @property
    def measurement_gates(self) -> list[str]:
        """List of gate pads that have a measurement channel configured."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.GATE and channel.measure_channel is not None
        ]

    @property
    def measurement_contacts(self) -> list[str]:
        """List of contact pads that have a measurement channel configured."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.CONTACT
            and channel.measure_channel is not None
        ]

    def get_gates_by_type(self, gate_type: str | GateType) -> list[str]:
        """Get the gate electrodes of a given type.

        Filters all gate channels in the device to return only those matching
        the specified gate type (e.g., BARRIER, PLUNGER, SCREENING).

        Args:
            gate_type: The type of gate electrodes to retrieve. Can be provided
                as either a GateType enum value or a string (case-insensitive)
                that will be converted to GateType.

        Returns:
            List of gate electrode names matching the specified type. Returns
            an empty list if no gates of the specified type are found.
        """
        if isinstance(gate_type, str):
            gate_type = GateType(gate_type.upper())
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.GATE and channel.electrode_type == gate_type
        ]

    def get_contacts_by_type(self, contact_type: str | ContactType) -> list[str]:
        """Get the contact electrodes of a given type.

        Filters all contact channels in the device to return only those matching
        the specified contact type (e.g., SOURCE, DRAIN, RESERVOIR).

        Args:
            contact_type: The type of contact electrodes to retrieve. Can be
                provided as either a ContactType enum value or a string
                (case-insensitive) that will be converted to ContactType.

        Returns:
            List of contact electrode names matching the specified type. Returns
            an empty list if no contacts of the specified type are found.
        """
        if isinstance(contact_type, str):
            contact_type = ContactType(contact_type.upper())
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.CONTACT
            and channel.electrode_type == contact_type
        ]

    def is_configured(self) -> bool:
        """Check if both instruments are configured.

        Verifies that the device has both a control instrument (for setting
        voltages) and a measurement instrument (for reading currents) properly
        configured and available for use.

        Returns:
            True if both control_instrument and measurement_instrument are not
            None, False otherwise.
        """
        return (
            self.control_instrument is not None
            and self.measurement_instrument is not None
        )

    def _jump(self, voltage: float, pad: str, wait_for_settling: bool = False) -> None:
        """Set the voltage of a single gate"""
        if not self.control_instrument:
            raise DeviceError("Control instrument not configured")

        try:
            settling_time = 0.0
            if wait_for_settling:
                current_voltage = self.control_instrument.get_voltage(pad)
                slew_rate = self.control_instrument.get_slew_rate(pad)
                voltage_diff = abs(voltage - current_voltage)
                settling_time = 1.2 * (voltage_diff / slew_rate)

            self.control_instrument.set_voltage(pad, voltage)
            if settling_time > 0:
                time.sleep(settling_time)
        except Exception as e:
            raise DeviceError(f"Failed to set voltage {voltage}V on {pad}: {e}") from e

    def jump(
        self, pad_voltages: dict[str, float], wait_for_settling: bool = False
    ) -> None:
        """Set the voltages of the device.

        Args:
            pad_voltages: A dictionary of pads and their voltages.
            wait_for_settling: Whether to wait for the device to settle after setting the voltages.

        Raises:
            DeviceError: If the control instrument is not configured.
        """
        for pad, voltage in pad_voltages.items():
            self._jump(voltage, pad, wait_for_settling)

    def _measure(self, pad: str) -> float:
        """Measure the current of a single gate"""
        if not self.measurement_instrument:
            raise DeviceError("Measurement instrument not configured")

        if pad not in self.channel_configs:
            raise DeviceError(f"Pad {pad} not found in channel configs")

        if self.channel_configs[pad].measure_channel is None:
            raise DeviceError(f"Pad {pad} has no measure channel")

        return self.measurement_instrument.measure(pad)

    @overload
    def measure(self, pad: str) -> float: ...

    @overload
    def measure(self, pad: list[str]) -> list[float]: ...

    def measure(self, pad: str | list[str]) -> float | list[float]:
        """Measure the current of the device.

        Performs current measurement on one or more pads using the measurement
        instrument. For multiple pads, attempts to use the instrument's batch
        measurement capability if available, otherwise measures sequentially.

        Args:
            pad: Either a single pad name or a list of pad names to measure.
                Each pad must be configured with a measure_channel.

        Returns:
            If a single pad name is provided, returns a single float current value.
            If a list of pad names is provided, returns a list of float current values
            corresponding to each pad in the input list.

        Raises:
            DeviceError: If the measurement instrument is not configured, if a
                specified pad is not found in channel configs, or if a pad has
                no measure channel configured.
        """
        if isinstance(pad, str):
            return self._measure(pad)
        else:
            if (
                self.measurement_instrument
                and hasattr(self.measurement_instrument, "measure")
                and callable(self.measurement_instrument.measure)
            ):
                try:
                    return self.measurement_instrument.measure(pad)
                except Exception as _:
                    return [self._measure(p) for p in pad]
            else:
                return [self._measure(p) for p in pad]

    def _check(self, pad: str) -> float:
        """Check the current voltage of a single gate electrode."""
        if not self.control_instrument:
            raise DeviceError("Control instrument not configured")

        if pad not in self.channel_configs:
            raise DeviceError(f"Pad {pad} not found in channel configs")

        if self.channel_configs[pad].control_channel is None:
            raise DeviceError(f"Pad {pad} has no control channel")

        return self.control_instrument.get_voltage(pad)

    @overload
    def check(self, pad: str) -> float: ...

    @overload
    def check(self, pad: list[str]) -> list[float]: ...

    def check(self, pad: str | list[str]) -> float | list[float]:
        """Check the current voltage of the device.

        Reads the current voltage setting from one or more pads using the control
        instrument. This returns the voltage that the control instrument believes
        it has set, not a measured value from the device itself.

        Args:
            pad: Either a single pad name or a list of pad names to check.
                Each pad must be configured with a control_channel.

        Returns:
            If a single pad name is provided, returns a single float voltage value.
            If a list of pad names is provided, returns a list of float voltage values
            corresponding to each pad in the input list.

        Raises:
            DeviceError: If the control instrument is not configured, if a
                specified pad is not found in channel configs, or if a pad has
                no control channel configured.
        """
        if isinstance(pad, str):
            return self._check(pad)
        else:
            return [self._check(p) for p in pad]

    def sweep_1d(
        self,
        gate_electrode: str,
        voltages: list[float],
        measure_electrode: str,
        session: LoggerSession | None = None,
    ) -> tuple[list[float], list[float]]:
        """Sweep a single gate electrode and measure the current of a single contact electrode.

        Performs a 1D voltage sweep by stepping through a list of voltages on a
        specified gate electrode while measuring the current through a contact
        electrode at each step. Optionally logs the sweep data to a session.

        Args:
            gate_electrode: Name of the gate electrode to sweep
            voltages: List of voltage values to apply to the gate electrode
            measure_electrode: Name of the contact electrode to measure current from
            session: Optional LoggerSession to log the sweep data. If provided,
                sweep results will be logged with metadata.

        Returns:
            Tuple of (voltage_measurements, current_measurements) where:
            - voltage_measurements: List of actual voltages read from the gate
            - current_measurements: List of measured current values at each voltage
        """
        voltage_measurements = []
        current_measurements = []

        if session is None:
            # No logging - just collect data
            for i, voltage in enumerate(voltages):
                should_settle = i == 0
                self.jump({gate_electrode: voltage}, wait_for_settling=should_settle)
                v_actual = self.check(gate_electrode)
                i_measured = self.measure(measure_electrode)
                voltage_measurements.append(v_actual)
                current_measurements.append(i_measured)
        else:
            # With logging and live plotting
            metadata = {
                "gate_electrodes": [gate_electrode],
                "measure_electrode": measure_electrode,
            }
            with session.sweep(
                f"{gate_electrode} sweep", "Voltage", "Current", metadata=metadata
            ) as s:
                for i, voltage in enumerate(voltages):
                    should_settle = i == 0
                    self.jump(
                        {gate_electrode: voltage}, wait_for_settling=should_settle
                    )
                    v_actual = self.check(gate_electrode)
                    i_measured = self.measure(measure_electrode)
                    voltage_measurements.append(v_actual)
                    current_measurements.append(i_measured)
                    s.append([v_actual], [i_measured])

        return voltage_measurements, current_measurements

    def sweep_2d(
        self,
        gate_1: str,
        voltages_1: list[float],
        gate_2: str,
        voltages_2: list[float],
        measure_electrode: str,
        session: LoggerSession | None = None,
    ) -> tuple[list[list[float]], list[float]]:
        """Sweep two gate electrodes and measure the current of a single contact electrode.

        Performs a 2D voltage sweep by iterating through all combinations of
        voltages on two gate electrodes while measuring current through a contact
        electrode. The sweep iterates through gate_1 voltages in the outer loop
        and gate_2 voltages in the inner loop.

        Args:
            gate_1: Name of the first gate electrode to sweep
            voltages_1: List of voltage values for the first gate electrode
            gate_2: Name of the second gate electrode to sweep
            voltages_2: List of voltage values for the second gate electrode
            measure_electrode: Name of the contact electrode to measure current from
            session: Optional LoggerSession to log the sweep data. If provided,
                sweep results will be logged with metadata.

        Returns:
            Tuple of (voltage_measurements, current_measurements) where:
            - voltage_measurements: List of [gate_1_voltage, gate_2_voltage] pairs
            - current_measurements: List of measured current values at each voltage pair
        """
        voltage_measurements = []
        current_measurements = []

        if session is None:
            # No logging - just collect data
            for i, voltage_1 in enumerate(voltages_1):
                for j, voltage_2 in enumerate(voltages_2):
                    should_settle = i == 0 and j == 0
                    self.jump(
                        {gate_1: voltage_1, gate_2: voltage_2},
                        wait_for_settling=should_settle,
                    )
                    v1_actual = self.check(gate_1)
                    v2_actual = self.check(gate_2)
                    i_measured = self.measure(measure_electrode)
                    voltage_measurements.append([v1_actual, v2_actual])
                    current_measurements.append(i_measured)
        else:
            # With logging and live plotting
            metadata = {
                "gate_electrodes": [gate_1, gate_2],
                "measure_electrode": measure_electrode,
            }
            with session.sweep(
                f"{gate_1} and {gate_2} sweep",
                [gate_1, gate_2],
                "Current",
                metadata=metadata,
            ) as s:
                for i, voltage_1 in enumerate(voltages_1):
                    for j, voltage_2 in enumerate(voltages_2):
                        should_settle = i == 0 and j == 0
                        self.jump(
                            {gate_1: voltage_1, gate_2: voltage_2},
                            wait_for_settling=should_settle,
                        )
                        v1_actual = self.check(gate_1)
                        v2_actual = self.check(gate_2)
                        i_measured = self.measure(measure_electrode)
                        voltage_measurements.append([v1_actual, v2_actual])
                        current_measurements.append(i_measured)
                        s.append([v1_actual, v2_actual], [i_measured])

        return voltage_measurements, current_measurements

    def sweep_all(
        self,
        voltages: list[float],
        measure_electrode: str,
        session: LoggerSession | None = None,
    ) -> tuple[list[list[float]], list[float]]:
        """Sweep all gate electrodes and measure the current of a single contact electrode.

        Performs a voltage sweep by setting all control gates to the same voltage
        at each step, while measuring current through a contact electrode. This is
        useful for characterizing device response to overall gate bias.

        Args:
            voltages: List of voltage values to apply to all control gates simultaneously
            measure_electrode: Name of the contact electrode to measure current from
            session: Optional LoggerSession to log the sweep data. If provided,
                sweep results will be logged with metadata.

        Returns:
            Tuple of (voltage_measurements, current_measurements) where:
            - voltage_measurements: List of lists, each containing voltage values for
              all control gates at that sweep step
            - current_measurements: List of measured current values at each voltage
        """
        voltage_measurements = []
        current_measurements = []

        if session is None:
            # No logging - just collect data
            for i, voltage in enumerate(voltages):
                should_settle = i == 0
                self.jump(
                    dict.fromkeys(self.control_gates, voltage),
                    wait_for_settling=should_settle,
                )
                i_measured = self.measure(measure_electrode)
                voltage_measurements.append([voltage])
                current_measurements.append(i_measured)
        else:
            # With logging and live plotting
            metadata = {
                "gate_electrodes": self.control_gates,
                "measure_electrode": measure_electrode,
            }
            with session.sweep(
                "all gates sweep", "Voltage", "Current", metadata=metadata
            ) as s:
                for i, voltage in enumerate(voltages):
                    should_settle = i == 0
                    self.jump(
                        dict.fromkeys(self.control_gates, voltage),
                        wait_for_settling=should_settle,
                    )
                    i_measured = self.measure(measure_electrode)
                    voltage_measurements.append([voltage])
                    current_measurements.append(i_measured)
                    s.append([voltage], [i_measured])

        return voltage_measurements, current_measurements

    def sweep_nd(
        self,
        gate_electrodes: list[str],
        voltages: list[list[float]],
        measure_electrode: str,
        session: LoggerSession | None = None,
    ) -> tuple[list[list[float]], list[float]]:
        """Sweep multiple gate electrodes and measure the current of a single contact electrode.

        Performs an N-dimensional voltage sweep where each specified gate can have
        a different voltage at each sweep step. This provides maximum flexibility
        for arbitrary multi-gate sweeps and trajectories through voltage space.

        Args:
            gate_electrodes: List of gate electrode names to sweep
            voltages: List of voltage lists, where each inner list contains the
                voltages for all gates at that sweep step. Each inner list must
                have the same length as gate_electrodes.
            measure_electrode: Name of the contact electrode to measure current from
            session: Optional LoggerSession to log the sweep data. If provided,
                sweep results will be logged with metadata.

        Returns:
            Tuple of (voltage_measurements, current_measurements) where:
            - voltage_measurements: List of lists, each containing the actual voltage
              values read from each gate at that sweep step
            - current_measurements: List of measured current values at each voltage combination
        """
        voltage_measurements = []
        current_measurements = []

        for i, voltage in enumerate(voltages):
            should_settle = i == 0
            self.jump(
                dict(zip(gate_electrodes, voltage, strict=True)),
                wait_for_settling=should_settle,
            )

            voltage_measurements.append(
                [
                    self.check(gate) or v
                    for gate, v in zip(gate_electrodes, voltage, strict=True)
                ]
            )
            current_measurements.append(self.measure(measure_electrode))

        if session:
            session.log_sweep(
                name="n gates sweep",
                x_data=voltage_measurements,
                y_data=current_measurements,
                x_label="Voltage",
                y_label="Current",
                metadata={
                    "gate_electrodes": gate_electrodes,
                    "measure_electrode": measure_electrode,
                },
            )
        return voltage_measurements, current_measurements

    def zero(self, type: str | PadType = PadType.ALL) -> None:
        """Set all controllable gates and/or controllable contacts to 0V.

        Safely brings specified electrodes to ground voltage (0V) with settling
        time and verification. This is typically used for device initialization
        or safe shutdown.

        Args:
            type: Specifies which pads to zero. Options are:
                - PadType.ALL or "ALL": Zero all control gates, contacts, and gpios (default)
                - PadType.GATE or "GATE": Zero only control gates
                - PadType.CONTACT or "CONTACT": Zero only control contacts
                - PadType.GPIO or "GPIO": Zero only control gpios
                Can be provided as PadType enum or case-insensitive string.

        Raises:
            DeviceError: If an invalid pad type is provided, or if any pad fails
                to reach 0V within tolerance (1e-6V) after the operation.
        """
        pads: list[str] = []
        match str(type).upper():
            case PadType.ALL:
                pads = self.control_gates + self.control_contacts + self.control_gpios
            case PadType.GATE:
                pads = self.control_gates
            case PadType.CONTACT:
                pads = self.control_contacts
            case PadType.GPIO:
                pads = self.control_gpios
            case _:
                raise DeviceError(f"Invalid pad type: {type}")

        gate_voltages = dict.fromkeys(pads, 0.0)
        self.jump(gate_voltages, wait_for_settling=True)

        actual_voltages = self.check(pads)
        if not np.allclose(actual_voltages, [0.0] * len(actual_voltages), atol=1e-6):
            raise DeviceError("Failed to set all controllable pads to 0V")

    def ground_breakout_lines(self) -> None:
        """Ground all breakout box lines.

        This method sets all breakout box lines to a grounded state.

        Raises:
            DeviceError: If the breakout box instrument is not configured.
        """
        if not self.breakout_box_instrument:
            raise DeviceError("Breakout box instrument not configured")

        self.breakout_box_instrument.set_grounded(self.breakout_lines)

    def unground_breakout_lines(self) -> None:
        """Unground all breakout box lines.

        This method sets all breakout box lines to an ungrounded state.

        Raises:
            DeviceError: If the breakout box instrument is not configured.
        """
        if not self.breakout_box_instrument:
            raise DeviceError("Breakout box instrument not configured")

        self.breakout_box_instrument.set_ungrounded(self.breakout_lines)

    def connect_breakout_lines(self) -> None:
        """Connect all breakout box lines.

        This method connects all breakout box lines to the instrument.

        Raises:
            DeviceError: If the breakout box instrument is not configured.
        """
        if not self.breakout_box_instrument:
            raise DeviceError("Breakout box instrument not configured")
        self._set_breakout_connection(self.breakout_box_instrument.set_connected)

    def disconnect_breakout_lines(self) -> None:
        """Disconnect all breakout box lines.

        This method disconnects all breakout box lines from the instrument.

        Raises:
            DeviceError: If the breakout box instrument is not configured.
        """
        if not self.breakout_box_instrument:
            raise DeviceError("Breakout box instrument not configured")
        self._set_breakout_connection(self.breakout_box_instrument.set_disconnected)

    def _get_lines_by_channel(
        self, has_channel: Callable[[ChannelConfig], bool]
    ) -> list[str]:
        """Filter breakout box lines by channel type.

        This method filters the breakout box lines by the channel type.

        Args:
            has_channel: A callable that returns True if the channel has the specified type.

        Returns:
            A list of breakout box lines that have the specified channel type.
        """
        return [
            line
            for line in self.breakout_lines
            if has_channel(self.channel_configs[line])
        ]

    def _apply_breakout_connection(
        self,
        lines: list[str],
        instrument: ControlInstrument | MeasurementInstrument | None,
        instrument_name: str,
        connection_method: Callable[[list[str], int], None],
    ) -> None:
        """Apply connection method to a list of lines for a specific instrument.

        This method applies the connection method to a list of lines for a specific instrument.

        Args:
            lines: A list of breakout box lines to apply the connection method to.
            instrument: The instrument to apply the connection method to.
            instrument_name: The name of the instrument.
            connection_method: The connection method to apply.
        """
        if not lines:
            return
        if not instrument:
            raise DeviceError(f"{instrument_name} instrument not configured")
        breakout_line = cast(
            BaseControlInstrument | BaseMeasurementInstrument, instrument
        ).instrument_config.breakout_line
        if breakout_line is None:
            raise DeviceError(
                f"{instrument_name} instrument breakout line not configured"
            )
        connection_method(lines, breakout_line)

    def _set_breakout_connection(
        self, connection_method: Callable[[list[str], int], None]
    ) -> None:
        """Set breakout box connection state for all lines, grouped by instrument type.

        This method sets the breakout box connection state for all lines, grouped by instrument type.

        Args:
            connection_method: The connection method to apply to the control and measurement instruments.
        """
        control_lines = self._get_lines_by_channel(
            lambda c: c.control_channel is not None
        )
        measurement_lines = self._get_lines_by_channel(
            lambda c: c.measure_channel is not None
        )

        all_instrument_lines = set(control_lines) | set(measurement_lines)
        orphan_lines = set(self.breakout_lines) - all_instrument_lines
        if orphan_lines:
            raise DeviceError(
                f"Breakout box line {orphan_lines.pop()} has no associated instrument channel"
            )

        self._apply_breakout_connection(
            control_lines, self.control_instrument, "Control", connection_method
        )
        self._apply_breakout_connection(
            measurement_lines,
            self.measurement_instrument,
            "Measurement",
            connection_method,
        )
