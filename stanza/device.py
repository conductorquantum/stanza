import time
from typing import Any, overload

import numpy as np

from stanza.base.channels import ChannelConfig
from stanza.base.protocols import ControlInstrument, MeasurementInstrument
from stanza.exceptions import DeviceError
from stanza.logger.session import LoggerSession
from stanza.models import ContactType, DeviceConfig, GateType, PadType


class Device:
    """
    Class that defines the interface by which voltage sweeps are applied to the device and
    current is measured.
    """

    def __init__(
        self,
        name: str,
        device_config: DeviceConfig,
        channel_configs: dict[str, ChannelConfig],
        control_instrument: Any | None,
        measurement_instrument: Any | None,
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

        self.control_instrument = control_instrument
        self.measurement_instrument = measurement_instrument
        self.channel_configs = channel_configs

    @property
    def gates(self) -> list[str]:
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.GATE
        ]

    @property
    def contacts(self) -> list[str]:
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.CONTACT
        ]

    @property
    def control_gates(self) -> list[str]:
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.GATE and channel.control_channel is not None
        ]

    @property
    def control_contacts(self) -> list[str]:
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.CONTACT
            and channel.control_channel is not None
        ]

    @property
    def measurement_gates(self) -> list[str]:
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.GATE and channel.measure_channel is not None
        ]

    @property
    def measurement_contacts(self) -> list[str]:
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.CONTACT
            and channel.measure_channel is not None
        ]

    def get_gates_by_type(self, gate_type: str | GateType) -> list[str]:
        """Get the gate electrodes of a given type."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.GATE
            and channel.electrode_type == str(gate_type).upper()
        ]

    def get_contacts_by_type(self, contact_type: str | ContactType) -> list[str]:
        """Get the contact electrodes of a given type."""
        return [
            channel.name
            for channel in self.channel_configs.values()
            if channel.pad_type == PadType.CONTACT
            and channel.electrode_type == str(contact_type).upper()
        ]

    def is_configured(self) -> bool:
        """Check if both instruments are configured."""
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
        """Measure the current of the device."""
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
        """Check the current voltage of the device."""
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
        """Sweep a single gate electrode and measure the current of a single contact electrode."""
        voltage_measurements = []
        current_measurements = []

        for i, voltage in enumerate(voltages):
            should_settle = i == 0
            self.jump({gate_electrode: voltage}, wait_for_settling=should_settle)
            voltage_measurements.append(self.check(gate_electrode))
            current_measurements.append(self.measure(measure_electrode))

        if session:
            session.log_sweep(
                name=f"{gate_electrode} sweep",
                x_data=voltage_measurements,
                y_data=current_measurements,
                x_label="Voltage",
                y_label="Current",
                metadata={
                    "gate_electrodes": [gate_electrode],
                    "measure_electrode": measure_electrode,
                },
            )

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
        """Sweep two gate electrodes and measure the current of a single contact electrode."""
        voltage_measurements = []
        current_measurements = []

        for i, voltage_1 in enumerate(voltages_1):
            for j, voltage_2 in enumerate(voltages_2):
                should_settle = i == 0 and j == 0
                self.jump(
                    {gate_1: voltage_1, gate_2: voltage_2},
                    wait_for_settling=should_settle,
                )
                voltage_measurements.append([self.check(gate_1), self.check(gate_2)])
                current_measurements.append(self.measure(measure_electrode))

        if session:
            session.log_sweep(
                name=f"{gate_1} and {gate_2} sweep",
                x_data=voltage_measurements,
                y_data=current_measurements,
                x_label="Voltage",
                y_label="Current",
                metadata={
                    "gate_electrodes": [gate_1, gate_2],
                    "measure_electrode": measure_electrode,
                },
            )
        return voltage_measurements, current_measurements

    def sweep_all(
        self,
        voltages: list[float],
        measure_electrode: str,
        session: LoggerSession | None = None,
    ) -> tuple[list[list[float]], list[float]]:
        """Sweep all gate electrodes and measure the current of a single contact electrode."""
        voltage_measurements = []
        current_measurements = []

        for i, voltage in enumerate(voltages):
            should_settle = i == 0
            self.jump(
                dict.fromkeys(self.control_gates, voltage),
                wait_for_settling=should_settle,
            )
            voltage_measurements.append(
                [self.check(gate) or voltage for gate in self.control_gates]
            )
            current_measurements.append(self.measure(measure_electrode))

        if session:
            session.log_sweep(
                name="all gates sweep",
                x_data=voltage_measurements,
                y_data=current_measurements,
                x_label="Voltage",
                y_label="Current",
                metadata={
                    "gate_electrodes": self.control_gates,
                    "measure_electrode": measure_electrode,
                },
            )
        return voltage_measurements, current_measurements

    def sweep_nd(
        self,
        gate_electrodes: list[str],
        voltages: list[list[float]],
        measure_electrode: str,
        session: LoggerSession | None = None,
    ) -> tuple[list[list[float]], list[float]]:
        """Sweep multiple gate electrodes and measure the current of a single contact electrode."""
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
        """Set all controllable gates and/or controllable contacts to 0V."""
        pads: list[str] = []
        if str(type).upper() == PadType.ALL:
            pads = self.control_gates + self.control_contacts
        elif str(type).upper() == PadType.GATE:
            pads = self.control_gates
        elif str(type).upper() == PadType.CONTACT:
            pads = self.control_contacts
        else:
            raise DeviceError(f"Invalid pad type: {type}")

        gate_voltages = dict.fromkeys(pads, 0.0)
        self.jump(gate_voltages, wait_for_settling=True)

        actual_voltages = self.check(pads)
        if not np.allclose(actual_voltages, [0.0] * len(actual_voltages), atol=1e-6):
            raise DeviceError("Failed to set all controllable pads to 0V")
