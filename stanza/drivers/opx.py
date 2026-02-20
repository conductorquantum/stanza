from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property

# mypy: disable-error-code="union-attr,attr-defined"
from typing import Any

import numpy as np

from stanza.base.channels import ChannelConfig, MeasurementChannel
from stanza.base.instruments import BaseMeasurementInstrument
from stanza.drivers.opx_config_builder import OPXConfigBuilder
from stanza.drivers.utils import demod2volts, wait_until_job_is_paused
from stanza.exceptions import InstrumentError
from stanza.models import MeasurementInstrumentConfig
from stanza.pulses import PulseDefinition, PulseRegistry
from stanza.timing import seconds_to_ns
from stanza.triggers import TriggerConfig, TriggerLink, TriggerMode

try:
    from qm import FullQuaConfig, Program, QuantumMachinesManager
    from qm.qua import (
        FUNCTIONS,
        IO1,
        assign,
        declare,
        declare_stream,
        else_,
        fixed,
        for_,
        if_,
        infinite_loop_,
        integration,
        measure,
        pause,
        play,
        program,
        save,
        stream_processing,
        wait,
    )

    HAS_QM = True
except ImportError:
    QuantumMachinesManager = None  # type: ignore[misc,assignment]
    Program = None  # type: ignore[misc,assignment]
    FullQuaConfig = None  # type: ignore[misc,assignment]
    RunningQmJob = None
    HAS_QM = False

logger = logging.getLogger(__name__)


# --- Pulse Sequencing ---


@dataclass(frozen=True)
class PulseStep:
    """A single step in a pulse sequence.

    Attributes:
        pulse_name: Must exist in the PulseRegistry.
        element: QUA element to play on.
        wait_after_cycles: Wait after this step in QUA cycles (0 = immediate).
    """

    pulse_name: str
    element: str
    wait_after_cycles: int = 0

    def __post_init__(self) -> None:
        if self.wait_after_cycles < 0:
            raise ValueError(
                f"wait_after_cycles must be non-negative, got {self.wait_after_cycles}"
            )


PulseSequence = list[PulseStep]


class OPXMeasurementChannel(MeasurementChannel):
    """OPX-specific measurement channel with hardware integration."""

    def __init__(self, name: str, channel_id: int, config: ChannelConfig):
        self.name = name
        self.channel_id = channel_id
        self.driver = None
        self.count = 0

        self.job_id: int | None = None
        self.read_len: int | None = None
        super().__init__(config)

    def set_driver(self, driver: Any) -> None:
        self.driver = driver

    def set_job_id(self, job_id: int) -> None:
        self.job_id = job_id

    def set_read_len(self, read_len: int) -> None:
        self.read_len = read_len

    def get_current(self) -> float:
        if getattr(self, "driver", None) is None:
            raise InstrumentError("OPX driver not set")

        if self.job_id is None:
            raise InstrumentError("job_id not set")

        if self.read_len is None:
            raise InstrumentError("read_len not set")

        job = self.driver.get_job(self.job_id)
        handle_name = f"measure_{self.name}"
        h = job.result_handles.get(handle_name)

        if h is None:
            raise InstrumentError(f"No output handle {handle_name}")

        prev = getattr(self, "count", 0)
        index = prev + 1

        job.resume()
        wait_until_job_is_paused(job)

        try:
            h.wait_for_values(index, timeout=10)
        except Exception:
            pass

        raw = h.fetch(index)
        val = demod2volts(raw, self.read_len, single_demod=True)

        self.count = index
        return float(-val)


class OPXInstrument(BaseMeasurementInstrument):
    """OPX measurement instrument using OPXConfigBuilder for config generation."""

    def __init__(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ):
        """Initialize OPXInstrument with the given configuration.

        Expected extra fields on instrument_config (passed via Pydantic extra="allow"):
            machine_type (str): OPX machine type (e.g. "OPX1000")
            cluster_name (str): OPX cluster name
            measurement_channels (list[int]): OPX measurement channels
            connection_headers (dict[str, str]): OPX connection headers
            octave (str | None): OPX octave configuration
        """
        if not HAS_QM:
            raise ImportError(
                "qm is not installed. Install with: pip install stanza[qm]"
            )

        super().__init__(instrument_config)

        self.host = instrument_config.ip_addr
        self.port = instrument_config.port
        self.machine_type = getattr(instrument_config, "machine_type", None)
        self.cluster_name = getattr(instrument_config, "cluster_name", None)
        self.connection_headers = getattr(instrument_config, "connection_headers", None)
        self.measurement_channels = getattr(
            instrument_config, "measurement_channels", None
        )

        self.channel_configs = channel_configs

        self.read_len = seconds_to_ns(instrument_config.sample_time)
        self.measurement_duration = seconds_to_ns(
            instrument_config.measurement_duration
        )
        self.measure_number = max(1, int(self.measurement_duration / self.read_len))
        self.octave = getattr(instrument_config, "octave", None)
        self.qmm = QuantumMachinesManager(
            host=self.host,
            port=self.port,
            connection_headers=self.connection_headers,
            cluster_name=self.cluster_name,
            octave=self.octave,
        )
        self._initialize_channels(channel_configs)
        self.driver = self.qmm.open_qm(self.qua_config)

    def _initialize_channels(self, channel_configs: dict[str, ChannelConfig]) -> None:
        for channel_config in channel_configs.values():
            if (
                channel_config.measure_channel is not None
                and self.measurement_channels is not None
                and channel_config.measure_channel in self.measurement_channels
            ):
                self.add_channel(
                    f"measure_{channel_config.name}",
                    OPXMeasurementChannel(
                        channel_config.name,
                        channel_config.measure_channel,
                        channel_config,
                    ),
                )

    @cached_property
    def qua_config(self) -> FullQuaConfig:
        builder = OPXConfigBuilder(machine_type=self.machine_type)
        builder.add_fem(slot=2, fem_type="LF")
        builder.add_analog_output(fem_slot=2, port=1)
        builder.add_analog_output(fem_slot=2, port=2)
        builder.add_analog_input(fem_slot=2, port=1, offset=-0.0078)
        builder.add_analog_input(fem_slot=2, port=2, offset=-0.007)

        for channel_name, channel in self.channels.items():
            builder.add_measurement_element(
                name=channel_name,
                port=("con1", 2, channel.channel_id),
                read_len_ns=self.read_len,
            )

        return FullQuaConfig(**builder.build())

    @property
    def qua_program(self) -> Program:
        chans = list(self.channels.keys())
        n_ch = len(chans)
        with program() as prog:
            seg = declare(int)
            acc = declare(fixed, size=n_ch)
            outs = [declare_stream() for _ in range(n_ch)]

            with infinite_loop_():
                pause()
                with for_(seg, 0, seg < self.measure_number, seg + 1):
                    for idx, ch in enumerate(chans):
                        measure(
                            "readout", ch, None, integration.full("const", acc[idx])
                        )
                        save(acc[idx], outs[idx])

            with stream_processing():
                for idx, ch in enumerate(chans):
                    outs[idx].buffer(self.measure_number).map(FUNCTIONS.average()).save(
                        ch
                    )

        return prog

    def prepare_measurement(self) -> None:
        """Prepare the measurement."""
        self.driver.compile(self.qua_program)
        job = self.driver.execute(self.qua_program)

        for channel in self.channels.values():
            channel.set_job_id(job.id)
            channel.set_read_len(self.read_len)
            channel.set_driver(self.driver)

    def teardown_measurement(self) -> None:
        try:
            jobs = self.driver.get_jobs()
            if hasattr(jobs, "__iter__"):
                for job in jobs:
                    try:
                        job.halt()
                    except Exception:
                        logger.warning(f"Failed to halt job {job.id}")
        except Exception:
            pass
        finally:
            try:
                self.driver.close()
            except Exception:
                pass

    def measure(self, channel_name: str) -> float:
        return super().measure(f"measure_{channel_name}")


class OPXPulseController:
    """OPX pulse generation + measurement + triggering.

    Composes PulseRegistry, OPXConfigBuilder, and TriggerConfig into a working
    OPX program. Not a GeneralInstrument — the ControlInstrument protocol
    (set_voltage/get_voltage) doesn't map to pulse operations.
    """

    def __init__(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        pulse_registry: PulseRegistry | None = None,
        trigger_config: TriggerConfig | None = None,
        control_fem: int = 2,
        control_ports: list[int] | None = None,
        trigger_links: list[TriggerLink] | None = None,
    ) -> None:
        if not HAS_QM:
            raise ImportError(
                "qm is not installed. Install with: pip install stanza[qm]"
            )

        self.instrument_config = instrument_config
        self.channel_configs = channel_configs
        self.pulse_registry = pulse_registry or PulseRegistry()
        self.trigger_config = trigger_config or TriggerConfig(mode=TriggerMode.SOFTWARE)
        self.control_fem = control_fem
        self.control_ports = control_ports or [1]
        self.trigger_links = trigger_links or []

        self.host = instrument_config.ip_addr
        self.port = instrument_config.port
        self.machine_type = getattr(instrument_config, "machine_type", None)
        self.cluster_name = getattr(instrument_config, "cluster_name", None)
        self.connection_headers = getattr(instrument_config, "connection_headers", None)
        self.measurement_channels = getattr(
            instrument_config, "measurement_channels", None
        )

        self.read_len = seconds_to_ns(instrument_config.sample_time)
        self.measurement_duration = seconds_to_ns(
            instrument_config.measurement_duration
        )
        self.measure_number = max(1, int(self.measurement_duration / self.read_len))
        self.octave = getattr(instrument_config, "octave", None)

        self._qmm: Any = None
        self._driver: Any = None
        self._job: Any = None
        self._measurement_channels: dict[str, OPXMeasurementChannel] = {}

        self._initialize_measurement_channels()

    def _initialize_measurement_channels(self) -> None:
        for channel_config in self.channel_configs.values():
            if (
                channel_config.measure_channel is not None
                and self.measurement_channels is not None
                and channel_config.measure_channel in self.measurement_channels
            ):
                ch = OPXMeasurementChannel(
                    channel_config.name,
                    channel_config.measure_channel,
                    channel_config,
                )
                self._measurement_channels[channel_config.name] = ch

    def add_pulse(self, pulse: PulseDefinition) -> None:
        """Add a pulse to the registry."""
        self.pulse_registry.add_pulse(pulse)

    def _build_config(self) -> dict:
        """Build QUA config dict using OPXConfigBuilder."""
        builder = OPXConfigBuilder(machine_type=self.machine_type)
        builder.add_fem(slot=self.control_fem, fem_type="LF")

        # Add control output ports
        for port_num in self.control_ports:
            builder.add_analog_output(fem_slot=self.control_fem, port=port_num)

        # Add measurement elements with analog I/O
        for ch_name, ch in self._measurement_channels.items():
            builder.add_analog_input(fem_slot=self.control_fem, port=ch.channel_id)
            # Ensure the output port for measurement element exists
            if (self.control_fem, ch.channel_id) not in builder._analog_outputs:
                builder.add_analog_output(fem_slot=self.control_fem, port=ch.channel_id)
            builder.add_measurement_element(
                name=f"measure_{ch_name}",
                port=("con1", self.control_fem, ch.channel_id),
                read_len_ns=self.read_len,
            )

        # Add digital I/O for trigger if needed
        trigger = self.trigger_config
        if trigger.digital_port is not None:
            _, fem_slot, port_num = trigger.digital_port
            if trigger.mode == TriggerMode.HARDWARE_OUT:
                builder.add_digital_output(fem_slot=fem_slot, port=port_num)
            elif trigger.mode == TriggerMode.HARDWARE_IN:
                builder.add_digital_input(fem_slot=fem_slot, port=port_num)

        # Add pulse registry contents
        if self.pulse_registry.pulse_names:
            builder.add_pulse_registry(self.pulse_registry)

            # Wire pulses as operations on control elements
            for pulse_name in self.pulse_registry.pulse_names:
                pulse = self.pulse_registry.get_pulse(pulse_name)
                if pulse.operation == "control":
                    # Create control element for each pulse/port combo
                    for port_num in self.control_ports:
                        elem_name = f"ctrl_{port_num}"
                        if elem_name not in builder._elements:
                            iq_ports = None
                            if pulse.is_iq and len(self.control_ports) >= 2:
                                iq_ports = (
                                    ("con1", self.control_fem, self.control_ports[0]),
                                    ("con1", self.control_fem, self.control_ports[1]),
                                )
                            builder.add_control_element(
                                name=elem_name,
                                port=("con1", self.control_fem, port_num),
                                operations={},
                                iq_ports=iq_ports if pulse.is_iq else None,
                            )
                        # Add operation to element
                        builder._elements[elem_name].setdefault("operations", {})[
                            pulse_name
                        ] = pulse_name

        # Auto-create trigger elements from TriggerLinks
        for link in self.trigger_links:
            _, fem_slot, port_num = link.source_port
            builder.add_digital_output(fem_slot=fem_slot, port=port_num)

            # Ensure a matching analog output exists for the element
            # (QUA elements require at least a singleInput even for digital-only)
            if (fem_slot, port_num) not in builder._analog_outputs:
                builder.add_analog_output(fem_slot=fem_slot, port=port_num)

            # Register a short digital trigger pulse
            trig_pulse_name = f"__trigger_{link.name}"
            duration_ns = link.trigger_duration_cycles * 4  # cycles to ns
            trig_marker_name = f"__marker_{link.name}"

            if trig_pulse_name not in builder._pulses:
                # Create zero-amplitude analog waveform for the trigger pulse
                if "zero_wf" not in builder._waveforms:
                    builder._waveforms["zero_wf"] = {
                        "type": "constant",
                        "sample": 0.0,
                    }

                # Create digital marker (high for trigger duration)
                builder._digital_waveforms[trig_marker_name] = {
                    "samples": [(1, duration_ns), (0, 0)],
                }

                builder._pulses[trig_pulse_name] = {
                    "operation": "control",
                    "length": duration_ns,
                    "waveforms": {"single": "zero_wf"},
                    "digital_marker": trig_marker_name,
                }

            # Create trigger-only element
            builder.add_element(
                name=link.name,
                input_ports={"single": link.source_port},
                operations={"trig": trig_pulse_name},
            )

            # Wire the digital output to the element
            builder._elements[link.name]["digitalInputs"] = {
                "switch": {
                    "port": link.source_port,
                    "delay": 0,
                    "buffer": 0,
                }
            }

        return builder.build()

    def _sequence_length_cycles(self, sequence: PulseSequence) -> int:
        """Compute total length of a pulse sequence in QUA cycles."""
        total = 0
        for step in sequence:
            pulse = self.pulse_registry.get_pulse(step.pulse_name)
            total += pulse.length_cycles + step.wait_after_cycles
        return total

    def build_program(self, sequence: PulseSequence | None = None) -> Any:
        """Build QUA program for the given sequence and trigger mode.

        Validates:
        - All pulse names in sequence exist in registry
        - All elements in sequence exist in config
        - For TIMED: interval_cycles >= total sequence length
        """
        if sequence is not None:
            # Validate all pulse names exist
            for step in sequence:
                if step.pulse_name not in self.pulse_registry.pulse_names:
                    raise InstrumentError(
                        f"Pulse '{step.pulse_name}' not found in registry. "
                        f"Available: {self.pulse_registry.pulse_names}"
                    )

            # Validate TIMED interval
            if self.trigger_config.mode == TriggerMode.TIMED:
                seq_len = self._sequence_length_cycles(sequence)
                if self.trigger_config.interval_cycles < seq_len:
                    raise InstrumentError(
                        f"TIMED interval ({self.trigger_config.interval_cycles} cycles) "
                        f"is shorter than sequence length ({seq_len} cycles). "
                        f"Increase interval or shorten sequence."
                    )

            # Validate element existence and operation wiring.
            # Cache the config dict to avoid a redundant build() in execute().
            self._cached_config = self._build_config()
            elem_defs = self._cached_config.get("elements", {})
            for step in sequence:
                if step.element not in elem_defs:
                    raise InstrumentError(
                        f"Element '{step.element}' not found in configuration. "
                        f"Available elements: {list(elem_defs.keys())}"
                    )
                # QUA play() dispatches by operation key, and _build_config wires
                # pulse_name as both key and value. Check keys (the play alias).
                ops = elem_defs[step.element].get("operations", {})
                if step.pulse_name not in ops:
                    raise InstrumentError(
                        f"Element '{step.element}' has no operation '{step.pulse_name}'. "
                        f"Available operations: {list(ops.keys())}"
                    )

        mode = self.trigger_config.mode
        measure_chans = [f"measure_{name}" for name in self._measurement_channels]

        if mode == TriggerMode.SOFTWARE:
            return self._build_software_program(sequence, measure_chans)
        elif mode == TriggerMode.TIMED:
            return self._build_timed_program(sequence, measure_chans)
        elif mode == TriggerMode.HARDWARE_OUT:
            return self._build_hardware_out_program(sequence, measure_chans)
        elif mode == TriggerMode.HARDWARE_IN:
            return self._build_hardware_in_program(sequence, measure_chans)
        else:
            raise InstrumentError(f"Unsupported trigger mode: {mode}")

    def _emit_play_sequence(self, sequence: PulseSequence | None) -> None:
        """Emit QUA play statements for a sequence (called inside program context)."""
        if sequence is None:
            return
        for step in sequence:
            play(step.pulse_name, step.element)
            if step.wait_after_cycles > 0:
                wait(step.wait_after_cycles, step.element)

    def _emit_measure_all(
        self,
        measure_chans: list[str],
        acc: Any,
        outs: list[Any],
    ) -> None:
        """Emit QUA measure/save statements for all measurement channels."""
        for idx, ch in enumerate(measure_chans):
            measure("readout", ch, None, integration.full("const", acc[idx]))
            save(acc[idx], outs[idx])

    def _build_software_program(
        self, sequence: PulseSequence | None, measure_chans: list[str]
    ) -> Any:
        n_ch = len(measure_chans)
        with program() as prog:
            seg = declare(int)
            acc = declare(fixed, size=max(n_ch, 1))
            outs = [declare_stream() for _ in range(n_ch)]

            with infinite_loop_():
                pause()
                self._emit_play_sequence(sequence)
                if measure_chans:
                    with for_(seg, 0, seg < self.measure_number, seg + 1):
                        self._emit_measure_all(measure_chans, acc, outs)

            with stream_processing():
                for idx, ch in enumerate(measure_chans):
                    outs[idx].buffer(self.measure_number).map(FUNCTIONS.average()).save(
                        ch
                    )

        return prog

    def _build_timed_program(
        self, sequence: PulseSequence | None, measure_chans: list[str]
    ) -> Any:
        n_ch = len(measure_chans)
        reps = self.trigger_config.repetitions or 1
        interval = self.trigger_config.interval_cycles

        with program() as prog:
            n = declare(int)
            seg = declare(int)
            acc = declare(fixed, size=max(n_ch, 1))
            outs = [declare_stream() for _ in range(n_ch)]

            with for_(n, 0, n < reps, n + 1):
                self._emit_play_sequence(sequence)
                if measure_chans:
                    with for_(seg, 0, seg < self.measure_number, seg + 1):
                        self._emit_measure_all(measure_chans, acc, outs)
                wait(interval)

            with stream_processing():
                for idx, ch in enumerate(measure_chans):
                    outs[idx].buffer(self.measure_number).map(FUNCTIONS.average()).save(
                        ch
                    )

        return prog

    def _build_hardware_out_program(
        self, sequence: PulseSequence | None, measure_chans: list[str]
    ) -> Any:
        # HARDWARE_OUT: same as SOFTWARE but digital marker fires with the first pulse
        # The digital marker is already attached to the pulse definition
        return self._build_software_program(sequence, measure_chans)

    def _build_hardware_in_program(
        self, sequence: PulseSequence | None, measure_chans: list[str]
    ) -> Any:
        n_ch = len(measure_chans)
        timeout = self.trigger_config.input_timeout_cycles
        has_timeout = timeout is not None

        with program() as prog:
            seg = declare(int)
            acc = declare(fixed, size=max(n_ch, 1))
            outs = [declare_stream() for _ in range(n_ch)]
            if has_timeout:
                timed_out = declare(bool)

            with infinite_loop_():
                if has_timeout:
                    # Placeholder trigger detection: waits for `timeout` cycles,
                    # then reads IO1 (host-side flag). A real edge-triggered
                    # implementation should use a digital input polling loop or
                    # QUA's native wait_for_trigger() when available.
                    wait(timeout)
                    with if_(IO1):
                        assign(timed_out, False)
                    with else_():
                        assign(timed_out, True)

                    # On timeout: skip play/measure entirely, save sentinel -1.0
                    # so the stream always produces the same number of entries.
                    with if_(timed_out):
                        with for_(seg, 0, seg < self.measure_number, seg + 1):
                            for idx, _ch in enumerate(measure_chans):
                                assign(acc[idx], -1.0)
                                save(acc[idx], outs[idx])
                    with else_():
                        self._emit_play_sequence(sequence)
                        if measure_chans:
                            with for_(seg, 0, seg < self.measure_number, seg + 1):
                                self._emit_measure_all(measure_chans, acc, outs)
                else:
                    # No timeout — block indefinitely until trigger.
                    # Same placeholder: host sets IO1 to signal trigger arrival.
                    pause()
                    self._emit_play_sequence(sequence)
                    if measure_chans:
                        with for_(seg, 0, seg < self.measure_number, seg + 1):
                            self._emit_measure_all(measure_chans, acc, outs)

            with stream_processing():
                for idx, ch in enumerate(measure_chans):
                    outs[idx].buffer(self.measure_number).map(FUNCTIONS.average()).save(
                        ch
                    )

        return prog

    def execute(self, prog: Any | None = None) -> None:
        """Execute a QUA program on the OPX."""
        if prog is None:
            raise InstrumentError("No program to execute")

        try:
            if self._qmm is None:
                self._qmm = QuantumMachinesManager(
                    host=self.host,
                    port=self.port,
                    connection_headers=self.connection_headers,
                    cluster_name=self.cluster_name,
                    octave=self.octave,
                )
                config = getattr(self, "_cached_config", None) or self._build_config()
                self._cached_config = None  # consume; don't hold stale ref
                self._driver = self._qmm.open_qm(FullQuaConfig(**config))

            self._driver.compile(prog)
            self._job = self._driver.execute(prog)

            for ch in self._measurement_channels.values():
                ch.set_job_id(self._job.id)
                ch.set_read_len(self.read_len)
                ch.set_driver(self._driver)
        except Exception:
            # Ensure resources are cleaned up on failure
            self.teardown()
            raise

    def measure_channel(self, channel_name: str) -> float:
        """Measure a single channel."""
        if channel_name not in self._measurement_channels:
            raise InstrumentError(
                f"Channel '{channel_name}' not found. "
                f"Available: {list(self._measurement_channels.keys())}"
            )
        return self._measurement_channels[channel_name].get_current()

    def play_and_measure(
        self,
        sequence: PulseSequence,
        measure_channels: list[str] | None = None,
        allow_timeouts: bool = False,
    ) -> dict[str, float]:
        """Play a pulse sequence and return measurements.

        For HARDWARE_IN with timeouts: sentinel values (-1.0) in the stream
        indicate skipped iterations. Raises InstrumentError unless
        allow_timeouts=True.
        """
        prog = self.build_program(sequence)
        self.execute(prog)

        channels = measure_channels or list(self._measurement_channels.keys())
        results: dict[str, float] = {}

        for ch_name in channels:
            val = self.measure_channel(ch_name)
            results[ch_name] = val

        # Check for timeout sentinels
        if (
            self.trigger_config.mode == TriggerMode.HARDWARE_IN
            and self.trigger_config.input_timeout_cycles is not None
            and not allow_timeouts
        ):
            timed_out = [k for k, v in results.items() if v == -1.0]
            if timed_out:
                raise InstrumentError(
                    f"Trigger timeout: {len(timed_out)} of {len(results)} "
                    f"channels timed out: {timed_out}"
                )

        return results

    def execute_sweep_1d(
        self,
        trigger_link_name: str,
        n_points: int,
        measure_electrode: str,
        n_avg: int = 1,
        settling_wait_ns: int = 250_000,
    ) -> np.ndarray:
        """Build and execute a 1D triggered sweep QUA program."""
        from stanza.timing import ns_to_cycles

        settling_cycles = ns_to_cycles(settling_wait_ns)
        measure_chan = f"measure_{measure_electrode}"

        with program() as prog:
            n = declare(int)
            i = declare(int)
            acc = declare(fixed, size=1)
            out_stream = declare_stream()

            with for_(n, 0, n < n_avg, n + 1):
                with for_(i, 0, i < n_points, i + 1):
                    play("trig", trigger_link_name)
                    wait(settling_cycles, measure_chan)
                    measure(
                        "readout",
                        measure_chan,
                        None,
                        integration.full("const", acc[0]),
                    )
                    save(acc[0], out_stream)

            with stream_processing():
                if n_avg > 1:
                    out_stream.buffer(n_points).average().save("results")
                else:
                    out_stream.buffer(n_points).save("results")

        self.execute(prog)

        job = self._job
        handle = job.result_handles.get("results")
        handle.wait_for_values(1)
        raw = handle.fetch_all()

        return np.array(raw, dtype=float)

    def execute_sweep_2d(
        self,
        outer_trigger_name: str,
        inner_trigger_name: str,
        n_outer: int,
        n_inner: int,
        measure_electrode: str,
        n_avg: int = 1,
        settling_wait_ns: int = 250_000,
    ) -> np.ndarray:
        """Build and execute a 2D triggered sweep QUA program."""
        from stanza.timing import ns_to_cycles

        settling_cycles = ns_to_cycles(settling_wait_ns)
        measure_chan = f"measure_{measure_electrode}"

        with program() as prog:
            n = declare(int)
            i = declare(int)
            j = declare(int)
            acc = declare(fixed, size=1)
            out_stream = declare_stream()

            with for_(n, 0, n < n_avg, n + 1):
                with for_(i, 0, i < n_outer, i + 1):
                    play("trig", outer_trigger_name)
                    with for_(j, 0, j < n_inner, j + 1):
                        play("trig", inner_trigger_name)
                        wait(settling_cycles, measure_chan)
                        measure(
                            "readout",
                            measure_chan,
                            None,
                            integration.full("const", acc[0]),
                        )
                        save(acc[0], out_stream)

            with stream_processing():
                total = n_outer * n_inner
                if n_avg > 1:
                    out_stream.buffer(total).average().save("results")
                else:
                    out_stream.buffer(total).save("results")

        self.execute(prog)

        job = self._job
        handle = job.result_handles.get("results")
        handle.wait_for_values(1)
        raw = handle.fetch_all()

        return np.array(raw, dtype=float).reshape(n_outer, n_inner)

    def teardown(self) -> None:
        """Halt all jobs and close the connection."""
        if self._driver is not None:
            try:
                jobs = self._driver.get_jobs()
                if hasattr(jobs, "__iter__"):
                    for job in jobs:
                        try:
                            job.halt()
                        except Exception:
                            logger.warning(f"Failed to halt job {job.id}")
            except Exception:
                pass
            finally:
                try:
                    self._driver.close()
                except Exception:
                    pass

        self._driver = None
        self._job = None
