"""Programmatic QUA config builder for OPX instruments.

OPX-specific. Translates driver-agnostic domain objects (PulseRegistry,
PulseDefinition) into QUA config dicts. Replaces the JSON template approach.

OPX+ (OPX1000, FEM-based) is first-class. OPX1 is a gated compatibility path.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from stanza.pulses import PulseRegistry
from stanza.timing import cycles_to_ns, ns_to_cycles

Port = tuple[str, int, int]  # (controller_name, fem_slot, port_number)


class OPXConfigBuilder:
    """Builds QUA config dicts programmatically."""

    def __init__(
        self,
        machine_type: str,
        controller_name: str = "con1",
        opx_version: Literal["opx_plus", "opx1"] = "opx_plus",
    ) -> None:
        self._machine_type = machine_type
        self._controller_name = controller_name
        self._opx_version = opx_version

        # Internal tracking for validation
        self._fems: dict[int, dict[str, Any]] = {}
        self._analog_outputs: dict[
            tuple[int, int], dict[str, Any]
        ] = {}  # (fem, port) -> config
        self._analog_inputs: dict[tuple[int, int], dict[str, Any]] = {}
        self._digital_outputs: dict[tuple[int, int], bool] = {}
        self._digital_inputs: dict[tuple[int, int], bool] = {}
        self._elements: dict[str, dict[str, Any]] = {}
        self._pulses: dict[str, dict[str, Any]] = {}
        self._waveforms: dict[str, dict[str, Any]] = {}
        self._digital_waveforms: dict[str, dict[str, Any]] = {}
        self._integration_weights: dict[str, dict[str, Any]] = {}

        # Track element input types for IQ validation
        self._element_input_type: dict[
            str, str
        ] = {}  # element_name -> "single" | "mix"

        # Track measurement element read lengths for weight validation
        self._element_read_len_cycles: dict[str, int] = {}

    # --- FEM and Port Setup ---

    def add_fem(self, slot: int, fem_type: str = "LF") -> OPXConfigBuilder:
        """Add a Front-End Module. No-op for OPX1."""
        if self._opx_version == "opx1":
            return self
        self._fems[slot] = {"type": fem_type}
        return self

    def add_analog_output(
        self,
        fem_slot: int,
        port: int,
        offset: float = 0.0,
        sampling_rate: float = 1e9,
    ) -> OPXConfigBuilder:
        """Add an analog output port. Auto-creates FEM if needed (OPX+)."""
        if self._opx_version == "opx_plus" and fem_slot not in self._fems:
            self.add_fem(fem_slot)
        self._analog_outputs[(fem_slot, port)] = {
            "offset": offset,
            "sampling_rate": sampling_rate,
        }
        return self

    def add_analog_input(
        self,
        fem_slot: int,
        port: int,
        offset: float = 0.0,
        sampling_rate: float = 1e9,
    ) -> OPXConfigBuilder:
        """Add an analog input port. Auto-creates FEM if needed (OPX+)."""
        if self._opx_version == "opx_plus" and fem_slot not in self._fems:
            self.add_fem(fem_slot)
        self._analog_inputs[(fem_slot, port)] = {
            "offset": offset,
            "sampling_rate": sampling_rate,
        }
        return self

    def add_digital_output(self, fem_slot: int, port: int) -> OPXConfigBuilder:
        """Add a digital output port."""
        if self._opx_version == "opx_plus" and fem_slot not in self._fems:
            self.add_fem(fem_slot)
        self._digital_outputs[(fem_slot, port)] = True
        return self

    def add_digital_input(self, fem_slot: int, port: int) -> OPXConfigBuilder:
        """Add a digital input port."""
        if self._opx_version == "opx_plus" and fem_slot not in self._fems:
            self.add_fem(fem_slot)
        self._digital_inputs[(fem_slot, port)] = True
        return self

    # --- Elements ---

    def add_element(
        self,
        name: str,
        input_ports: dict[str, Port] | None = None,
        output_ports: dict[str, Port] | None = None,
        operations: dict[str, str] | None = None,
        intermediate_frequency: float = 0,
        time_of_flight: int = 28,
        smearing: int = 0,
        digital_inputs: dict[str, dict] | None = None,
    ) -> OPXConfigBuilder:
        """Add a QUA element."""
        element: dict[str, Any] = {
            "intermediate_frequency": intermediate_frequency,
            "time_of_flight": time_of_flight,
            "smearing": smearing,
        }

        if input_ports is not None:
            if "single" in input_ports:
                element["singleInput"] = {"port": input_ports["single"]}
                self._element_input_type[name] = "single"
            elif "I" in input_ports and "Q" in input_ports:
                element["mixInputs"] = {
                    "I": input_ports["I"],
                    "Q": input_ports["Q"],
                }
                self._element_input_type[name] = "mix"

        if output_ports is not None:
            element["outputs"] = dict(output_ports)

        if operations is not None:
            element["operations"] = dict(operations)

        if digital_inputs is not None:
            element["digitalInputs"] = dict(digital_inputs)

        self._elements[name] = element
        return self

    def add_measurement_element(
        self,
        name: str,
        port: Port,
        read_len_ns: int,
    ) -> OPXConfigBuilder:
        """Convenience: measurement element with readout pulse, zero waveform, const weights."""
        read_len_cycles = ns_to_cycles(read_len_ns)

        # Auto-create readout infrastructure if not present
        if "readout_pulse" not in self._pulses:
            self._pulses["readout_pulse"] = {
                "operation": "measurement",
                "length": read_len_ns,
                "waveforms": {"single": "zero_wf"},
                "integration_weights": {"const": "const_weights"},
            }

        if "zero_wf" not in self._waveforms:
            self._waveforms["zero_wf"] = {"type": "constant", "sample": 0.0}

        if "const_weights" not in self._integration_weights:
            name_w, cosine, sine = self.const_integration_weights(read_len_cycles)
            self._integration_weights[name_w] = {
                "cosine": [[c, cycles_to_ns(d)] for c, d in cosine],
                "sine": [[s, cycles_to_ns(d)] for s, d in sine],
            }

        self._element_read_len_cycles[name] = read_len_cycles

        return self.add_element(
            name=name,
            input_ports={"single": port},
            output_ports={"out1": port},
            operations={"readout": "readout_pulse"},
        )

    def add_control_element(
        self,
        name: str,
        port: Port,
        operations: dict[str, str] | None = None,
        digital_input_port: Port | None = None,
        iq_ports: tuple[Port, Port] | None = None,
    ) -> OPXConfigBuilder:
        """Convenience: control element for pulse output.

        If iq_ports is provided, creates a mixInputs element with (I_port, Q_port).
        Otherwise creates a singleInput element.
        """
        digital_inputs = None
        if digital_input_port is not None:
            digital_inputs = {
                "trigger": {
                    "port": digital_input_port,
                    "delay": 0,
                    "buffer": 0,
                }
            }

        if iq_ports is not None:
            input_ports: dict[str, Port] = {"I": iq_ports[0], "Q": iq_ports[1]}
        else:
            input_ports = {"single": port}

        return self.add_element(
            name=name,
            input_ports=input_ports,
            operations=operations or {},
            digital_inputs=digital_inputs,
        )

    # --- Pulses and Waveforms ---

    def add_pulse_registry(self, registry: PulseRegistry) -> OPXConfigBuilder:
        """Convert PulseRegistry contents into QUA config entries.

        Deduplicates waveforms via registry.waveform_key(). Emits each unique
        waveform only once.
        """
        emitted_waveforms: set[str] = set(self._waveforms.keys())

        for pulse_name in registry.pulse_names:
            pulse = registry.get_pulse(pulse_name)

            # Resolve deduped waveform keys
            key_i = registry.waveform_key(pulse.waveform_i)
            if key_i not in emitted_waveforms:
                self._waveforms[key_i] = {
                    "type": "arbitrary",
                    "samples": pulse.waveform_i.tolist(),
                }
                emitted_waveforms.add(key_i)

            pulse_config: dict[str, Any] = {
                "operation": pulse.operation,
                "length": pulse.length_ns,
            }

            if pulse.is_iq:
                key_q = registry.waveform_key(pulse.waveform_q)
                if key_q not in emitted_waveforms:
                    self._waveforms[key_q] = {
                        "type": "arbitrary",
                        "samples": pulse.waveform_q.tolist(),
                    }
                    emitted_waveforms.add(key_q)
                pulse_config["waveforms"] = {"I": key_i, "Q": key_q}
            else:
                pulse_config["waveforms"] = {"single": key_i}

            if pulse.digital_marker is not None:
                pulse_config["digital_marker"] = pulse.digital_marker

            self._pulses[pulse_name] = pulse_config

        # Emit digital markers
        for marker_name in registry.marker_names:
            marker = registry.get_marker(marker_name)
            self._digital_waveforms[marker_name] = {
                "samples": [(v, d) for v, d in marker.samples],
            }

        return self

    def add_integration_weights(
        self,
        name: str,
        cosine: list[tuple[float, int]],
        sine: list[tuple[float, int]],
    ) -> OPXConfigBuilder:
        """Add integration weights. Durations are in QUA clock cycles."""
        self._integration_weights[name] = {
            "cosine": [[c, cycles_to_ns(d)] for c, d in cosine],
            "sine": [[s, cycles_to_ns(d)] for s, d in sine],
        }
        return self

    @staticmethod
    def const_integration_weights(
        read_len_cycles: int,
    ) -> tuple[str, list[tuple[float, int]], list[tuple[float, int]]]:
        """Canonical factory for constant integration weights.

        Returns (name, cosine, sine) suitable for add_integration_weights().
        """
        return (
            "const_weights",
            [(1.0, read_len_cycles)],
            [(0.0, read_len_cycles)],
        )

    # --- Build ---

    def build(self) -> dict:
        """Return the complete QUA config dict.

        Validates:
        1. Element operations reference defined pulses
        2. Pulse waveforms reference defined waveform entries
        3. Element ports reference defined FEM ports
        4. IQ/element type parity (both directions)
        5. Digital markers referenced by pulses are defined
        6. Integration weights duration matches read_len for measurement elements
        7. Port tuples reference valid FEM slots
        """
        self._validate()

        if self._opx_version == "opx_plus":
            controller = self._emit_opx_plus_controller()
        else:
            controller = self._emit_opx1_controller()

        config: dict[str, Any] = {
            "version": "1",
            "controllers": {self._controller_name: controller},
            "elements": copy.deepcopy(self._elements),
            "pulses": copy.deepcopy(self._pulses),
            "waveforms": copy.deepcopy(self._waveforms),
            "integration_weights": copy.deepcopy(self._integration_weights),
        }

        if self._digital_waveforms:
            config["digital_waveforms"] = copy.deepcopy(self._digital_waveforms)

        return config

    # --- Validation ---

    def _validate(self) -> None:
        """Run all cross-reference validations."""
        self._validate_element_operations()
        self._validate_pulse_waveforms()
        self._validate_element_ports()
        self._validate_iq_parity()
        self._validate_digital_markers()
        self._validate_integration_weights()

    def _validate_element_operations(self) -> None:
        for elem_name, elem in self._elements.items():
            ops = elem.get("operations", {})
            for op_name, pulse_name in ops.items():
                if pulse_name not in self._pulses:
                    raise ValueError(
                        f"Element '{elem_name}' operation '{op_name}' references "
                        f"undefined pulse '{pulse_name}'. "
                        f"Defined pulses: {list(self._pulses.keys())}"
                    )

    def _validate_pulse_waveforms(self) -> None:
        for pulse_name, pulse in self._pulses.items():
            wfs = pulse.get("waveforms", {})
            for wf_role, wf_name in wfs.items():
                if wf_name not in self._waveforms:
                    raise ValueError(
                        f"Pulse '{pulse_name}' waveform '{wf_role}' references "
                        f"undefined waveform '{wf_name}'. "
                        f"Defined waveforms: {list(self._waveforms.keys())}"
                    )

    def _validate_element_ports(self) -> None:
        if self._opx_version == "opx1":
            return  # OPX1 port validation is less strict

        for elem_name, elem in self._elements.items():
            # Check singleInput port
            single_input = elem.get("singleInput")
            if single_input is not None:
                port = single_input["port"]
                self._check_port_exists(elem_name, port, "analog output")

            # Check mixInputs ports
            mix_inputs = elem.get("mixInputs")
            if mix_inputs is not None:
                for channel in ("I", "Q"):
                    if channel in mix_inputs:
                        self._check_port_exists(
                            elem_name, mix_inputs[channel], "analog output"
                        )

            # Check output ports
            outputs = elem.get("outputs", {})
            for _, port in outputs.items():
                self._check_port_exists(elem_name, port, "analog input")

    def _check_port_exists(self, elem_name: str, port: tuple, port_type: str) -> None:
        if len(port) != 3:
            return  # Skip non-standard port tuples
        _, fem_slot, port_num = port
        if port_type == "analog output":
            if (fem_slot, port_num) not in self._analog_outputs:
                raise ValueError(
                    f"Element '{elem_name}' references {port_type} port {port} "
                    f"which has not been configured. "
                    f"Call add_analog_output(fem_slot={fem_slot}, port={port_num}) first."
                )
        elif port_type == "analog input":
            if (fem_slot, port_num) not in self._analog_inputs:
                raise ValueError(
                    f"Element '{elem_name}' references {port_type} port {port} "
                    f"which has not been configured. "
                    f"Call add_analog_input(fem_slot={fem_slot}, port={port_num}) first."
                )

    def _validate_iq_parity(self) -> None:
        for elem_name, elem in self._elements.items():
            ops = elem.get("operations", {})
            input_type = self._element_input_type.get(elem_name)
            if input_type is None:
                continue

            for _, pulse_name in ops.items():
                pulse = self._pulses.get(pulse_name)
                if pulse is None:
                    continue  # Already caught by _validate_element_operations

                wfs = pulse.get("waveforms", {})
                pulse_is_iq = "I" in wfs and "Q" in wfs

                if pulse_is_iq and input_type == "single":
                    raise ValueError(
                        f"Pulse '{pulse_name}' is IQ but element '{elem_name}' uses singleInput. "
                        f"Use add_control_element(iq_ports=(port_i, port_q)) to create a mixInputs element."
                    )
                if not pulse_is_iq and input_type == "mix":
                    raise ValueError(
                        f"Pulse '{pulse_name}' is single-channel but element '{elem_name}' uses mixInputs. "
                        f"Use a single-channel element or provide an IQ pulse."
                    )

    def _validate_digital_markers(self) -> None:
        for pulse_name, pulse in self._pulses.items():
            marker_name = pulse.get("digital_marker")
            if marker_name is not None and marker_name not in self._digital_waveforms:
                raise ValueError(
                    f"Pulse '{pulse_name}' references digital marker '{marker_name}' "
                    f"which is not defined. "
                    f"Defined markers: {list(self._digital_waveforms.keys())}"
                )

    def _validate_integration_weights(self) -> None:
        for elem_name, read_len_cycles in self._element_read_len_cycles.items():
            elem = self._elements.get(elem_name)
            if elem is None:
                continue
            ops = elem.get("operations", {})
            for _, pulse_name in ops.items():
                pulse = self._pulses.get(pulse_name)
                if pulse is None:
                    continue
                iw = pulse.get("integration_weights", {})
                for _, iw_ref in iw.items():
                    weights = self._integration_weights.get(iw_ref)
                    if weights is None:
                        continue
                    # Sum cosine durations (stored as ns in the config)
                    cosine_total_ns = sum(d for _, d in weights["cosine"])
                    expected_ns = cycles_to_ns(read_len_cycles)
                    if cosine_total_ns != expected_ns:
                        raise ValueError(
                            f"Integration weights '{iw_ref}' total duration "
                            f"{cosine_total_ns} ns does not match read_len "
                            f"{expected_ns} ns for element '{elem_name}'."
                        )

    # --- Controller Emission ---

    def _emit_opx_plus_controller(self) -> dict[str, Any]:
        controller: dict[str, Any] = {
            "type": self._machine_type,
            "fems": {},
        }

        # Group ports by FEM slot
        for slot, fem_config in self._fems.items():
            fem: dict[str, Any] = {"type": fem_config["type"]}

            # Analog outputs for this FEM
            ao = {
                str(port): cfg
                for (s, port), cfg in self._analog_outputs.items()
                if s == slot
            }
            if ao:
                fem["analog_outputs"] = ao

            # Analog inputs for this FEM
            ai = {
                str(port): cfg
                for (s, port), cfg in self._analog_inputs.items()
                if s == slot
            }
            if ai:
                fem["analog_inputs"] = ai

            # Digital outputs
            do = {str(port): {} for (s, port) in self._digital_outputs if s == slot}
            if do:
                fem["digital_outputs"] = do

            # Digital inputs
            di = {str(port): {} for (s, port) in self._digital_inputs if s == slot}
            if di:
                fem["digital_inputs"] = di

            controller["fems"][str(slot)] = fem

        return controller

    def _emit_opx1_controller(self) -> dict[str, Any]:
        controller: dict[str, Any] = {"type": self._machine_type}

        # Flatten ports (ignore FEM slot for OPX1)
        ao = {str(port): cfg for (_, port), cfg in self._analog_outputs.items()}
        if ao:
            controller["analog_outputs"] = ao

        ai = {str(port): cfg for (_, port), cfg in self._analog_inputs.items()}
        if ai:
            controller["analog_inputs"] = ai

        do = {str(port): {} for (_, port) in self._digital_outputs}
        if do:
            controller["digital_outputs"] = do

        di = {str(port): {} for (_, port) in self._digital_inputs}
        if di:
            controller["digital_inputs"] = di

        return controller
