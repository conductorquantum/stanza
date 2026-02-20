"""Hardware-accelerated sweep orchestration across multiple instruments.

Coordinates pre-loading, execution, and result collection across instrument
boundaries. Driver-agnostic: works against ListSweepInstrument and
HardwareSweepController protocols rather than specific instrument classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from stanza.base.protocols import HardwareSweepController, ListSweepInstrument
from stanza.triggers import TriggerLink

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepAxis:
    """One axis of a hardware sweep.

    Attributes:
        gate: Pad/electrode name (resolved to instrument channel).
        voltages: Voltage array for this axis.
        trigger_link: Which TriggerLink steps this axis.
    """

    gate: str
    voltages: np.ndarray
    trigger_link: TriggerLink


class SweepOrchestrator:
    """Coordinates hardware-triggered sweeps across multiple instruments.

    Replaces the Python loop in Device.sweep_nd with autonomous
    hardware execution for supported instrument combinations.
    """

    def __init__(
        self,
        list_sweep_instrument: ListSweepInstrument,
        sweep_controller: HardwareSweepController,
        trigger_links: list[TriggerLink],
    ) -> None:
        self.list_sweep_instrument = list_sweep_instrument
        self.sweep_controller = sweep_controller
        self.trigger_links = {link.name: link for link in trigger_links}

    def _validate_axes(self, axes: list[SweepAxis]) -> None:
        """Validate that all axes have registered trigger links."""
        for axis in axes:
            if axis.trigger_link.name not in self.trigger_links:
                raise ValueError(
                    f"Trigger link '{axis.trigger_link.name}' not registered. "
                    f"Available: {list(self.trigger_links.keys())}"
                )

    def prepare(self, axes: list[SweepAxis]) -> None:
        """Pre-load all voltage lists to their respective instruments."""
        self._validate_axes(axes)
        for axis in axes:
            link = axis.trigger_link
            self.list_sweep_instrument.load_voltage_list(
                channel_name=axis.gate,
                voltages=axis.voltages,
                trigger_port=link.sink_trigger_port,
            )

    def teardown(self, axes: list[SweepAxis]) -> None:
        """Reset all channels to fixed-voltage mode."""
        for axis in axes:
            try:
                self.list_sweep_instrument.reset_voltage_list(axis.gate)
            except Exception:
                logger.warning(f"Failed to reset voltage list for {axis.gate}")

    def sweep_1d(
        self,
        axis: SweepAxis,
        measure_electrode: str,
        n_avg: int = 1,
        settling_wait_ns: int = 250_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute a 1D hardware-triggered sweep.

        Args:
            axis: Sweep axis definition.
            measure_electrode: Name of the measurement electrode.
            n_avg: Number of averaging repetitions.
            settling_wait_ns: Wait time after trigger in nanoseconds.

        Returns:
            (voltages, currents) arrays.
        """
        axes = [axis]
        self._validate_axes(axes)

        try:
            self.prepare(axes)
            currents = self.sweep_controller.execute_sweep_1d(
                trigger_link_name=axis.trigger_link.name,
                n_points=len(axis.voltages),
                measure_electrode=measure_electrode,
                n_avg=n_avg,
                settling_wait_ns=settling_wait_ns,
            )
            return axis.voltages.copy(), currents
        finally:
            self.teardown(axes)

    def sweep_2d(
        self,
        outer_axis: SweepAxis,
        inner_axis: SweepAxis,
        measure_electrode: str,
        n_avg: int = 1,
        settling_wait_ns: int = 250_000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute a 2D hardware-triggered sweep.

        Args:
            outer_axis: Outer sweep axis.
            inner_axis: Inner sweep axis.
            measure_electrode: Name of the measurement electrode.
            n_avg: Number of averaging repetitions.
            settling_wait_ns: Wait time after trigger in nanoseconds.

        Returns:
            (outer_voltages, inner_voltages, currents_2d) arrays.
        """
        axes = [outer_axis, inner_axis]
        self._validate_axes(axes)

        try:
            self.prepare(axes)
            currents = self.sweep_controller.execute_sweep_2d(
                outer_trigger_name=outer_axis.trigger_link.name,
                inner_trigger_name=inner_axis.trigger_link.name,
                n_outer=len(outer_axis.voltages),
                n_inner=len(inner_axis.voltages),
                measure_electrode=measure_electrode,
                n_avg=n_avg,
                settling_wait_ns=settling_wait_ns,
            )
            return (
                outer_axis.voltages.copy(),
                inner_axis.voltages.copy(),
                currents,
            )
        finally:
            self.teardown(axes)
