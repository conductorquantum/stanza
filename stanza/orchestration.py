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


def hardware_sweep(
    axes: list[SweepAxis],
    measure_electrode: str,
    voltage_source: ListSweepInstrument,
    controller: HardwareSweepController,
    n_avg: int = 1,
    settling_wait_ns: int = 250_000,
) -> tuple[np.ndarray, ...]:
    """Execute a hardware-accelerated voltage sweep.

    Pre-loads voltage lists onto the voltage source instrument, runs
    the sweep program on the controller, and resets channels afterward.

    Args:
        axes: 1 or 2 SweepAxis definitions.
        measure_electrode: Electrode to measure.
        voltage_source: Instrument that pre-loads voltage lists (e.g. QDAC).
        controller: Instrument that runs the sweep program (e.g. OPX).
        n_avg: Number of averaging repetitions.
        settling_wait_ns: Wait time after trigger before measurement, in nanoseconds.

    Returns:
        1D: (voltages, currents)
        2D: (outer_voltages, inner_voltages, currents_2d)

    Raises:
        ValueError: If number of axes is not 1 or 2.
    """
    if len(axes) not in (1, 2):
        raise ValueError(f"hardware_sweep supports 1 or 2 axes, got {len(axes)}")

    for axis in axes:
        voltage_source.load_voltage_list(
            channel_name=axis.gate,
            voltages=axis.voltages,
            trigger_port=axis.trigger_link.sink_trigger_port,
        )

    try:
        if len(axes) == 1:
            ax = axes[0]
            currents = controller.execute_sweep_1d(
                trigger_link_name=ax.trigger_link.name,
                n_points=len(ax.voltages),
                measure_electrode=measure_electrode,
                n_avg=n_avg,
                settling_wait_ns=settling_wait_ns,
            )
            return ax.voltages.copy(), currents
        else:
            outer, inner = axes
            currents = controller.execute_sweep_2d(
                outer_trigger_name=outer.trigger_link.name,
                inner_trigger_name=inner.trigger_link.name,
                n_outer=len(outer.voltages),
                n_inner=len(inner.voltages),
                measure_electrode=measure_electrode,
                n_avg=n_avg,
                settling_wait_ns=settling_wait_ns,
            )
            return outer.voltages.copy(), inner.voltages.copy(), currents
    finally:
        for axis in axes:
            try:
                voltage_source.reset_voltage_list(axis.gate)
            except Exception:
                logger.warning(f"Failed to reset voltage list for {axis.gate}")
