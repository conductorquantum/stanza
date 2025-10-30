from __future__ import annotations

from dataclasses import dataclass

from stanza.models import TriggerEdge


@dataclass
class SequenceConfig:
    """A channel that steps through values on trigger."""

    pad_name: str
    values: list[float]
    trigger_input: str


@dataclass
class TriggerRoute:
    """A physical trigger connection between two instruments."""

    source_instrument: str
    source_channel: str
    target_instrument: str
    target_channel: str
    edge: TriggerEdge = TriggerEdge.RISING


@dataclass
class TriggerPlan:
    """Complete trigger sequence specification."""

    sequences: list[SequenceConfig]
    trigger_routes: list[TriggerRoute]
    measure_pad: str
    coordinator_instrument: str
    coordinator_trigger_output: str
    trigger_period_us: float = 10.0

    def validate(self) -> None:
        """Validate trigger plan."""


def make_sweep_1d_plan(
    sequencer_instrument: str,
    coordinator_instrument: str,
    control_pad: str,
    voltages: list[float],
    measure_pad: str,
    trigger_out: str = "sync_out",
    trigger_in: str = "ext1",
) -> TriggerPlan:
    """Make a 1D voltage sweep plan.

    Args:
        sequencer_instrument: Instrument that can load sequences and respond to hardware triggers
        coordinator_instrument: Instrument that can generate trigger pulses and coordinate timing
        control_pad: Pad to sequence through voltages
        voltages: List of voltages to sequence through
        measure_pad: Pad to measure from
        trigger_out: Physical trigger output to send triggers from
        trigger_in: Physical trigger input to listen for triggers

    Returns:
        TriggerPlan: A trigger plan for a 1D voltage sweep
    """
    return TriggerPlan(
        sequences=[
            SequenceConfig(
                pad_name=control_pad, values=voltages, trigger_input=trigger_in
            )
        ],
        trigger_routes=[
            TriggerRoute(
                source_instrument=coordinator_instrument,
                source_channel=trigger_out,
                target_instrument=sequencer_instrument,
                target_channel=trigger_in,
            )
        ],
        measure_pad=measure_pad,
        coordinator_instrument=coordinator_instrument,
        coordinator_trigger_output=trigger_out,
    )


def make_sweep_2d_plan(
    sequencer_instrument: str,
    coordinator_instrument: str,
    control_pad_1: str,
    control_pad_2: str,
    voltages_1: list[float],
    voltages_2: list[float],
    measure_pad: str,
    trigger_out: str = "sync_out",
    trigger_in: str = "ext1",
) -> TriggerPlan:
    """Make a 2D voltage sweep plan."""
    return TriggerPlan(
        sequences=[
            SequenceConfig(
                pad_name=control_pad_1, values=voltages_1, trigger_input=trigger_in
            ),
            SequenceConfig(
                pad_name=control_pad_2, values=voltages_2, trigger_input=trigger_in
            ),
        ],
        trigger_routes=[
            TriggerRoute(
                source_instrument=coordinator_instrument,
                source_channel=trigger_out,
                target_instrument=sequencer_instrument,
                target_channel=trigger_in,
            ),
            TriggerRoute(
                source_instrument=coordinator_instrument,
                source_channel=trigger_out,
                target_instrument=sequencer_instrument,
                target_channel=trigger_in,
            ),
        ],
        measure_pad=measure_pad,
        coordinator_instrument=coordinator_instrument,
        coordinator_trigger_output=trigger_out,
    )


# TODO: Implement the trigger executor
# class TriggerExecutor:
#     def execute(self, plan: TriggerPlan) -> None:
#         """Run the trigger plan."""
#         plan.validate()
#         self._load_sequences(plan)
#         self._configure_triggers(plan)
#         self._arm_triggers(plan)

#         measurements = self._run_coordinator(plan)
#         voltages = self._reconstruct_voltages(plan)
#         return voltages, measurements
