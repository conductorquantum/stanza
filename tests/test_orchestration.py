"""Tests for SweepOrchestrator — hardware-accelerated sweep coordination."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from stanza.orchestration import SweepAxis, SweepOrchestrator
from stanza.triggers import TriggerLink

# --- Fixtures ---


def _make_link(name: str = "trig1", port: int = 5, ext: str = "ext1") -> TriggerLink:
    return TriggerLink(
        name=name,
        source_port=("con1", 2, port),
        sink_instrument="qdac",
        sink_channel="gate1",
        sink_trigger_port=ext,
    )


@pytest.fixture
def link() -> TriggerLink:
    return _make_link()


@pytest.fixture
def link2() -> TriggerLink:
    return _make_link(name="trig2", port=6, ext="ext2")


@pytest.fixture
def list_sweep_instrument() -> Mock:
    return Mock()


@pytest.fixture
def sweep_controller() -> Mock:
    return Mock()


@pytest.fixture
def orchestrator(
    list_sweep_instrument: Mock, sweep_controller: Mock, link: TriggerLink
) -> SweepOrchestrator:
    return SweepOrchestrator(
        list_sweep_instrument=list_sweep_instrument,
        sweep_controller=sweep_controller,
        trigger_links=[link],
    )


@pytest.fixture
def orchestrator_2d(
    list_sweep_instrument: Mock,
    sweep_controller: Mock,
    link: TriggerLink,
    link2: TriggerLink,
) -> SweepOrchestrator:
    return SweepOrchestrator(
        list_sweep_instrument=list_sweep_instrument,
        sweep_controller=sweep_controller,
        trigger_links=[link, link2],
    )


# --- SweepAxis Tests ---


class TestSweepAxis:
    def test_creation(self, link: TriggerLink) -> None:
        axis = SweepAxis(
            gate="gate1",
            voltages=np.array([0.0, 0.5, 1.0]),
            trigger_link=link,
        )
        assert axis.gate == "gate1"
        assert len(axis.voltages) == 3
        assert axis.trigger_link is link

    def test_frozen(self, link: TriggerLink) -> None:
        axis = SweepAxis(
            gate="gate1",
            voltages=np.array([0.0]),
            trigger_link=link,
        )
        with pytest.raises(AttributeError):
            axis.gate = "other"  # type: ignore[misc]


# --- SweepOrchestrator Tests ---


class TestSweepOrchestrator:
    def test_prepare_loads_all_axes(
        self,
        orchestrator_2d: SweepOrchestrator,
        list_sweep_instrument: Mock,
        link: TriggerLink,
        link2: TriggerLink,
    ) -> None:
        """prepare() should call load_voltage_list for each axis."""
        axis1 = SweepAxis(
            gate="gate1", voltages=np.array([0.0, 1.0]), trigger_link=link
        )
        axis2 = SweepAxis(
            gate="gate2", voltages=np.array([0.5, 1.5]), trigger_link=link2
        )

        orchestrator_2d.prepare([axis1, axis2])

        assert list_sweep_instrument.load_voltage_list.call_count == 2
        list_sweep_instrument.load_voltage_list.assert_any_call(
            channel_name="gate1",
            voltages=pytest.approx(np.array([0.0, 1.0])),
            trigger_port="ext1",
        )
        list_sweep_instrument.load_voltage_list.assert_any_call(
            channel_name="gate2",
            voltages=pytest.approx(np.array([0.5, 1.5])),
            trigger_port="ext2",
        )

    def test_teardown_resets_all_channels(
        self,
        orchestrator_2d: SweepOrchestrator,
        list_sweep_instrument: Mock,
        link: TriggerLink,
        link2: TriggerLink,
    ) -> None:
        """teardown() should call reset_voltage_list for each axis."""
        axis1 = SweepAxis(gate="gate1", voltages=np.array([0.0]), trigger_link=link)
        axis2 = SweepAxis(gate="gate2", voltages=np.array([0.5]), trigger_link=link2)

        orchestrator_2d.teardown([axis1, axis2])

        assert list_sweep_instrument.reset_voltage_list.call_count == 2
        list_sweep_instrument.reset_voltage_list.assert_any_call("gate1")
        list_sweep_instrument.reset_voltage_list.assert_any_call("gate2")

    def test_validates_trigger_link_exists(
        self,
        orchestrator: SweepOrchestrator,
    ) -> None:
        """Should raise ValueError if trigger link is not registered."""
        unregistered_link = _make_link(name="unknown_link", port=9)
        axis = SweepAxis(
            gate="gate1",
            voltages=np.array([0.0]),
            trigger_link=unregistered_link,
        )
        with pytest.raises(ValueError, match="not registered"):
            orchestrator._validate_axes([axis])

    def test_sweep_1d_preloads_voltage_list(
        self,
        orchestrator: SweepOrchestrator,
        list_sweep_instrument: Mock,
        sweep_controller: Mock,
        link: TriggerLink,
    ) -> None:
        """sweep_1d should call prepare (load_voltage_list) before execution."""
        axis = SweepAxis(
            gate="gate1", voltages=np.array([0.0, 0.5, 1.0]), trigger_link=link
        )

        sweep_controller.execute_sweep_1d.return_value = np.array([0.001, 0.002, 0.003])

        voltages, currents = orchestrator.sweep_1d(axis, "contact1")

        list_sweep_instrument.load_voltage_list.assert_called_once()
        assert len(voltages) == 3
        assert len(currents) == 3

    def test_sweep_1d_delegates_to_sweep_controller(
        self,
        orchestrator: SweepOrchestrator,
        sweep_controller: Mock,
        link: TriggerLink,
    ) -> None:
        """sweep_1d should call sweep_controller.execute_sweep_1d."""
        axis = SweepAxis(gate="gate1", voltages=np.array([0.0, 0.5]), trigger_link=link)

        sweep_controller.execute_sweep_1d.return_value = np.array([0.001, 0.002])

        orchestrator.sweep_1d(axis, "contact1")

        sweep_controller.execute_sweep_1d.assert_called_once_with(
            trigger_link_name="trig1",
            n_points=2,
            measure_electrode="contact1",
            n_avg=1,
            settling_wait_ns=250_000,
        )

    def test_sweep_1d_resets_on_completion(
        self,
        orchestrator: SweepOrchestrator,
        list_sweep_instrument: Mock,
        sweep_controller: Mock,
        link: TriggerLink,
    ) -> None:
        """sweep_1d should reset channels after successful execution."""
        axis = SweepAxis(gate="gate1", voltages=np.array([0.0, 1.0]), trigger_link=link)

        sweep_controller.execute_sweep_1d.return_value = np.array([0.001, 0.002])

        orchestrator.sweep_1d(axis, "contact1")

        list_sweep_instrument.reset_voltage_list.assert_called_once_with("gate1")

    def test_sweep_1d_resets_on_failure(
        self,
        orchestrator: SweepOrchestrator,
        list_sweep_instrument: Mock,
        sweep_controller: Mock,
        link: TriggerLink,
    ) -> None:
        """sweep_1d should reset channels even if execution fails."""
        axis = SweepAxis(gate="gate1", voltages=np.array([0.0, 1.0]), trigger_link=link)

        sweep_controller.execute_sweep_1d.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            orchestrator.sweep_1d(axis, "contact1")

        list_sweep_instrument.reset_voltage_list.assert_called_once_with("gate1")

    def test_sweep_2d_nested_trigger_order(
        self,
        orchestrator_2d: SweepOrchestrator,
        list_sweep_instrument: Mock,
        sweep_controller: Mock,
        link: TriggerLink,
        link2: TriggerLink,
    ) -> None:
        """sweep_2d should preload both axes."""
        outer = SweepAxis(
            gate="gate1", voltages=np.array([0.0, 1.0]), trigger_link=link
        )
        inner = SweepAxis(
            gate="gate2", voltages=np.array([0.0, 0.5, 1.0]), trigger_link=link2
        )

        result_2d = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        sweep_controller.execute_sweep_2d.return_value = result_2d

        v_outer, v_inner, currents = orchestrator_2d.sweep_2d(outer, inner, "contact1")

        assert list_sweep_instrument.load_voltage_list.call_count == 2
        assert currents.shape == (2, 3)

    def test_sweep_2d_outer_inner_voltage_shapes(
        self,
        orchestrator_2d: SweepOrchestrator,
        sweep_controller: Mock,
        link: TriggerLink,
        link2: TriggerLink,
    ) -> None:
        """sweep_2d should return correct voltage array shapes."""
        outer = SweepAxis(
            gate="gate1", voltages=np.array([0.0, 1.0, 2.0]), trigger_link=link
        )
        inner = SweepAxis(
            gate="gate2", voltages=np.array([0.5, 1.5]), trigger_link=link2
        )

        result_2d = np.zeros((3, 2))
        sweep_controller.execute_sweep_2d.return_value = result_2d

        v_outer, v_inner, currents = orchestrator_2d.sweep_2d(outer, inner, "contact1")

        assert len(v_outer) == 3
        assert len(v_inner) == 2
        assert currents.shape == (3, 2)

    def test_sweep_2d_delegates_to_sweep_controller(
        self,
        orchestrator_2d: SweepOrchestrator,
        sweep_controller: Mock,
        link: TriggerLink,
        link2: TriggerLink,
    ) -> None:
        """sweep_2d should call sweep_controller.execute_sweep_2d."""
        outer = SweepAxis(
            gate="gate1", voltages=np.array([0.0, 1.0]), trigger_link=link
        )
        inner = SweepAxis(
            gate="gate2", voltages=np.array([0.0, 0.5, 1.0]), trigger_link=link2
        )

        sweep_controller.execute_sweep_2d.return_value = np.zeros((2, 3))

        orchestrator_2d.sweep_2d(
            outer, inner, "contact1", n_avg=2, settling_wait_ns=100_000
        )

        sweep_controller.execute_sweep_2d.assert_called_once_with(
            outer_trigger_name="trig1",
            inner_trigger_name="trig2",
            n_outer=2,
            n_inner=3,
            measure_electrode="contact1",
            n_avg=2,
            settling_wait_ns=100_000,
        )

    def test_validates_list_sweep_channel_accessible(
        self,
        orchestrator: SweepOrchestrator,
        list_sweep_instrument: Mock,
        link: TriggerLink,
    ) -> None:
        """Should propagate error if instrument channel is not accessible."""
        axis = SweepAxis(gate="gate1", voltages=np.array([0.0]), trigger_link=link)

        list_sweep_instrument.load_voltage_list.side_effect = KeyError(
            "channel not found"
        )

        with pytest.raises(KeyError, match="channel not found"):
            orchestrator.prepare([axis])
