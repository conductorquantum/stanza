"""Tests for hardware_sweep — hardware-accelerated sweep coordination."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from stanza.orchestration import SweepAxis, hardware_sweep
from stanza.triggers import TriggerLink

# --- Helpers ---


def _make_link(name: str = "trig1", port: int = 5, ext: str = "ext1") -> TriggerLink:
    return TriggerLink(
        name=name,
        source_port=("con1", 2, port),
        sink_instrument="qdac",
        sink_channel="gate1",
        sink_trigger_port=ext,
    )


# --- SweepAxis Tests ---


class TestSweepAxis:
    def test_creation(self) -> None:
        link = _make_link()
        axis = SweepAxis(
            gate="gate1",
            voltages=np.array([0.0, 0.5, 1.0]),
            trigger_link=link,
        )
        assert axis.gate == "gate1"
        assert len(axis.voltages) == 3
        assert axis.trigger_link is link

    def test_frozen(self) -> None:
        link = _make_link()
        axis = SweepAxis(
            gate="gate1",
            voltages=np.array([0.0]),
            trigger_link=link,
        )
        with pytest.raises(AttributeError):
            axis.gate = "other"  # type: ignore[misc]


# --- hardware_sweep Tests ---


class TestHardwareSweep:
    def test_1d_preloads_and_resets(self) -> None:
        """1D sweep should load voltage list, execute, and reset."""
        link = _make_link()
        axis = SweepAxis(
            gate="gate1", voltages=np.array([0.0, 0.5, 1.0]), trigger_link=link
        )
        vs = Mock()
        ctrl = Mock()
        ctrl.execute_sweep_1d.return_value = np.array([0.001, 0.002, 0.003])

        voltages, currents = hardware_sweep(
            axes=[axis],
            measure_electrode="contact1",
            voltage_source=vs,
            controller=ctrl,
        )

        vs.load_voltage_list.assert_called_once_with(
            channel_name="gate1",
            voltages=pytest.approx(np.array([0.0, 0.5, 1.0])),
            trigger_port="ext1",
        )
        vs.reset_voltage_list.assert_called_once_with("gate1")
        assert len(voltages) == 3
        assert len(currents) == 3

    def test_1d_delegates_to_controller(self) -> None:
        """1D sweep should call controller.execute_sweep_1d with correct args."""
        link = _make_link()
        axis = SweepAxis(gate="gate1", voltages=np.array([0.0, 0.5]), trigger_link=link)
        vs = Mock()
        ctrl = Mock()
        ctrl.execute_sweep_1d.return_value = np.array([0.001, 0.002])

        hardware_sweep(
            axes=[axis],
            measure_electrode="contact1",
            voltage_source=vs,
            controller=ctrl,
        )

        ctrl.execute_sweep_1d.assert_called_once_with(
            trigger_link_name="trig1",
            n_points=2,
            measure_electrode="contact1",
            n_avg=1,
            settling_wait_ns=250_000,
        )

    def test_1d_resets_on_failure(self) -> None:
        """1D sweep should reset channels even if execution fails."""
        link = _make_link()
        axis = SweepAxis(gate="gate1", voltages=np.array([0.0, 1.0]), trigger_link=link)
        vs = Mock()
        ctrl = Mock()
        ctrl.execute_sweep_1d.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            hardware_sweep(
                axes=[axis],
                measure_electrode="contact1",
                voltage_source=vs,
                controller=ctrl,
            )

        vs.reset_voltage_list.assert_called_once_with("gate1")

    def test_2d_preloads_both_axes(self) -> None:
        """2D sweep should preload voltage lists for both axes."""
        link1 = _make_link()
        link2 = _make_link(name="trig2", port=6, ext="ext2")
        outer = SweepAxis(
            gate="gate1", voltages=np.array([0.0, 1.0]), trigger_link=link1
        )
        inner = SweepAxis(
            gate="gate2", voltages=np.array([0.0, 0.5, 1.0]), trigger_link=link2
        )
        vs = Mock()
        ctrl = Mock()
        ctrl.execute_sweep_2d.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        v_outer, v_inner, currents = hardware_sweep(
            axes=[outer, inner],
            measure_electrode="contact1",
            voltage_source=vs,
            controller=ctrl,
        )

        assert vs.load_voltage_list.call_count == 2
        assert currents.shape == (2, 3)
        assert len(v_outer) == 2
        assert len(v_inner) == 3

    def test_2d_delegates_to_controller(self) -> None:
        """2D sweep should call controller.execute_sweep_2d with correct args."""
        link1 = _make_link()
        link2 = _make_link(name="trig2", port=6, ext="ext2")
        outer = SweepAxis(
            gate="gate1", voltages=np.array([0.0, 1.0]), trigger_link=link1
        )
        inner = SweepAxis(
            gate="gate2", voltages=np.array([0.0, 0.5, 1.0]), trigger_link=link2
        )
        vs = Mock()
        ctrl = Mock()
        ctrl.execute_sweep_2d.return_value = np.zeros((2, 3))

        hardware_sweep(
            axes=[outer, inner],
            measure_electrode="contact1",
            voltage_source=vs,
            controller=ctrl,
            n_avg=2,
            settling_wait_ns=100_000,
        )

        ctrl.execute_sweep_2d.assert_called_once_with(
            outer_trigger_name="trig1",
            inner_trigger_name="trig2",
            n_outer=2,
            n_inner=3,
            measure_electrode="contact1",
            n_avg=2,
            settling_wait_ns=100_000,
        )

    def test_2d_resets_both_axes(self) -> None:
        """2D sweep should reset both channels."""
        link1 = _make_link()
        link2 = _make_link(name="trig2", port=6, ext="ext2")
        outer = SweepAxis(gate="gate1", voltages=np.array([0.0]), trigger_link=link1)
        inner = SweepAxis(gate="gate2", voltages=np.array([0.5]), trigger_link=link2)
        vs = Mock()
        ctrl = Mock()
        ctrl.execute_sweep_2d.return_value = np.zeros((1, 1))

        hardware_sweep(
            axes=[outer, inner],
            measure_electrode="contact1",
            voltage_source=vs,
            controller=ctrl,
        )

        assert vs.reset_voltage_list.call_count == 2
        vs.reset_voltage_list.assert_any_call("gate1")
        vs.reset_voltage_list.assert_any_call("gate2")

    def test_rejects_3d(self) -> None:
        """Should raise ValueError for >2 axes."""
        links = [
            _make_link(name=f"t{i}", port=5 + i, ext=f"ext{i + 1}") for i in range(3)
        ]
        axes = [
            SweepAxis(gate="gate1", voltages=np.array([0.0]), trigger_link=link)
            for link in links
        ]

        with pytest.raises(ValueError, match="supports 1 or 2 axes"):
            hardware_sweep(
                axes=axes,
                measure_electrode="contact1",
                voltage_source=Mock(),
                controller=Mock(),
            )

    def test_propagates_load_error(self) -> None:
        """Should propagate error if instrument channel is not accessible."""
        link = _make_link()
        axis = SweepAxis(gate="gate1", voltages=np.array([0.0]), trigger_link=link)
        vs = Mock()
        vs.load_voltage_list.side_effect = KeyError("channel not found")

        with pytest.raises(KeyError, match="channel not found"):
            hardware_sweep(
                axes=[axis],
                measure_electrode="contact1",
                voltage_source=vs,
                controller=Mock(),
            )
