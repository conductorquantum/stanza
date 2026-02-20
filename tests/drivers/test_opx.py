"""Tests for OPX driver: OPXInstrument backward compat and OPXPulseController."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from stanza.base.channels import ChannelConfig
from stanza.drivers.opx import (
    OPXInstrument,
    OPXPulseController,
    PulseSequence,
    PulseStep,
)
from stanza.drivers.opx_config_builder import OPXConfigBuilder
from stanza.drivers.opx_triggers import (
    TriggerMode,
    hardware_in_trigger,
    hardware_out_trigger,
    software_trigger,
    timed_trigger,
)
from stanza.exceptions import InstrumentError
from stanza.models import (
    ContactType,
    InstrumentType,
    MeasurementInstrumentConfig,
    PadType,
)
from stanza.pulses import PulseRegistry, make_square_pulse
from stanza.triggers import TriggerLink

# --- Fixtures ---


def _make_channel_config(name: str, measure_channel: int) -> ChannelConfig:
    return ChannelConfig(
        name=name,
        voltage_range=(-1.0, 1.0),
        pad_type=PadType.CONTACT,
        electrode_type=ContactType.DRAIN,
        measure_channel=measure_channel,
    )


@pytest.fixture
def channel_configs() -> dict[str, ChannelConfig]:
    return {
        "drain1": _make_channel_config("drain1", measure_channel=1),
        "drain2": _make_channel_config("drain2", measure_channel=2),
    }


@pytest.fixture
def instrument_config() -> MeasurementInstrumentConfig:
    return MeasurementInstrumentConfig(
        name="opx_test",
        type=InstrumentType.MEASUREMENT,
        ip_addr="127.0.0.1",
        port=9510,
        machine_type="OPX1000",
        cluster_name="test_cluster",
        sample_time=1e-6,
        measurement_duration=10e-6,
        measurement_channels=[1, 2],
    )


@pytest.fixture
def pulse_registry() -> PulseRegistry:
    reg = PulseRegistry()
    pulse = make_square_pulse("test_pulse", amplitude=0.1, duration_ns=100)
    reg.add_pulse(pulse)
    return reg


# --- PulseStep Tests ---


class TestPulseStep:
    def test_basic_creation(self) -> None:
        step = PulseStep(pulse_name="test", element="ctrl_1")
        assert step.pulse_name == "test"
        assert step.element == "ctrl_1"
        assert step.wait_after_cycles == 0

    def test_with_wait(self) -> None:
        step = PulseStep(pulse_name="test", element="ctrl_1", wait_after_cycles=100)
        assert step.wait_after_cycles == 100

    def test_negative_wait_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            PulseStep(pulse_name="test", element="ctrl_1", wait_after_cycles=-1)

    def test_frozen(self) -> None:
        step = PulseStep(pulse_name="test", element="ctrl_1")
        with pytest.raises(AttributeError):
            step.pulse_name = "other"  # type: ignore[misc]

    def test_pulse_sequence_is_list(self) -> None:
        seq: PulseSequence = [
            PulseStep(pulse_name="a", element="e1"),
            PulseStep(pulse_name="b", element="e2", wait_after_cycles=25),
        ]
        assert len(seq) == 2
        assert seq[0].pulse_name == "a"
        assert seq[1].wait_after_cycles == 25


# --- OPXInstrument Backward Compat Tests ---


class TestOPXInstrumentConfig:
    def test_qua_config_uses_builder(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> None:
        """OPXInstrument.qua_config should use OPXConfigBuilder internally."""
        with (
            patch.object(
                OPXConfigBuilder,
                "build",
                return_value={
                    "version": "1",
                    "controllers": {},
                    "elements": {},
                    "pulses": {},
                    "waveforms": {},
                    "integration_weights": {},
                },
            ) as mock_build,
            patch("stanza.drivers.opx.QuantumMachinesManager") as mock_qmm,
            patch("stanza.drivers.opx.FullQuaConfig"),
        ):
            mock_qmm.return_value.open_qm.return_value = Mock()
            inst = OPXInstrument(instrument_config, channel_configs)
            _ = inst.qua_config
            mock_build.assert_called_once()

    def test_qua_config_has_elements(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> None:
        """Channels should produce elements in the config."""
        with (
            patch("stanza.drivers.opx.QuantumMachinesManager") as mock_qmm,
            patch("stanza.drivers.opx.FullQuaConfig", side_effect=lambda **kw: kw),
        ):
            mock_qmm.return_value.open_qm.return_value = Mock()
            inst = OPXInstrument(instrument_config, channel_configs)
            config = inst.qua_config
            assert "elements" in config
            elem_names = list(config["elements"].keys())
            assert len(elem_names) == 2
            assert all("measure_" in n for n in elem_names)


# --- OPXPulseController Tests ---


class TestOPXPulseController:
    @pytest.fixture
    def controller(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        pulse_registry: PulseRegistry,
    ) -> OPXPulseController:
        return OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            pulse_registry=pulse_registry,
            trigger_config=software_trigger(),
        )

    def test_add_pulse(self, controller: OPXPulseController) -> None:
        new_pulse = make_square_pulse("second_pulse", amplitude=0.2, duration_ns=200)
        controller.add_pulse(new_pulse)
        assert "second_pulse" in controller.pulse_registry.pulse_names

    def test_build_config(self, controller: OPXPulseController) -> None:
        config = controller._build_config()
        assert "version" in config
        assert "controllers" in config
        assert "elements" in config
        assert "pulses" in config

    def test_build_program_software(self, controller: OPXPulseController) -> None:
        """SOFTWARE mode should produce a program."""
        seq = [PulseStep(pulse_name="test_pulse", element="ctrl_1")]
        prog = controller.build_program(seq)
        assert prog is not None

    def test_build_program_timed(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        pulse_registry: PulseRegistry,
    ) -> None:
        """TIMED mode with sufficient interval should succeed."""
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            pulse_registry=pulse_registry,
            trigger_config=timed_trigger(interval_ns=1000, repetitions=10),
        )
        seq = [PulseStep(pulse_name="test_pulse", element="ctrl_1")]
        prog = controller.build_program(seq)
        assert prog is not None

    def test_build_program_hardware_out(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        pulse_registry: PulseRegistry,
    ) -> None:
        """HARDWARE_OUT mode should produce a program."""
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            pulse_registry=pulse_registry,
            trigger_config=hardware_out_trigger(
                digital_port=("con1", 2, 1), duration_ns=100
            ),
        )
        seq = [PulseStep(pulse_name="test_pulse", element="ctrl_1")]
        prog = controller.build_program(seq)
        assert prog is not None

    def test_build_program_hardware_in(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        pulse_registry: PulseRegistry,
    ) -> None:
        """HARDWARE_IN mode should produce a program."""
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            pulse_registry=pulse_registry,
            trigger_config=hardware_in_trigger(
                digital_port=("con1", 2, 1), timeout_ns=10000
            ),
        )
        seq = [PulseStep(pulse_name="test_pulse", element="ctrl_1")]
        prog = controller.build_program(seq)
        assert prog is not None

    def test_build_raises_on_missing_pulse(
        self, controller: OPXPulseController
    ) -> None:
        """build_program should fail if a pulse name doesn't exist in registry."""
        seq = [PulseStep(pulse_name="nonexistent", element="ctrl_1")]
        with pytest.raises(InstrumentError, match="not found in registry"):
            controller.build_program(seq)

    def test_timed_raises_on_short_interval(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> None:
        """TIMED interval shorter than sequence length should fail."""
        reg = PulseRegistry()
        long_pulse = make_square_pulse("long_pulse", amplitude=0.1, duration_ns=400)
        reg.add_pulse(long_pulse)

        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            pulse_registry=reg,
            trigger_config=timed_trigger(interval_ns=100, repetitions=5),
        )
        seq = [PulseStep(pulse_name="long_pulse", element="ctrl_1")]
        with pytest.raises(InstrumentError, match="shorter than sequence length"):
            controller.build_program(seq)

    def test_sequence_length_cycles(self, controller: OPXPulseController) -> None:
        """_sequence_length_cycles should sum pulse lengths + waits."""
        seq = [
            PulseStep(pulse_name="test_pulse", element="ctrl_1", wait_after_cycles=10),
        ]
        length = controller._sequence_length_cycles(seq)
        assert length == 25 + 10  # 25 cycles (100ns) pulse + 10 cycles wait

    def test_teardown(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> None:
        """teardown should halt jobs and close connection."""
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
        )

        mock_driver = Mock()
        mock_job = Mock()
        mock_job.id = 1
        mock_driver.get_jobs.return_value = [mock_job]
        controller._driver = mock_driver

        controller.teardown()

        mock_job.halt.assert_called_once()
        mock_driver.close.assert_called_once()
        assert controller._driver is None
        assert controller._job is None

    def test_teardown_handles_missing_driver(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> None:
        """teardown with no driver should not raise."""
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
        )
        controller.teardown()

    def test_execute_requires_program(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> None:
        """execute with no program should raise."""
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
        )
        with pytest.raises(InstrumentError, match="No program"):
            controller.execute(None)

    def test_measure_channel_unknown_raises(
        self, controller: OPXPulseController
    ) -> None:
        """measure_channel with unknown name should raise."""
        with pytest.raises(InstrumentError, match="not found"):
            controller.measure_channel("nonexistent")

    def test_hardware_in_no_timeout_waits_forever(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        pulse_registry: PulseRegistry,
    ) -> None:
        """HARDWARE_IN with timeout=None should build without sentinel logic."""
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            pulse_registry=pulse_registry,
            trigger_config=hardware_in_trigger(
                digital_port=("con1", 2, 1), timeout_ns=None
            ),
        )
        seq = [PulseStep(pulse_name="test_pulse", element="ctrl_1")]
        prog = controller.build_program(seq)
        assert prog is not None

    def test_play_and_measure_raises_on_timeout(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        pulse_registry: PulseRegistry,
    ) -> None:
        """play_and_measure should raise InstrumentError when timeout sentinels present."""
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            pulse_registry=pulse_registry,
            trigger_config=hardware_in_trigger(
                digital_port=("con1", 2, 1), timeout_ns=10000
            ),
        )

        with (
            patch.object(controller, "build_program", return_value=Mock()),
            patch.object(controller, "execute"),
            patch.object(controller, "measure_channel", return_value=-1.0),
        ):
            seq = [PulseStep(pulse_name="test_pulse", element="ctrl_1")]
            with pytest.raises(InstrumentError, match="Trigger timeout"):
                controller.play_and_measure(seq)

    def test_play_and_measure_allow_timeouts(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
        pulse_registry: PulseRegistry,
    ) -> None:
        """play_and_measure with allow_timeouts=True should not raise on sentinels."""
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            pulse_registry=pulse_registry,
            trigger_config=hardware_in_trigger(
                digital_port=("con1", 2, 1), timeout_ns=10000
            ),
        )

        with (
            patch.object(controller, "build_program", return_value=Mock()),
            patch.object(controller, "execute"),
            patch.object(controller, "measure_channel", return_value=-1.0),
        ):
            seq = [PulseStep(pulse_name="test_pulse", element="ctrl_1")]
            results = controller.play_and_measure(seq, allow_timeouts=True)
            assert all(v == -1.0 for v in results.values())

    def test_build_program_with_no_sequence(
        self, controller: OPXPulseController
    ) -> None:
        """build_program with no sequence should still build (measurement only)."""
        prog = controller.build_program(sequence=None)
        assert prog is not None

    def test_controller_default_trigger(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> None:
        """Controller with no trigger config defaults to SOFTWARE."""
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
        )
        assert controller.trigger_config.mode == TriggerMode.SOFTWARE

    def test_build_config_with_trigger_links_creates_elements(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> None:
        """Trigger links should create corresponding elements in the config."""
        link = TriggerLink(
            name="qdac_ch1_trigger",
            source_port=("con1", 2, 5),
            sink_instrument="qdac",
            sink_channel="gate1",
            sink_trigger_port="ext1",
        )
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            trigger_links=[link],
        )
        config = controller._build_config()

        assert "qdac_ch1_trigger" in config["elements"]
        elem = config["elements"]["qdac_ch1_trigger"]
        assert "trig" in elem.get("operations", {})

    def test_build_config_trigger_link_digital_output_port(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> None:
        """Trigger link should register a digital output port on the FEM."""
        link = TriggerLink(
            name="qdac_trig",
            source_port=("con1", 2, 7),
            sink_instrument="qdac",
            sink_channel="gate1",
            sink_trigger_port="ext2",
        )
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            trigger_links=[link],
        )
        config = controller._build_config()

        # Check that the FEM has a digital output on port 7
        fem = config["controllers"]["con1"]["fems"]["2"]
        assert "digital_outputs" in fem
        assert "7" in fem["digital_outputs"]

    def test_build_config_multiple_trigger_links(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> None:
        """Multiple trigger links should each create separate elements."""
        link1 = TriggerLink(
            name="trig_outer",
            source_port=("con1", 2, 5),
            sink_instrument="qdac",
            sink_channel="gate1",
            sink_trigger_port="ext1",
        )
        link2 = TriggerLink(
            name="trig_inner",
            source_port=("con1", 2, 6),
            sink_instrument="qdac",
            sink_channel="gate2",
            sink_trigger_port="ext2",
        )
        controller = OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            trigger_links=[link1, link2],
        )
        config = controller._build_config()

        assert "trig_outer" in config["elements"]
        assert "trig_inner" in config["elements"]
        assert "trig" in config["elements"]["trig_outer"]["operations"]
        assert "trig" in config["elements"]["trig_inner"]["operations"]


class TestOPXPulseControllerSweepExecution:
    """Tests for execute_sweep_1d and execute_sweep_2d methods."""

    @pytest.fixture
    def controller_with_links(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ) -> OPXPulseController:
        link1 = TriggerLink(
            name="trig_outer",
            source_port=("con1", 2, 5),
            sink_instrument="qdac",
            sink_channel="gate1",
            sink_trigger_port="ext1",
        )
        link2 = TriggerLink(
            name="trig_inner",
            source_port=("con1", 2, 6),
            sink_instrument="qdac",
            sink_channel="gate2",
            sink_trigger_port="ext2",
        )
        return OPXPulseController(
            instrument_config=instrument_config,
            channel_configs=channel_configs,
            trigger_links=[link1, link2],
        )

    def test_execute_sweep_1d_calls_execute(
        self,
        controller_with_links: OPXPulseController,
    ) -> None:
        """execute_sweep_1d should build a QUA program and call execute."""
        import numpy as np

        mock_handle = Mock()
        mock_handle.fetch_all.return_value = np.array([0.1, 0.2, 0.3])
        mock_job = Mock()
        mock_job.result_handles.get.return_value = mock_handle

        with patch.object(controller_with_links, "execute") as mock_exec:
            controller_with_links._job = mock_job
            result = controller_with_links.execute_sweep_1d(
                trigger_link_name="trig_outer",
                n_points=3,
                measure_electrode="drain1",
            )

            mock_exec.assert_called_once()
            assert result.shape == (3,)
            np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    def test_execute_sweep_2d_calls_execute(
        self,
        controller_with_links: OPXPulseController,
    ) -> None:
        """execute_sweep_2d should build a QUA program and call execute."""
        import numpy as np

        mock_handle = Mock()
        mock_handle.fetch_all.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        mock_job = Mock()
        mock_job.result_handles.get.return_value = mock_handle

        with patch.object(controller_with_links, "execute") as mock_exec:
            controller_with_links._job = mock_job
            result = controller_with_links.execute_sweep_2d(
                outer_trigger_name="trig_outer",
                inner_trigger_name="trig_inner",
                n_outer=2,
                n_inner=3,
                measure_electrode="drain1",
            )

            mock_exec.assert_called_once()
            assert result.shape == (2, 3)

    def test_execute_sweep_1d_with_averaging(
        self,
        controller_with_links: OPXPulseController,
    ) -> None:
        """execute_sweep_1d with n_avg > 1 should still return correct shape."""
        import numpy as np

        mock_handle = Mock()
        mock_handle.fetch_all.return_value = np.array([0.15, 0.25])
        mock_job = Mock()
        mock_job.result_handles.get.return_value = mock_handle

        with patch.object(controller_with_links, "execute"):
            controller_with_links._job = mock_job
            result = controller_with_links.execute_sweep_1d(
                trigger_link_name="trig_outer",
                n_points=2,
                measure_electrode="drain1",
                n_avg=5,
                settling_wait_ns=500_000,
            )

            assert result.shape == (2,)
