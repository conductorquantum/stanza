from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest

from stanza.base.channels import ChannelConfig
from stanza.drivers.opx import OPXInstrument, OPXMeasurementChannel
from stanza.exceptions import InstrumentError
from stanza.models import GateType, InstrumentType, MeasurementInstrumentConfig, PadType


@pytest.fixture
def channel_config():
    return ChannelConfig(
        "test_channel", (-1.0, 1.0), PadType.GATE, GateType.PLUNGER, measure_channel=1
    )


@pytest.fixture
def instrument_config():
    return MeasurementInstrumentConfig(
        name="opx_test",
        type=InstrumentType.MEASUREMENT,
        ip_addr="192.168.1.1",
        port=80,
        machine_type="OPX1",
        cluster_name="test_cluster",
        sample_time=0.001,
        measurement_duration=0.1,
        measurement_channels=[1, 2],
    )


@pytest.fixture
def opx_mocks():
    mock_driver, mock_qmm = Mock(), Mock()
    mock_qmm.open_qm.return_value = mock_driver
    patches = [
        ("stanza.drivers.opx.HAS_QM", True),
        ("stanza.drivers.opx.QuantumMachinesManager", mock_qmm),
        ("stanza.drivers.opx.get_config_resource", "{}"),
        ("stanza.drivers.opx.substitute_parameters", "{}"),
        ("stanza.drivers.opx.FullQuaConfig", Mock()),
    ]
    with ExitStack() as stack:
        for target, value in patches:
            stack.enter_context(patch(target, return_value=value))
        stack.enter_context(patch.object(OPXInstrument, "_initialize_channels"))
        yield mock_driver


class TestOPXMeasurementChannel:
    def test_initialization(self, channel_config):
        driver = Mock()
        channel = OPXMeasurementChannel("test_channel", 1, channel_config)
        channel.set_driver(driver)

        assert (channel.name, channel.channel_id, channel.driver) == (
            "test_channel",
            1,
            driver,
        )
        assert (channel.count, channel.job_id, channel.read_len) == (0, None, None)

    def test_setters(self, channel_config):
        channel = OPXMeasurementChannel("test_channel", 1, channel_config)
        channel.set_driver(Mock())
        channel.set_job_id(123)
        channel.set_read_len(1000)

        assert (channel.job_id, channel.read_len) == (123, 1000)

    @pytest.mark.parametrize(
        "driver,job_id,read_len,error_msg",
        [
            (None, None, None, "OPX driver not set"),
            (Mock(), None, None, "job_id not set"),
            (Mock(), 123, None, "read_len not set"),
        ],
    )
    def test_get_current_validation(
        self, channel_config, driver, job_id, read_len, error_msg
    ):
        channel = OPXMeasurementChannel("test_channel", 1, channel_config)
        if driver is not None:
            channel.set_driver(driver)
        if job_id:
            channel.set_job_id(job_id)
        if read_len:
            channel.set_read_len(read_len)

        with pytest.raises(InstrumentError, match=error_msg):
            channel.get_current()

    @patch("stanza.drivers.opx.wait_until_job_is_paused")
    @patch("stanza.drivers.opx.demod2volts", return_value=1.5)
    def test_get_current_success(self, mock_demod2volts, mock_wait, channel_config):
        driver, job, handle = Mock(), Mock(), Mock()
        driver.get_job.return_value = job
        job.result_handles.get.return_value = handle
        handle.fetch.return_value = "raw_data"

        channel = OPXMeasurementChannel("test_channel", 1, channel_config)
        channel.set_driver(driver)
        channel.set_job_id(123)
        channel.set_read_len(1000)

        result = channel.get_current()

        assert result == -1.5
        assert channel.count == 1
        job.resume.assert_called_once()
        mock_wait.assert_called_once_with(job)
        handle.wait_for_values.assert_called_once_with(1, timeout=10)

    def test_get_current_missing_handle(self, channel_config):
        driver = Mock()
        job = Mock()
        driver.get_job.return_value = job
        job.result_handles.get.return_value = None

        channel = OPXMeasurementChannel("test_channel", 1, channel_config)
        channel.set_driver(driver)
        channel.set_job_id(123)
        channel.set_read_len(1000)

        with pytest.raises(
            InstrumentError, match="No output handle measure_test_channel"
        ):
            channel.get_current()

    @patch("stanza.drivers.opx.wait_until_job_is_paused")
    @patch("stanza.drivers.opx.demod2volts", return_value=2.5)
    def test_get_current_handles_wait_exception(
        self, mock_demod2volts, mock_wait, channel_config
    ):
        driver, job, handle = Mock(), Mock(), Mock()
        driver.get_job.return_value = job
        job.result_handles.get.return_value = handle
        handle.fetch.return_value = "raw_data"
        handle.wait_for_values.side_effect = RuntimeError("Timeout")

        channel = OPXMeasurementChannel("test_channel", 1, channel_config)
        channel.set_driver(driver)
        channel.set_job_id(123)
        channel.set_read_len(1000)

        result = channel.get_current()

        assert result == -2.5
        assert channel.count == 1
        handle.wait_for_values.assert_called_once_with(1, timeout=10)


class TestOPXInstrument:
    def test_requires_qm_library(self, instrument_config):
        with patch("stanza.drivers.opx.HAS_QM", False):
            with pytest.raises(ImportError, match="qm is not installed"):
                OPXInstrument(instrument_config, {})

    def test_initialization(self, instrument_config, opx_mocks):
        channel_configs = {
            "ch1": ChannelConfig(
                name="ch1",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.GATE,
                electrode_type=GateType.PLUNGER,
                measure_channel=1,
            )
        }
        instrument = OPXInstrument(instrument_config, channel_configs)

        assert (
            instrument.host,
            instrument.port,
            instrument.machine_type,
            instrument.cluster_name,
        ) == ("192.168.1.1", 80, "OPX1", "test_cluster")
        assert instrument.driver == opx_mocks

    def test_prepare_measurement(self, instrument_config, opx_mocks):
        instrument = OPXInstrument(instrument_config, {})
        mock_job = Mock()
        mock_job.id = 456
        opx_mocks.execute.return_value = mock_job

        mock_channel = Mock()
        instrument.channels = {"test_ch": mock_channel}

        with patch.object(type(instrument), "qua_program", Mock()) as mock_program:
            instrument.prepare_measurement()

        opx_mocks.compile.assert_called_once_with(mock_program)
        opx_mocks.execute.assert_called_once_with(mock_program)
        mock_channel.set_job_id.assert_called_once_with(456)
        mock_channel.set_read_len.assert_called_once()

    def test_teardown_measurement(self, instrument_config, opx_mocks):
        job1, job2 = Mock(), Mock()
        opx_mocks.get_jobs.return_value = [job1, job2]

        instrument = OPXInstrument(instrument_config, {})
        instrument.teardown_measurement()

        job1.halt.assert_called_once()
        job2.halt.assert_called_once()
        opx_mocks.close.assert_called_once()

    def test_measure_method(self, instrument_config, opx_mocks):
        instrument = OPXInstrument(instrument_config, {})
        mock_channel = Mock()
        mock_channel.get_parameter_value.return_value = 2.5

        with (
            patch.object(instrument, "prepare_measurement_context"),
            patch.object(instrument, "get_channel", return_value=mock_channel),
        ):
            result = instrument.measure("test_channel")

        assert result == 2.5
        mock_channel.get_parameter_value.assert_called_once_with("current")

    @patch("stanza.drivers.opx.HAS_QM", True)
    def test_initialize_channels_with_measurement_channels(self, instrument_config):
        channel_configs = {
            "ch1": ChannelConfig(
                "ch1", (-1.0, 1.0), PadType.GATE, GateType.PLUNGER, measure_channel=1
            ),
            "ch2": ChannelConfig(
                "ch2", (-1.0, 1.0), PadType.GATE, GateType.PLUNGER, measure_channel=2
            ),
            "ch3": ChannelConfig(
                "ch3", (-1.0, 1.0), PadType.GATE, GateType.PLUNGER, measure_channel=3
            ),
        }

        mock_qmm = Mock()
        mock_driver = Mock()
        mock_qmm.open_qm.return_value = mock_driver
        patches = [
            patch("stanza.drivers.opx.QuantumMachinesManager", return_value=mock_qmm),
            patch(
                "stanza.drivers.opx.get_config_resource",
                return_value='{"elements": {}}',
            ),
            patch(
                "stanza.drivers.opx.substitute_parameters",
                return_value='{"elements": {}}',
            ),
            patch("stanza.drivers.opx.FullQuaConfig", return_value=Mock()),
        ]

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            instrument = OPXInstrument(instrument_config, channel_configs)

            assert "measure_ch1" in instrument.channels
            assert "measure_ch2" in instrument.channels
            assert "measure_ch3" not in instrument.channels

    @patch("stanza.drivers.opx.HAS_QM", True)
    def test_initialize_channels_with_none_measurement_channels(self):
        config = MeasurementInstrumentConfig(
            name="opx_test",
            type=InstrumentType.MEASUREMENT,
            ip_addr="192.168.1.1",
            port=80,
            machine_type="OPX1",
            cluster_name="test_cluster",
            sample_time=0.001,
            measurement_duration=0.1,
            measurement_channels=None,
        )
        channel_configs = {
            "ch1": ChannelConfig(
                "ch1", (-1.0, 1.0), PadType.GATE, GateType.PLUNGER, measure_channel=1
            )
        }

        mock_qmm = Mock()
        mock_driver = Mock()
        mock_qmm.open_qm.return_value = mock_driver
        patches = [
            patch("stanza.drivers.opx.QuantumMachinesManager", return_value=mock_qmm),
            patch(
                "stanza.drivers.opx.get_config_resource",
                return_value='{"elements": {}}',
            ),
            patch(
                "stanza.drivers.opx.substitute_parameters",
                return_value='{"elements": {}}',
            ),
            patch("stanza.drivers.opx.FullQuaConfig", return_value=Mock()),
        ]

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            instrument = OPXInstrument(config, channel_configs)

            assert "measure_ch1" not in instrument.channels

    @patch("stanza.drivers.opx.HAS_QM", True)
    def test_qua_config_property(self, instrument_config, opx_mocks):
        channel_configs = {
            "test_ch": ChannelConfig(
                "test_ch",
                (-1.0, 1.0),
                PadType.GATE,
                GateType.PLUNGER,
                measure_channel=1,
            )
        }

        with (
            patch("stanza.drivers.opx.get_config_resource") as mock_get_config,
            patch("stanza.drivers.opx.substitute_parameters") as mock_substitute,
            patch("stanza.drivers.opx.FullQuaConfig") as mock_qua_config,
        ):
            mock_get_config.return_value = '{"elements": {}}'
            mock_substitute.return_value = '{"elements": {}}'
            mock_qua_config_instance = Mock()
            mock_qua_config.return_value = mock_qua_config_instance

            instrument = OPXInstrument(instrument_config, channel_configs)
            config = instrument.qua_config

            assert config == mock_qua_config_instance
            mock_get_config.assert_called_once_with("templates/qua_config.json")
            mock_substitute.assert_called_once()

    @patch("stanza.drivers.opx.HAS_QM", True)
    def test_qua_program_property(self, instrument_config, opx_mocks):
        mock_qmm = Mock()
        mock_driver = Mock()
        mock_qmm.open_qm.return_value = mock_driver
        patches = [
            patch("stanza.drivers.opx.QuantumMachinesManager", return_value=mock_qmm),
            patch(
                "stanza.drivers.opx.get_config_resource",
                return_value='{"elements": {}}',
            ),
            patch(
                "stanza.drivers.opx.substitute_parameters",
                return_value='{"elements": {}}',
            ),
            patch("stanza.drivers.opx.FullQuaConfig", return_value=Mock()),
        ]

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            instrument = OPXInstrument(instrument_config, {})
            instrument.channels = {"test_ch1": Mock(), "test_ch2": Mock()}

            mock_prog = Mock()
            with patch.object(type(instrument), "qua_program", mock_prog):
                program = instrument.qua_program
                assert program == mock_prog

    @patch("stanza.drivers.opx.HAS_QM", True)
    def test_teardown_measurement_handles_job_halt_exceptions(
        self, instrument_config, opx_mocks
    ):
        job1 = Mock()
        job2 = Mock()
        job1.halt.side_effect = RuntimeError("Halt failed")
        job1.id = "job1"
        job2.id = "job2"
        opx_mocks.get_jobs.return_value = [job1, job2]

        instrument = OPXInstrument(instrument_config, {})

        with patch("stanza.drivers.opx.logger") as mock_logger:
            instrument.teardown_measurement()

            job1.halt.assert_called_once()
            job2.halt.assert_called_once()
            opx_mocks.close.assert_called_once()
            mock_logger.warning.assert_called_once_with("Failed to halt job job1")
