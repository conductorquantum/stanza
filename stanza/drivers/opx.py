from __future__ import annotations

import json
import logging
from functools import cached_property

# mypy: disable-error-code="union-attr,attr-defined"
from typing import Any

from stanza.base.channels import ChannelConfig, MeasurementChannel
from stanza.base.instruments import BaseMeasurementInstrument
from stanza.drivers.utils import demod2volts, wait_until_job_is_paused
from stanza.exceptions import InstrumentError
from stanza.models import MeasurementInstrumentConfig
from stanza.timing import seconds_to_ns
from stanza.utils import get_config_resource, substitute_parameters

try:
    from qm import FullQuaConfig, Program, QuantumMachinesManager
    from qm.qua import (
        FUNCTIONS,
        declare,
        declare_stream,
        fixed,
        for_,
        infinite_loop_,
        integration,
        measure,
        pause,
        program,
        save,
        stream_processing,
    )

    HAS_QM = True
except ImportError:
    QuantumMachinesManager = None  # type: ignore[misc,assignment]
    Program = None  # type: ignore[misc,assignment]
    FullQuaConfig = None  # type: ignore[misc,assignment]
    RunningQmJob = None
    HAS_QM = False

logger = logging.getLogger(__name__)


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
    """OPX-specific instrument with hardware integration."""

    def __init__(
        self,
        instrument_config: MeasurementInstrumentConfig,
        channel_configs: dict[str, ChannelConfig],
    ):
        if not HAS_QM:
            raise ImportError(
                "qm is not installed. Install with: pip install stanza[qm]"
            )

        super().__init__(instrument_config)

        self.host = instrument_config.ip_addr
        self.port = instrument_config.port
        self.machine_type = instrument_config.machine_type
        self.cluster_name = instrument_config.cluster_name
        self.connection_headers = instrument_config.connection_headers
        self.measurement_channels = instrument_config.measurement_channels

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
        qua_config_template = get_config_resource("templates/qua_config.json")
        qua_config_str = substitute_parameters(
            qua_config_template,
            {
                "MACHINE_TYPE": self.machine_type,
                "READ_LEN": self.read_len,
            },
        )
        qua_config = json.loads(qua_config_str)

        for channel_name, channel in self.channels.items():
            port = ("con1", 2, channel.channel_id)
            qua_config["elements"][channel_name] = {
                "singleInput": {"port": port},
                "outputs": {"out1": port},
                "intermediate_frequency": 0,
                "operations": {"readout": "readout_pulse"},
                "time_of_flight": 28,
                "smearing": 0,
            }

        return FullQuaConfig(**qua_config)

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
