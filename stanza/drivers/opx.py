from __future__ import annotations

import logging
from typing import Protocol

from stanza.drivers.utils import demod2volts, wait_until_job_is_paused
from stanza.exceptions import InstrumentError
from stanza.instruments.channels import ChannelConfig, MeasurementChannel

logger = logging.getLogger(__name__)


class OPXResultHandle(Protocol):
    """Protocol for OPX result handle interface."""

    def wait_for_values(self, count: int, timeout: int = 10) -> None: ...

    def fetch(self, count: int) -> float: ...


class OPXJob(Protocol):
    """Protocol for OPX job interface."""

    def resume(self) -> None: ...

    @property
    def result_handles(self) -> dict[str, OPXResultHandle]: ...


class OPXDriver(Protocol):
    """Protocol for OPX driver interface."""

    def get_job(self, job_id: int) -> OPXJob: ...


class OPXMeasurementChannel(MeasurementChannel):
    """OPX-specific measurement channel with hardware integration."""

    def __init__(
        self, name: str, channel_id: int, config: ChannelConfig, driver: OPXDriver
    ):
        self.name = name
        self.channel_id = channel_id
        self.driver = driver
        self.count = 0

        self.job_id: int | None = None
        self.read_len: int | None = None
        super().__init__(config)

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
        prev = self.count
        index = prev + 1

        job.resume()

        wait_until_job_is_paused(job)
        out_handle = job.result_handles.get("out")
        if out_handle is None:
            raise InstrumentError("No output handle found")

        try:
            out_handle.wait_for_values(index, timeout=10)
        except Exception:
            pass

        raw_data = out_handle.fetch(index)
        data = demod2volts(raw_data, self.read_len, single_demod=True)

        self.count = index

        return float(-data)
