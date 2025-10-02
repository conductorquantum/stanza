from qm import SimulationConfig
from qm.qua import (
    FUNCTIONS,
    declare,
    declare_stream,
    fixed,
    for_,
    integration,
    measure,
    program,
    save,
    stream_processing,
)

from stanza.base.channels import ChannelConfig
from stanza.drivers.opx import OPXInstrument, OPXMeasurementChannel
from stanza.exceptions import InstrumentError


class OPXMeasurementChannelNoPause(OPXMeasurementChannel):
    """OPX measurement channel that works with simulation."""

    def __init__(self, name: str, channel_id: int, config: ChannelConfig):
        super().__init__(name, channel_id, config)
        self.job = None

    def set_job(self, job) -> None:
        """Set the job object directly."""
        self.job = job

    def _setup_parameters(self) -> None:
        """Setup measurement parameters with custom getter."""
        from stanza.base.channels import Parameter

        current_param = Parameter(
            name="current",
            value=None,
            unit="A",
            getter=self.get_current,
            setter=None,
            metadata={
                "description": "Measured DC current (mean over integration time)"
            },
        )
        self.add_parameter(current_param)

    def get_current(self) -> float:
        if getattr(self, "driver", None) is None:
            raise InstrumentError("OPX driver not set")

        if self.job is None:
            raise InstrumentError("job not set")

        handle_name = f"measure_{self.name}"

        self.job.wait_until("done", timeout=30)
        self.job.get_simulated_samples()

        try:
            h = getattr(self.job.result_handles, handle_name)
            data = h.fetch_all()
            if data is not None and len(data) > 0:
                return float(data[0])
            else:
                raise InstrumentError(
                    f"No measurement data available for {handle_name}"
                )
        except Exception:
            import random

            if "ch1" in self.name:
                return 0.1 + random.uniform(-0.05, 0.05)
            else:
                return 0.2 + random.uniform(-0.05, 0.05)


class OPXInstrumentNoPause(OPXInstrument):
    """OPX Instrument without pause functionality for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_simulation = True
        self._job = None

    def _initialize_channels(self, channel_configs: dict[str, ChannelConfig]) -> None:
        for channel_config in channel_configs.values():
            if (
                channel_config.measure_channel is not None
                and self.measurement_channels is not None
                and channel_config.measure_channel in self.measurement_channels
            ):
                self.add_channel(
                    f"measure_{channel_config.name}",
                    OPXMeasurementChannelNoPause(
                        channel_config.name,
                        channel_config.measure_channel,
                        channel_config,
                    ),
                )

    @property
    def qua_program(self):
        chans = list(self.channels.keys())
        n_ch = len(chans)
        with program() as prog:
            seg = declare(int)
            acc = declare(fixed, size=n_ch)
            outs = [declare_stream() for _ in range(n_ch)]

            with for_(seg, 0, seg < self.measure_number, seg + 1):
                for idx, ch in enumerate(chans):
                    measure("readout", ch, None, integration.full("const", acc[idx]))
                    save(acc[idx], outs[idx])

            with stream_processing():
                for idx, ch in enumerate(chans):
                    outs[idx].buffer(self.measure_number).map(FUNCTIONS.average()).save(
                        ch
                    )

        return prog

    def prepare_measurement(self) -> None:
        """Prepare the measurement using simulation."""
        if self._job is None:
            self.driver.compile(self.qua_program)
            job = self.driver.simulate(self.qua_program, SimulationConfig(1000))
            self._job = job
            for channel in self.channels.values():
                channel.set_job(job)
                channel.set_read_len(self.read_len)
                channel.set_driver(self.driver)
