import os

import pytest
from dotenv import load_dotenv
from qm_saas import QmSaas, QOPVersion

from stanza.base.channels import ChannelConfig
from stanza.models import GateType, InstrumentType, MeasurementInstrumentConfig, PadType
from tests.test_opx_cloud_sim.conftest import OPXInstrumentNoPause


@pytest.fixture
def opx_sim():
    # Load environment variables from .env file
    load_dotenv()

    email = os.getenv("QM_SAAS_EMAIL")
    password = os.getenv("QM_SAAS_PASSWORD")

    if not email or not password:
        pytest.skip("QM_SAAS_EMAIL and QM_SAAS_PASSWORD must be set in .env file")

    client = QmSaas(email=email, password=password)
    version = QOPVersion("v3_5_0")
    with client.simulator(version) as instance:
        instrument_config = MeasurementInstrumentConfig(
            name="opx_sim",
            type=InstrumentType.MEASUREMENT,
            ip_addr=instance.host,
            port=instance.port,
            connection_headers=instance.default_connection_headers,
            machine_type="opx1000",
            sample_time=0.001,
            measurement_duration=0.1,
            measurement_channels=[1, 2],
        )
        channel_configs = {
            "ch1": ChannelConfig(
                name="ch1",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.GATE,
                electrode_type=GateType.PLUNGER,
                measure_channel=1,
            ),
            "ch2": ChannelConfig(
                name="ch2",
                voltage_range=(-1.0, 1.0),
                pad_type=PadType.GATE,
                electrode_type=GateType.BARRIER,
                measure_channel=2,
            ),
        }
        yield OPXInstrumentNoPause(instrument_config, channel_configs)


def test_opx_simulator_loads(opx_sim):
    """Test that the OPX simulator loads successfully."""
    assert opx_sim is not None
    assert hasattr(opx_sim, "driver")
    assert opx_sim.qua_config["version"] == "1"


def test_opx_simulator_measures(opx_sim):
    """Test that the OPX simulator measures successfully."""
    opx_sim.prepare_measurement()

    result_ch1 = opx_sim.measure("ch1")
    result_ch2 = opx_sim.measure("ch2")

    assert isinstance(result_ch1, float), f"Expected float, got {type(result_ch1)}"
    assert isinstance(result_ch2, float), f"Expected float, got {type(result_ch2)}"

    opx_sim.teardown_measurement()
