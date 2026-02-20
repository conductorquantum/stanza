"""Tests for stanza.models — verifying OPX-specific fields are removed and extras pass through."""

from stanza.models import InstrumentType, MeasurementInstrumentConfig


def test_measurement_config_no_opx_fields():
    """Verify that OPX-specific fields are NOT declared on MeasurementInstrumentConfig."""
    declared_fields = MeasurementInstrumentConfig.model_fields
    assert "machine_type" not in declared_fields
    assert "cluster_name" not in declared_fields
    assert "measurement_channels" not in declared_fields
    assert "connection_headers" not in declared_fields
    assert "octave" not in declared_fields


def test_measurement_config_extra_fields_allowed():
    """Verify that extra fields (like OPX-specific ones) pass through via Pydantic extra='allow'."""
    config = MeasurementInstrumentConfig(
        name="opx_test",
        type=InstrumentType.MEASUREMENT,
        ip_addr="192.168.1.1",
        port=80,
        measurement_duration=1.0,
        sample_time=0.1,
        machine_type="OPX1000",
        cluster_name="test_cluster",
        measurement_channels=[1, 2],
        connection_headers={"key": "value"},
        octave="octave1",
    )
    assert config.machine_type == "OPX1000"
    assert config.cluster_name == "test_cluster"
    assert config.measurement_channels == [1, 2]
    assert config.connection_headers == {"key": "value"}
    assert config.octave == "octave1"
