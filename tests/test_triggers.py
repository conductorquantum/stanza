import pytest

from stanza.triggers import TriggerLink

# --- TriggerLink Tests ---


def test_trigger_link_creation():
    link = TriggerLink(
        name="qdac_ch1_trigger",
        source_port=("con1", 2, 5),
        sink_instrument="qdac",
        sink_channel="gate1",
        sink_trigger_port="ext1",
    )
    assert link.name == "qdac_ch1_trigger"
    assert link.source_port == ("con1", 2, 5)
    assert link.sink_instrument == "qdac"
    assert link.sink_channel == "gate1"
    assert link.sink_trigger_port == "ext1"


def test_trigger_link_frozen():
    link = TriggerLink(
        name="trig",
        source_port=("con1", 2, 1),
        sink_instrument="qdac",
        sink_channel="gate1",
        sink_trigger_port="ext1",
    )
    with pytest.raises(AttributeError):
        link.name = "other"  # type: ignore[misc]


def test_trigger_link_default_duration():
    link = TriggerLink(
        name="trig",
        source_port=("con1", 2, 1),
        sink_instrument="qdac",
        sink_channel="gate1",
        sink_trigger_port="ext1",
    )
    assert link.trigger_duration_ns == 100


def test_trigger_link_custom_duration():
    link = TriggerLink(
        name="trig",
        source_port=("con1", 2, 1),
        sink_instrument="qdac",
        sink_channel="gate1",
        sink_trigger_port="ext2",
        trigger_duration_ns=200,
    )
    assert link.trigger_duration_ns == 200


def test_trigger_link_accepts_arbitrary_trigger_port():
    """sink_trigger_port is no longer validated at the TriggerLink level."""
    link = TriggerLink(
        name="trig",
        source_port=("con1", 2, 1),
        sink_instrument="qdac",
        sink_channel="gate1",
        sink_trigger_port="ext5",
    )
    assert link.sink_trigger_port == "ext5"


def test_trigger_link_generic_source_port():
    """source_port accepts arbitrary tuple of str|int."""
    link = TriggerLink(
        name="trig",
        source_port=("device", 3),
        sink_instrument="qdac",
        sink_channel="gate1",
        sink_trigger_port="ext1",
    )
    assert link.source_port == ("device", 3)


def test_trigger_link_invalid_duration():
    with pytest.raises(ValueError, match="trigger_duration_ns must be positive"):
        TriggerLink(
            name="trig",
            source_port=("con1", 2, 1),
            sink_instrument="qdac",
            sink_channel="gate1",
            sink_trigger_port="ext1",
            trigger_duration_ns=0,
        )
