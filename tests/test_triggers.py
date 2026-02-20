import pytest

from stanza.triggers import (
    TriggerConfig,
    TriggerLink,
    TriggerMode,
    hardware_in_trigger,
    hardware_out_trigger,
    software_trigger,
    timed_trigger,
)


def test_software_trigger():
    cfg = software_trigger()
    assert cfg.mode == TriggerMode.SOFTWARE
    assert cfg.digital_port is None
    assert cfg.interval_cycles is None


def test_timed_trigger():
    cfg = timed_trigger(interval_ns=1000, repetitions=50)
    assert cfg.mode == TriggerMode.TIMED
    assert cfg.interval_cycles == 250  # 1000 / 4
    assert cfg.repetitions == 50


def test_timed_trigger_raises_on_misaligned_interval():
    with pytest.raises(ValueError, match="not aligned"):
        timed_trigger(interval_ns=101)


def test_hardware_out_trigger():
    port = ("con1", 2, 1)
    cfg = hardware_out_trigger(digital_port=port, duration_ns=200)
    assert cfg.mode == TriggerMode.HARDWARE_OUT
    assert cfg.digital_port == port
    assert cfg.trigger_duration_cycles == 50  # 200 / 4


def test_hardware_out_raises_on_no_port():
    with pytest.raises(ValueError, match="HARDWARE_OUT requires digital_port"):
        TriggerConfig(mode=TriggerMode.HARDWARE_OUT)


def test_hardware_in_trigger():
    port = ("con1", 2, 3)
    cfg = hardware_in_trigger(digital_port=port, timeout_ns=400, debounce_ns=40)
    assert cfg.mode == TriggerMode.HARDWARE_IN
    assert cfg.digital_port == port
    assert cfg.input_timeout_cycles == 100  # 400 / 4
    assert cfg.input_debounce_cycles == 10  # 40 / 4


def test_hardware_in_trigger_no_timeout():
    port = ("con1", 2, 3)
    cfg = hardware_in_trigger(digital_port=port)
    assert cfg.input_timeout_cycles is None
    assert cfg.input_debounce_cycles == 0


def test_hardware_in_raises_on_no_port():
    with pytest.raises(ValueError, match="HARDWARE_IN requires digital_port"):
        TriggerConfig(mode=TriggerMode.HARDWARE_IN)


def test_timed_raises_on_no_interval():
    with pytest.raises(ValueError, match="TIMED requires interval_cycles"):
        TriggerConfig(mode=TriggerMode.TIMED)


def test_timed_raises_on_negative_interval():
    with pytest.raises(ValueError, match="interval_cycles must be positive"):
        TriggerConfig(mode=TriggerMode.TIMED, interval_cycles=-1)


def test_trigger_duration_raises_on_nonpositive():
    with pytest.raises(ValueError, match="trigger_duration_cycles must be positive"):
        TriggerConfig(mode=TriggerMode.SOFTWARE, trigger_duration_cycles=0)


def test_timed_trigger_infinite_repetitions():
    cfg = timed_trigger(interval_ns=100)
    assert cfg.repetitions is None


def test_trigger_mode_values():
    assert TriggerMode.SOFTWARE.value == "software"
    assert TriggerMode.HARDWARE_OUT.value == "hardware_out"
    assert TriggerMode.HARDWARE_IN.value == "hardware_in"
    assert TriggerMode.TIMED.value == "timed"


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
    assert link.trigger_duration_cycles == 25


def test_trigger_link_custom_duration():
    link = TriggerLink(
        name="trig",
        source_port=("con1", 2, 1),
        sink_instrument="qdac",
        sink_channel="gate1",
        sink_trigger_port="ext2",
        trigger_duration_cycles=50,
    )
    assert link.trigger_duration_cycles == 50


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
    with pytest.raises(ValueError, match="trigger_duration_cycles must be positive"):
        TriggerLink(
            name="trig",
            source_port=("con1", 2, 1),
            sink_instrument="qdac",
            sink_channel="gate1",
            sink_trigger_port="ext1",
            trigger_duration_cycles=0,
        )
