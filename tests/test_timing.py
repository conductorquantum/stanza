import datetime

import pytest

from stanza.timing import (
    cycles_to_ns,
    ns_to_cycles,
    ns_to_samples,
    samples_to_ns,
    seconds_to_ns,
    to_epoch,
)


class TestSecondsToNs:
    def test_converts_seconds_to_nanoseconds(self):
        assert seconds_to_ns(1.0) == 1_000_000_000
        assert seconds_to_ns(0.001) == 1_000_000
        assert seconds_to_ns(2.5) == 2_500_000_000

    def test_handles_negative_and_zero(self):
        assert seconds_to_ns(0.0) == 0
        assert seconds_to_ns(-1.5) == -1_500_000_000


class TestToEpoch:
    def test_passthrough_numeric_timestamps(self):
        assert to_epoch(1234567890.0) == 1234567890.0
        assert to_epoch(1234567890) == 1234567890.0

    def test_converts_datetime_to_epoch(self):
        dt = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
        epoch = to_epoch(dt)
        assert isinstance(epoch, float)
        assert epoch == dt.timestamp()


# --- Timing conversion helpers ---


def test_ns_to_cycles():
    assert ns_to_cycles(100) == 25
    assert ns_to_cycles(0) == 0
    assert ns_to_cycles(4) == 1


def test_ns_to_cycles_rejects_unaligned():
    with pytest.raises(ValueError, match="not aligned"):
        ns_to_cycles(5)


def test_cycles_to_ns():
    assert cycles_to_ns(25) == 100
    assert cycles_to_ns(0) == 0
    assert cycles_to_ns(1) == 4


def test_ns_to_samples_default_1ghz():
    assert ns_to_samples(100) == 100
    assert ns_to_samples(4) == 4


def test_ns_to_samples_custom_rate():
    assert ns_to_samples(100, sample_rate_hz=2e9) == 200


def test_samples_to_ns_default_1ghz():
    assert samples_to_ns(100) == 100
    assert samples_to_ns(4) == 4


def test_samples_to_ns_custom_rate():
    assert samples_to_ns(200, sample_rate_hz=2e9) == 100


def test_roundtrip_ns_cycles():
    for ns in [0, 4, 100, 400]:
        assert cycles_to_ns(ns_to_cycles(ns)) == ns


def test_roundtrip_ns_samples():
    for ns in [0, 4, 100, 400]:
        assert samples_to_ns(ns_to_samples(ns)) == ns
