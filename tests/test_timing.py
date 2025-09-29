import datetime

from stanza.timing import seconds_to_ns, to_epoch


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
