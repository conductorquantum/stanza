from stanza.exceptions import DeviceError, InstrumentError, LoggingError, WriterError


class TestInstrumentError:
    def test_error_with_message(self):
        error = InstrumentError("Connection failed")
        assert error.message == "Connection failed"
        assert str(error) == "Connection failed"
        assert isinstance(error, RuntimeError)


class TestDeviceError:
    def test_error_with_message(self):
        error = DeviceError("Device not configured")
        assert error.message == "Device not configured"
        assert str(error) == "Device not configured"
        assert isinstance(error, RuntimeError)


class TestLoggingError:
    def test_error_with_code(self):
        error = LoggingError("Write failed", error_code="LOG_001")
        assert str(error) == "Write failed"
        assert error.error_code == "LOG_001"


class TestWriterError:
    def test_error_with_metadata(self):
        error = WriterError(
            "Cannot write to file",
            writer_type="HDF5",
            file_path="/tmp/data.h5",
            error_code="WR_001",
        )
        assert str(error) == "Cannot write to file"
        assert error.writer_type == "HDF5"
        assert error.file_path == "/tmp/data.h5"
        assert error.error_code == "WR_001"
        assert isinstance(error, LoggingError)
