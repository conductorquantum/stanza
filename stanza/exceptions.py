class InstrumentError(RuntimeError):
    """Exception raised when an instrument operation fails."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DeviceError(RuntimeError):
    """Exception raised when an instrument operation fails."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class LoggingError(Exception):
    """Base exception for logging-related errors."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code


class WriterError(LoggingError):
    """Raised when data writer operations fail."""

    def __init__(
        self,
        message: str,
        writer_type: str | None = None,
        file_path: str | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message, error_code)
        self.writer_type = writer_type
        self.file_path = file_path
