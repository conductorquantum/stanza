class InstrumentError(RuntimeError):
    """Exception raised when an instrument operation fails."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DeviceError(RuntimeError):
    """Exception raised when a device operation fails."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class LoggingError(Exception):
    """Base exception for logging-related errors."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code


class RoutineError(Exception):
    """Exception raised when a routine operation fails."""

    def __init__(self, message: str):
        super().__init__(message)


class LoggerSessionError(LoggingError):
    """Exception raised when a logging session operation fails."""

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message, error_code)
        self.session_id = session_id


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
