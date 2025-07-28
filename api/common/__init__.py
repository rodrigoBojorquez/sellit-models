class Error(Exception):
    """Base class for all exceptions in this module."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class ValidationError(Error):
    """Exception raised for validation errors."""
    pass

class NotFoundError(Error):
    """Exception raised when a resource is not found."""
    pass

class ConflictError(Error):
    """Exception raised for conflicts, such as duplicate entries."""
    pass