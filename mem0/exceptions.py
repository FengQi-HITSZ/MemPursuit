"""Structured validation errors retained from the memory backbone."""

from typing import Any, Dict, Optional


class MemoryError(Exception):
    """Base exception for all memory-related errors.

    This is the base class for all Mem0-specific exceptions. It provides a structured
    approach to error handling with error codes, contextual details, suggestions for
    resolution, and debug information.

    Attributes:
        message (str): Human-readable error message.
        error_code (str): Unique error identifier for programmatic handling.
        details (dict): Additional context about the error.
        suggestion (str): User-friendly suggestion for resolving the error.
        debug_info (dict): Technical debugging information.

    Example:
        raise MemoryError(
            message="Memory operation failed",
            error_code="MEM_001",
            details={"operation": "add", "user_id": "user123"},
            suggestion="Please check your API key and try again",
            debug_info={"request_id": "req_456", "timestamp": "2024-01-01T00:00:00Z"}
        )
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a MemoryError.

        Args:
            message: Human-readable error message.
            error_code: Unique error identifier.
            details: Additional context about the error.
            suggestion: User-friendly suggestion for resolving the error.
            debug_info: Technical debugging information.
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.suggestion = suggestion
        self.debug_info = debug_info or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"details={self.details!r}, "
            f"suggestion={self.suggestion!r}, "
            f"debug_info={self.debug_info!r})"
        )


class ValidationError(MemoryError):
    """Raised when input validation fails.

    This exception is raised when request parameters, memory content,
    or configuration values fail validation checks.

    Common scenarios:
        - Invalid user_id format
        - Missing required fields
        - Content too long or too short
        - Invalid metadata format
        - Malformed filters

    Example:
        raise ValidationError(
            message="Invalid user_id format",
            error_code="VAL_001",
            details={"field": "user_id", "value": "123", "expected": "string"},
            suggestion="User ID must be a non-empty string"
        )
    """

    pass
