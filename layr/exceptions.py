class LayrError(Exception):
    """Base Layr SDK exception."""


class ExportError(LayrError):
    """Raised when an exporter fails in non-silent mode."""


class ConfigurationError(LayrError):
    """Raised for invalid SDK configuration."""


class EvidenceError(LayrError):
    """Raised when an evidence payload cannot be hashed or a chain is malformed."""


class ReviewError(LayrError):
    """Raised when a human-review request or decision is invalid."""
