class LayrError(Exception):
    """Base Layr SDK exception."""


class ExportError(LayrError):
    """Raised when an exporter fails in non-silent mode."""


class ConfigurationError(LayrError):
    """Raised for invalid SDK configuration."""
