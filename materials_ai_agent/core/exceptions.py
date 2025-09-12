"""Custom exceptions for the Materials AI Agent."""


class MaterialsAgentError(Exception):
    """Base exception for Materials AI Agent."""
    pass


class SimulationError(MaterialsAgentError):
    """Exception raised during simulation setup or execution."""
    pass


class AnalysisError(MaterialsAgentError):
    """Exception raised during data analysis."""
    pass


class DatabaseError(MaterialsAgentError):
    """Exception raised during database operations."""
    pass


class MLModelError(MaterialsAgentError):
    """Exception raised during ML model operations."""
    pass


class ConfigurationError(MaterialsAgentError):
    """Exception raised due to configuration issues."""
    pass
