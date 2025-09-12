"""Base tool class for Materials AI Agent tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.exceptions import MaterialsAgentError


class BaseMaterialsTool(BaseTool, ABC):
    """Base class for all Materials AI Agent tools."""
    
    config: Config = Field(..., description="Configuration object")
    
    model_config = {
        "ignored_types": (property,),
        "arbitrary_types_allowed": True
    }
    
    def __init__(self, config: Config, **kwargs):
        """Initialize the tool with configuration.
        
        Args:
            config: Configuration object
            **kwargs: Additional arguments
        """
        # Set config before calling super().__init__
        self.config = config
        super().__init__(**kwargs)
        self.logger = self._get_logger()
    
    def _get_logger(self):
        """Get logger for this tool."""
        import logging
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Raises:
            MaterialsAgentError: If validation fails
        """
        if not isinstance(input_data, dict):
            raise MaterialsAgentError("Input must be a dictionary")
    
    def _handle_error(self, error: Exception, context: str = "") -> str:
        """Handle errors and return formatted error message.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            
        Returns:
            Formatted error message
        """
        error_msg = f"Error in {self.__class__.__name__}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"
        
        self.logger.error(error_msg)
        return error_msg
    
    def _run(self, query: str) -> str:
        """Run the tool with a query string.
        
        Args:
            query: Query string
            
        Returns:
            Tool response as string
        """
        # Default implementation - can be overridden by subclasses
        return f"Tool {self.name} received query: {query}"
