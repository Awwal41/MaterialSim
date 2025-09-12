"""
Materials AI Agent - An autonomous LLM agent for computational materials science.
"""

from .core.agent import MaterialsAgent
from .core.config import Config
from .core.exceptions import MaterialsAgentError

__version__ = "0.1.0"
__author__ = "Materials AI Agent Team"

__all__ = [
    "MaterialsAgent",
    "Config", 
    "MaterialsAgentError",
]
