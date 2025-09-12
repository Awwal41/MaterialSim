"""Configuration management for the Materials AI Agent."""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration class for the Materials AI Agent."""
    
    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key")
    mp_api_key: Optional[str] = Field(None, description="Materials Project API key")
    nomad_api_key: Optional[str] = Field(None, description="NOMAD API key")
    
    # LAMMPS Configuration
    lammps_executable: str = Field("lmp", description="LAMMPS executable path")
    
    # Default Simulation Parameters
    default_temperature: float = Field(300.0, description="Default temperature in K")
    default_pressure: float = Field(1.0, description="Default pressure in atm")
    default_timestep: float = Field(0.001, description="Default timestep in ps")
    
    # Output Directories
    simulation_output_dir: Path = Field(Path("./simulations"), description="Simulation output directory")
    analysis_output_dir: Path = Field(Path("./analysis"), description="Analysis output directory")
    visualization_output_dir: Path = Field(Path("./visualizations"), description="Visualization output directory")
    
    # Logging
    log_level: str = Field("INFO", description="Logging level")
    
    # Model Configuration
    model_name: str = Field("gpt-4", description="OpenAI model name")
    max_tokens: int = Field(4000, description="Maximum tokens for LLM responses")
    temperature: float = Field(0.1, description="LLM temperature")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        load_dotenv()
        
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            mp_api_key=os.getenv("MP_API_KEY"),
            nomad_api_key=os.getenv("NOMAD_API_KEY"),
            lammps_executable=os.getenv("LAMMPS_EXECUTABLE", "lmp"),
            default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "300.0")),
            default_pressure=float(os.getenv("DEFAULT_PRESSURE", "1.0")),
            default_timestep=float(os.getenv("DEFAULT_TIMESTEP", "0.001")),
            simulation_output_dir=Path(os.getenv("SIMULATION_OUTPUT_DIR", "./simulations")),
            analysis_output_dir=Path(os.getenv("ANALYSIS_OUTPUT_DIR", "./analysis")),
            visualization_output_dir=Path(os.getenv("VISUALIZATION_OUTPUT_DIR", "./visualizations")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            model_name=os.getenv("MODEL_NAME", "gpt-4"),
            max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
        )
    
    def create_directories(self) -> None:
        """Create necessary output directories."""
        self.simulation_output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_output_dir.mkdir(parents=True, exist_ok=True)
