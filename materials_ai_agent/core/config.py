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
    default_n_steps: int = Field(10000, description="Default number of simulation steps")
    default_ensemble: str = Field("NVT", description="Default thermodynamic ensemble")
    default_thermostat: str = Field("Nose-Hoover", description="Default thermostat")
    default_force_field: str = Field("tersoff", description="Default force field")
    default_structure_source: str = Field("generate", description="Default structure source")
    
    # Simulation Limits
    min_temperature: float = Field(1.0, description="Minimum temperature in K")
    max_temperature: float = Field(5000.0, description="Maximum temperature in K")
    min_timestep: float = Field(0.0001, description="Minimum timestep in ps")
    max_timestep: float = Field(0.01, description="Maximum timestep in ps")
    min_n_steps: int = Field(1000, description="Minimum number of simulation steps")
    max_n_steps: int = Field(1000000, description="Maximum number of simulation steps")
    
    # Available Options
    available_ensembles: list = Field(["NVT", "NPT", "NVE"], description="Available thermodynamic ensembles")
    available_thermostats: list = Field(["Nose-Hoover", "Berendsen", "Langevin"], description="Available thermostats")
    available_force_fields: list = Field(["tersoff", "eam", "lj", "reaxff"], description="Available force fields")
    available_structure_sources: list = Field(["generate", "upload", "material_project"], description="Available structure sources")
    
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
            default_n_steps=int(os.getenv("DEFAULT_N_STEPS", "10000")),
            default_ensemble=os.getenv("DEFAULT_ENSEMBLE", "NVT"),
            default_thermostat=os.getenv("DEFAULT_THERMOSTAT", "Nose-Hoover"),
            default_force_field=os.getenv("DEFAULT_FORCE_FIELD", "tersoff"),
            default_structure_source=os.getenv("DEFAULT_STRUCTURE_SOURCE", "generate"),
            min_temperature=float(os.getenv("MIN_TEMPERATURE", "1.0")),
            max_temperature=float(os.getenv("MAX_TEMPERATURE", "5000.0")),
            min_timestep=float(os.getenv("MIN_TIMESTEP", "0.0001")),
            max_timestep=float(os.getenv("MAX_TIMESTEP", "0.01")),
            min_n_steps=int(os.getenv("MIN_N_STEPS", "1000")),
            max_n_steps=int(os.getenv("MAX_N_STEPS", "1000000")),
            available_ensembles=os.getenv("AVAILABLE_ENSEMBLES", "NVT,NPT,NVE").split(","),
            available_thermostats=os.getenv("AVAILABLE_THERMOSTATS", "Nose-Hoover,Berendsen,Langevin").split(","),
            available_force_fields=os.getenv("AVAILABLE_FORCE_FIELDS", "tersoff,eam,lj,reaxff").split(","),
            available_structure_sources=os.getenv("AVAILABLE_STRUCTURE_SOURCES", "generate,upload,material_project").split(","),
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
