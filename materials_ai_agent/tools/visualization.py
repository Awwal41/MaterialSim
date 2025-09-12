"""Visualization tools for creating plots and reports."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json

from langchain.tools import tool
from ase import Atoms
from ase.visualize.plot import plot_atoms
import matplotlib.patches as mpatches

from .base import BaseMaterialsTool
from ..core.exceptions import AnalysisError


class VisualizationTool(BaseMaterialsTool):
    """Tool for creating visualizations and reports."""
    
    name: str = "visualization"
    description: str = "Create visualizations and reports for simulation results"
    
    def __init__(self, config):
        super().__init__(config)
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_property_dashboard(
        self,
        simulation_data: Dict[str, Any],
        output_file: str = "property_dashboard.html"
    ) -> Dict[str, Any]:
        """Create an interactive dashboard of material properties.
        
        Args:
            simulation_data: Dictionary containing simulation results
            output_file: Output HTML file name
            
        Returns:
            Dictionary containing dashboard information
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Temperature vs Time', 'Pressure vs Time', 
                              'Energy vs Time', 'Volume vs Time'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Extract data
            if "thermo_data" in simulation_data:
                df = simulation_data["thermo_data"]
                
                # Temperature plot
                if "Temp" in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df["Step"], y=df["Temp"], 
                                 mode='lines', name='Temperature'),
                        row=1, col=1
                    )
                
                # Pressure plot
                if "Press" in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df["Step"], y=df["Press"], 
                                 mode='lines', name='Pressure'),
                        row=1, col=2
                    )
                
                # Energy plot
                if "TotEng" in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df["Step"], y=df["TotEng"], 
                                 mode='lines', name='Total Energy'),
                        row=2, col=1
                    )
                
                # Volume plot
                if "Volume" in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df["Step"], y=df["Volume"], 
                                 mode='lines', name='Volume'),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title="Materials Simulation Dashboard",
                showlegend=True,
                height=800,
                width=1200
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Time Step", row=2, col=1)
            fig.update_xaxes(title_text="Time Step", row=2, col=2)
            fig.update_yaxes(title_text="Temperature (K)", row=1, col=1)
            fig.update_yaxes(title_text="Pressure (atm)", row=1, col=2)
            fig.update_yaxes(title_text="Energy (eV)", row=2, col=1)
            fig.update_yaxes(title_text="Volume (Å³)", row=2, col=2)
            
            # Save dashboard
            output_path = self.config.visualization_output_dir / output_file
            fig.write_html(str(output_path))
            
            return {
                "success": True,
                "dashboard_file": str(output_path),
                "plot_type": "interactive_dashboard"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "create_property_dashboard")
            }
    
    def plot_structure_3d(
        self,
        structure: Dict[str, Any],
        output_file: str = "structure_3d.html"
    ) -> Dict[str, Any]:
        """Create 3D visualization of atomic structure.
        
        Args:
            structure: Structure dictionary or ASE Atoms object
            output_file: Output HTML file name
            
        Returns:
            Dictionary containing visualization information
        """
        try:
            # Convert structure to ASE Atoms if needed
            if isinstance(structure, dict):
                atoms = Atoms.from_dict(structure)
            else:
                atoms = structure
            
            # Get atomic positions and symbols
            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            
            # Create 3D scatter plot
            fig = go.Figure()
            
            # Get unique elements and colors
            unique_symbols = list(set(symbols))
            colors = px.colors.qualitative.Set1[:len(unique_symbols)]
            symbol_colors = {sym: colors[i] for i, sym in enumerate(unique_symbols)}
            
            # Add atoms
            for symbol in unique_symbols:
                mask = np.array(symbols) == symbol
                pos = positions[mask]
                
                fig.add_trace(go.Scatter3d(
                    x=pos[:, 0],
                    y=pos[:, 1],
                    z=pos[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=symbol_colors[symbol],
                        opacity=0.8
                    ),
                    name=symbol,
                    text=[f"{symbol}_{i}" for i in range(len(pos))],
                    hovertemplate="%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
                ))
            
            # Update layout
            fig.update_layout(
                title=f"3D Structure: {atoms.get_chemical_formula()}",
                scene=dict(
                    xaxis_title="X (Å)",
                    yaxis_title="Y (Å)",
                    zaxis_title="Z (Å)",
                    aspectmode="data"
                ),
                width=800,
                height=600
            )
            
            # Save plot
            output_path = self.config.visualization_output_dir / output_file
            fig.write_html(str(output_path))
            
            return {
                "success": True,
                "plot_file": str(output_path),
                "plot_type": "3d_structure",
                "formula": atoms.get_chemical_formula(),
                "n_atoms": len(atoms)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "plot_structure_3d")
            }
    
    def create_comparison_plot(
        self,
        data_sets: List[Dict[str, Any]],
        x_column: str,
        y_column: str,
        title: str = "Comparison Plot",
        output_file: str = "comparison_plot.png"
    ) -> Dict[str, Any]:
        """Create comparison plot of multiple data sets.
        
        Args:
            data_sets: List of data dictionaries with 'data' and 'label' keys
            x_column: Name of x-axis column
            y_column: Name of y-axis column
            title: Plot title
            output_file: Output file name
            
        Returns:
            Dictionary containing plot information
        """
        try:
            plt.figure(figsize=(12, 8))
            
            for i, data_set in enumerate(data_sets):
                df = data_set["data"]
                label = data_set.get("label", f"Dataset {i+1}")
                
                if x_column in df.columns and y_column in df.columns:
                    plt.plot(df[x_column], df[y_column], 
                            label=label, linewidth=2, marker='o', markersize=4)
            
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            output_path = self.config.visualization_output_dir / output_file
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "plot_file": str(output_path),
                "plot_type": "comparison",
                "n_datasets": len(data_sets)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "create_comparison_plot")
            }
    
    def create_heatmap(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        x_labels: List[str] = None,
        y_labels: List[str] = None,
        title: str = "Heatmap",
        output_file: str = "heatmap.png"
    ) -> Dict[str, Any]:
        """Create heatmap visualization.
        
        Args:
            data: 2D data array or DataFrame
            x_labels: X-axis labels
            y_labels: Y-axis labels
            title: Plot title
            output_file: Output file name
            
        Returns:
            Dictionary containing plot information
        """
        try:
            plt.figure(figsize=(10, 8))
            
            if isinstance(data, pd.DataFrame):
                sns.heatmap(data, annot=True, fmt='.2f', cmap='viridis')
            else:
                sns.heatmap(data, annot=True, fmt='.2f', cmap='viridis',
                           xticklabels=x_labels, yticklabels=y_labels)
            
            plt.title(title)
            plt.tight_layout()
            
            # Save plot
            output_path = self.config.visualization_output_dir / output_file
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "plot_file": str(output_path),
                "plot_type": "heatmap"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "create_heatmap")
            }
    
    def create_property_correlation_matrix(
        self,
        data: pd.DataFrame,
        properties: List[str] = None,
        output_file: str = "correlation_matrix.png"
    ) -> Dict[str, Any]:
        """Create correlation matrix heatmap for material properties.
        
        Args:
            data: DataFrame containing property data
            properties: List of properties to include (None for all numeric)
            output_file: Output file name
            
        Returns:
            Dictionary containing plot information
        """
        try:
            # Select numeric columns
            if properties:
                numeric_data = data[properties]
            else:
                numeric_data = data.select_dtypes(include=[np.number])
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='RdBu_r', center=0, square=True)
            
            plt.title("Property Correlation Matrix")
            plt.tight_layout()
            
            # Save plot
            output_path = self.config.visualization_output_dir / output_file
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "plot_file": str(output_path),
                "plot_type": "correlation_matrix",
                "properties": list(numeric_data.columns)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "create_property_correlation_matrix")
            }
    
    def create_ml_performance_plot(
        self,
        training_history: Dict[str, List[float]],
        output_file: str = "ml_performance.png"
    ) -> Dict[str, Any]:
        """Create ML model performance visualization.
        
        Args:
            training_history: Dictionary with 'train_loss', 'val_loss', etc.
            output_file: Output file name
            
        Returns:
            Dictionary containing plot information
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Training/validation loss
            if 'train_loss' in training_history and 'val_loss' in training_history:
                axes[0, 0].plot(training_history['train_loss'], label='Training Loss')
                axes[0, 0].plot(training_history['val_loss'], label='Validation Loss')
                axes[0, 0].set_title('Training and Validation Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy (if available)
            if 'train_acc' in training_history and 'val_acc' in training_history:
                axes[0, 1].plot(training_history['train_acc'], label='Training Accuracy')
                axes[0, 1].plot(training_history['val_acc'], label='Validation Accuracy')
                axes[0, 1].set_title('Training and Validation Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate (if available)
            if 'lr' in training_history:
                axes[1, 0].plot(training_history['lr'])
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Prediction vs actual (if available)
            if 'predictions' in training_history and 'actual' in training_history:
                pred = training_history['predictions']
                actual = training_history['actual']
                axes[1, 1].scatter(actual, pred, alpha=0.6)
                axes[1, 1].plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
                axes[1, 1].set_title('Predictions vs Actual')
                axes[1, 1].set_xlabel('Actual')
                axes[1, 1].set_ylabel('Predicted')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.config.visualization_output_dir / output_file
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "plot_file": str(output_path),
                "plot_type": "ml_performance"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "create_ml_performance_plot")
            }
    
    def generate_report(
        self,
        simulation_results: Dict[str, Any],
        analysis_results: Dict[str, Any] = None,
        output_file: str = "simulation_report.html"
    ) -> Dict[str, Any]:
        """Generate comprehensive HTML report.
        
        Args:
            simulation_results: Simulation results dictionary
            analysis_results: Analysis results dictionary
            output_file: Output HTML file name
            
        Returns:
            Dictionary containing report information
        """
        try:
            # Create HTML report
            html_content = self._generate_html_report(simulation_results, analysis_results)
            
            # Save report
            output_path = self.config.visualization_output_dir / output_file
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            return {
                "success": True,
                "report_file": str(output_path),
                "report_type": "html"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "generate_report")
            }
    
    def _generate_html_report(
        self,
        simulation_results: Dict[str, Any],
        analysis_results: Dict[str, Any] = None
    ) -> str:
        """Generate HTML report content.
        
        Args:
            simulation_results: Simulation results
            analysis_results: Analysis results
            
        Returns:
            HTML content as string
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Materials Simulation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .property { margin: 10px 0; }
                .value { font-weight: bold; color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Materials Simulation Report</h1>
                <p>Generated by Materials AI Agent</p>
            </div>
        """
        
        # Simulation results section
        if simulation_results:
            html += """
            <div class="section">
                <h2>Simulation Results</h2>
            """
            
            for key, value in simulation_results.items():
                if isinstance(value, (int, float, str)):
                    html += f"""
                    <div class="property">
                        <span>{key}:</span> <span class="value">{value}</span>
                    </div>
                    """
            
            html += "</div>"
        
        # Analysis results section
        if analysis_results:
            html += """
            <div class="section">
                <h2>Analysis Results</h2>
            """
            
            for key, value in analysis_results.items():
                if isinstance(value, (int, float, str)):
                    html += f"""
                    <div class="property">
                        <span>{key}:</span> <span class="value">{value}</span>
                    </div>
                    """
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
