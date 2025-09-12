"""Machine learning tools for property prediction and model training."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import json

from langchain.tools import tool
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseMaterialsTool
from ..core.exceptions import MLModelError


class MLTool(BaseMaterialsTool):
    """Tool for machine learning-based property prediction."""
    
    name: str = "ml"
    description: str = "Machine learning tools for property prediction and model training"
    
    def __init__(self, config):
        super().__init__(config)
        self.models = {}
        self.scalers = {}
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
    def train_property_predictor(
        self,
        training_data: str,
        target_property: str,
        model_type: str = "random_forest",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train a machine learning model for property prediction.
        
        Args:
            training_data: Path to training data CSV file
            target_property: Name of target property column
            model_type: Type of model ('random_forest', 'gradient_boosting', 'neural_network')
            test_size: Fraction of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing training results
        """
        try:
            # Load training data
            df = pd.read_csv(training_data)
            
            if target_property not in df.columns:
                return {
                    "success": False,
                    "error": f"Target property '{target_property}' not found in data"
                }
            
            # Prepare features and target
            feature_columns = [col for col in df.columns if col != target_property]
            X = df[feature_columns].values
            y = df[target_property].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self._create_model(model_type)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Save model and scaler
            model_name = f"{target_property}_{model_type}"
            model_path = self.model_dir / f"{model_name}_model.joblib"
            scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Store in memory
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            return {
                "success": True,
                "model_name": model_name,
                "model_type": model_type,
                "target_property": target_property,
                "test_mse": mse,
                "test_r2": r2,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "n_features": len(feature_columns),
                "n_samples": len(df),
                "model_path": str(model_path),
                "scaler_path": str(scaler_path)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "train_property_predictor")
            }
    
    def predict_property(
        self,
        model_name: str,
        features: List[float]
    ) -> Dict[str, Any]:
        """Predict property using trained model.
        
        Args:
            model_name: Name of trained model
            features: Feature values for prediction
            
        Returns:
            Dictionary containing prediction and uncertainty
        """
        try:
            if model_name not in self.models:
                # Try to load model from disk
                model_path = self.model_dir / f"{model_name}_model.joblib"
                scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
                
                if not model_path.exists():
                    return {
                        "success": False,
                        "error": f"Model '{model_name}' not found"
                    }
                
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                self.models[model_name] = model
                self.scalers[model_name] = scaler
            
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            # Scale features
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Estimate uncertainty (simplified)
            if hasattr(model, 'predict_proba'):
                # For models that support probability prediction
                uncertainty = np.std(model.predict_proba(features_scaled))
            else:
                # Use model variance if available
                if hasattr(model, 'estimators_'):
                    # For ensemble models
                    predictions = [est.predict(features_scaled)[0] for est in model.estimators_]
                    uncertainty = np.std(predictions)
                else:
                    uncertainty = 0.1  # Default uncertainty
            
            return {
                "success": True,
                "model_name": model_name,
                "prediction": float(prediction),
                "uncertainty": float(uncertainty),
                "features": features
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "predict_property")
            }
    
    def train_neural_network(
        self,
        training_data: str,
        target_property: str,
        hidden_layers: List[int] = [100, 50],
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Train a neural network for property prediction.
        
        Args:
            training_data: Path to training data CSV file
            target_property: Name of target property column
            hidden_layers: List of hidden layer sizes
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training results
        """
        try:
            # Load training data
            df = pd.read_csv(training_data)
            
            if target_property not in df.columns:
                return {
                    "success": False,
                    "error": f"Target property '{target_property}' not found in data"
                }
            
            # Prepare features and target
            feature_columns = [col for col in df.columns if col != target_property]
            X = df[feature_columns].values
            y = df[target_property].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features and target
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            X_train_scaled = X_scaler.fit_transform(X_train)
            X_test_scaled = X_scaler.transform(X_test)
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.FloatTensor(y_train_scaled)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_test_tensor = torch.FloatTensor(y_test_scaled)
            
            # Create neural network
            input_size = X_train_scaled.shape[1]
            model = self._create_neural_network(input_size, hidden_layers)
            
            # Define loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            train_losses = []
            test_losses = []
            
            for epoch in range(epochs):
                # Training
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
                # Testing
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    test_loss = criterion(test_outputs.squeeze(), y_test_tensor)
                    test_losses.append(test_loss.item())
            
            # Final evaluation
            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(X_test_tensor).squeeze().numpy()
                y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            
            # Save model
            model_name = f"{target_property}_neural_network"
            model_path = self.model_dir / f"{model_name}_model.pth"
            scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
            
            torch.save(model.state_dict(), model_path)
            joblib.dump((X_scaler, y_scaler), scaler_path)
            
            return {
                "success": True,
                "model_name": model_name,
                "target_property": target_property,
                "test_mse": mse,
                "test_r2": r2,
                "final_train_loss": train_losses[-1],
                "final_test_loss": test_losses[-1],
                "n_features": input_size,
                "n_samples": len(df),
                "model_path": str(model_path),
                "scaler_path": str(scaler_path)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "train_neural_network")
            }
    
    def predict_with_uncertainty(
        self,
        model_name: str,
        features: List[float],
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """Predict property with uncertainty quantification.
        
        Args:
            model_name: Name of trained model
            features: Feature values for prediction
            n_samples: Number of samples for uncertainty estimation
            
        Returns:
            Dictionary containing prediction with uncertainty
        """
        try:
            if model_name not in self.models:
                # Try to load model from disk
                model_path = self.model_dir / f"{model_name}_model.joblib"
                scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
                
                if not model_path.exists():
                    return {
                        "success": False,
                        "error": f"Model '{model_name}' not found"
                    }
                
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                self.models[model_name] = model
                self.scalers[model_name] = scaler
            
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            # Scale features
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            # Bootstrap uncertainty estimation
            predictions = []
            
            if hasattr(model, 'estimators_'):
                # For ensemble models, use individual estimators
                for estimator in model.estimators_:
                    pred = estimator.predict(features_scaled)[0]
                    predictions.append(pred)
            else:
                # For single models, use bootstrap sampling
                for _ in range(n_samples):
                    # Add small random noise to features
                    noise = np.random.normal(0, 0.01, features_scaled.shape)
                    noisy_features = features_scaled + noise
                    pred = model.predict(noisy_features)[0]
                    predictions.append(pred)
            
            predictions = np.array(predictions)
            
            return {
                "success": True,
                "model_name": model_name,
                "prediction": float(np.mean(predictions)),
                "uncertainty": float(np.std(predictions)),
                "confidence_interval": {
                    "lower": float(np.percentile(predictions, 2.5)),
                    "upper": float(np.percentile(predictions, 97.5))
                },
                "features": features
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "predict_with_uncertainty")
            }
    
    def list_available_models(self) -> Dict[str, Any]:
        """List available trained models.
        
        Returns:
            Dictionary containing available models
        """
        try:
            models = []
            
            # Check memory
            for name, model in self.models.items():
                models.append({
                    "name": name,
                    "type": type(model).__name__,
                    "location": "memory"
                })
            
            # Check disk
            for model_file in self.model_dir.glob("*_model.*"):
                model_name = model_file.stem.replace("_model", "")
                models.append({
                    "name": model_name,
                    "type": "saved",
                    "location": "disk",
                    "path": str(model_file)
                })
            
            return {
                "success": True,
                "models": models,
                "n_models": len(models)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": self._handle_error(e, "list_available_models")
            }
    
    def _create_model(self, model_type: str):
        """Create model instance based on type.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        if model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == "neural_network":
            return MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_neural_network(self, input_size: int, hidden_layers: List[int]) -> nn.Module:
        """Create neural network architecture.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            
        Returns:
            PyTorch neural network
        """
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        return nn.Sequential(*layers)
