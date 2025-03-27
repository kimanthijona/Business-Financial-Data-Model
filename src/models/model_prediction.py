import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import pickle
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.exceptions import ModelPredictionError
from src.data.automated_feature_engineering import AutomatedFeatureEngineering

logger = setup_logger(__name__)

class ModelPredictor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelPredictor class
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = logger
        self.model = None
        self.encoders = None
        self.feature_list = None
        self.predictions_dir = config['paths']['predictions_dir']
        self.auto_feature_engineering = AutomatedFeatureEngineering(max_features=50)
        os.makedirs(self.predictions_dir, exist_ok=True)

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise ModelPredictionError(f"Failed to load model: {str(e)}")

    def load_encoders(self, encoders_path: str) -> None:
        """
        Load label encoders from disk
        
        Args:
            encoders_path (str): Path to the saved encoders
        """
        try:
            with open(encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
            self.logger.info(f"Encoders loaded successfully from {encoders_path}")
        except Exception as e:
            self.logger.error(f"Error loading encoders: {str(e)}")
            raise ModelPredictionError(f"Failed to load encoders: {str(e)}")

    def load_feature_list(self, feature_list_path: str) -> None:
        """
        Load the feature list used during training
        
        Args:
            feature_list_path (str): Path to the feature list file
        """
        try:
            with open(feature_list_path, 'r') as f:
                self.feature_list = [line.strip() for line in f.readlines()]
            self.logger.info(f"Feature list loaded successfully with {len(self.feature_list)} features")
        except Exception as e:
            self.logger.error(f"Error loading feature list: {str(e)}")
            raise ModelPredictionError(f"Failed to load feature list: {str(e)}")

    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data using loaded encoders and generate automated features
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        try:
            if self.encoders is None:
                raise ModelPredictionError("Encoders not loaded")
            if self.feature_list is None:
                raise ModelPredictionError("Feature list not loaded")
            
            X_processed = X.copy()
            
            # Identify categorical columns
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            # Encode each categorical column using loaded encoders
            for col in categorical_columns:
                if col in self.encoders:
                    X_processed[col] = self.encoders[col].transform(X_processed[col])
            
            # Generate automated features
            automated_features = self.auto_feature_engineering.generate_features(X_processed)
            
            # Combine original and automated features
            X_processed = pd.concat([X_processed, automated_features], axis=1)
            
            # Ensure all required features are present
            missing_features = set(self.feature_list) - set(X_processed.columns)
            if missing_features:
                raise ModelPredictionError(f"Missing required features: {missing_features}")
            
            # Select only the features used during training
            X_processed = X_processed[self.feature_list]
            
            return X_processed
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise ModelPredictionError(f"Failed to preprocess data: {str(e)}")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Predictions with probabilities
        """
        try:
            if self.model is None:
                raise ModelPredictionError("Model not loaded")
            
            # Preprocess data
            X_processed = self.preprocess_data(X)
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(X_processed)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'prediction': predictions,
                'probability': probabilities[:, 1]
            })
            
            # Add original features
            for col in X.columns:
                results[f'original_{col}'] = X[col]
            
            return results
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise ModelPredictionError(f"Failed to make predictions: {str(e)}")

    def save_predictions(self, predictions: pd.DataFrame, filename: str) -> None:
        """
        Save predictions to CSV file
        
        Args:
            predictions (pd.DataFrame): Predictions to save
            filename (str): Name of the file to save predictions to
        """
        try:
            filepath = os.path.join(self.predictions_dir, filename)
            predictions.to_csv(filepath, index=False)
            self.logger.info(f"Predictions saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
            raise ModelPredictionError(f"Failed to save predictions: {str(e)}")

def main():
    """Main function to test the ModelPredictor class"""
    try:
        # Example configuration
        config = {
            'paths': {
                'predictions_dir': 'models/predictions',
                'saved_models_dir': 'models/saved_models',
                'features_dir': 'data/features'
            }
        }
        
        # Initialize predictor
        predictor = ModelPredictor(config)
        
        # Load model, encoders, and feature list
        model_path = os.path.join(config['paths']['saved_models_dir'], 'random_forest_model.pkl')
        encoders_path = os.path.join(config['paths']['saved_models_dir'], 'random_forest_encoders.pkl')
        feature_list_path = os.path.join(config['paths']['features_dir'], 'feature_list.txt')
        
        predictor.load_model(model_path)
        predictor.load_encoders(encoders_path)
        predictor.load_feature_list(feature_list_path)
        
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'age': [35, 28, 45],
            'years_experience': [5, 2, 10],
            'years_location': [3, 2, 7],
            'weekday_hours': [8, 10, 12],
            'weekend_hours': [6, 8, 10],
            'total_inventory': [5000, 3000, 8000],
            'monthly_purchases': [2000, 1500, 3500],
            'daily_sales': [6000, 4000, 7500],
            'monthly_revenue': [180000, 120000, 225000],
            'total_employees': [3, 2, 5],
            'business_type': ['retail', 'service', 'wholesale'],
            'location_type': ['mall', 'street', 'standalone'],
            'ownership_type': ['sole_proprietor', 'partnership', 'limited_company'],
            'education_level': ['bachelors', 'diploma', 'masters'],
            'customer_segment': ['middle_income', 'low_income', 'high_income'],
            'payment_methods': ['cash_mobile', 'cash_only', 'all_methods'],
            'business_registration': ['registered', 'not_registered', 'registered']
        })
        
        # Make predictions
        predictions = predictor.predict(sample_data)
        
        # Save predictions
        predictor.save_predictions(predictions, 'sample_predictions.csv')
        
        print("\nPredictions:")
        print(predictions[['prediction', 'probability']])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 