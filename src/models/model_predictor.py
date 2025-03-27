import os
import pickle
import pandas as pd
from typing import List, Dict, Any
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import setup_logger
from src.utils.exceptions import ModelPredictionError

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
        self.label_encoders = {}
        
        # Load the model and feature list
        self._load_model()
        self.feature_list = self.load_feature_list()

    def _load_model(self) -> None:
        """
        Load the trained model from disk
        """
        try:
            model_name = self.config['model_config']['model_name']
            model_path = os.path.join(self.config['paths']['saved_models_dir'], f'{model_name}_model.pkl')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise ModelPredictionError(f"Failed to load model: {str(e)}")

    def load_feature_list(self) -> List[str]:
        """
        Load the list of features used during training
        
        Returns:
            List[str]: List of feature names
        """
        try:
            feature_list_path = os.path.join(self.config['paths']['features_dir'], 'feature_list.txt')
            
            if not os.path.exists(feature_list_path):
                self.logger.error(f"Feature list file not found at {feature_list_path}")
                raise FileNotFoundError(f"Feature list file not found at {feature_list_path}")
            
            with open(feature_list_path, 'r') as f:
                feature_list = [line.strip() for line in f.readlines()]
            
            self.logger.info(f"Loaded {len(feature_list)} features from {feature_list_path}")
            return feature_list
            
        except Exception as e:
            self.logger.error(f"Error loading feature list: {str(e)}")
            raise ModelPredictionError(f"Failed to load feature list: {str(e)}")

    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data to match training format
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        try:
            # Ensure all required features are present
            missing_features = set(self.feature_list) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select only the features used during training
            X_processed = X[self.feature_list].copy()
            
            # Encode categorical variables
            categorical_columns = X_processed.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit the encoder on the current data
                    self.label_encoders[col].fit(X_processed[col])
                X_processed[col] = self.label_encoders[col].transform(X_processed[col])
            
            return X_processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise ModelPredictionError(f"Failed to preprocess data: {str(e)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted labels
        """
        try:
            # Preprocess the input data
            X_processed = self.preprocess_data(X)
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            
            self.logger.info(f"Successfully made predictions for {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise ModelPredictionError(f"Failed to make predictions: {str(e)}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions for each class
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Probability predictions
        """
        try:
            # Preprocess the input data
            X_processed = self.preprocess_data(X)
            
            # Get probability predictions
            probabilities = self.model.predict_proba(X_processed)
            
            self.logger.info(f"Successfully generated probability predictions for {len(probabilities)} samples")
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error generating probability predictions: {str(e)}")
            raise ModelPredictionError(f"Failed to generate probability predictions: {str(e)}") 