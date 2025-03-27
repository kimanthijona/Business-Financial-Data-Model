import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
from sklearn.metrics import accuracy_score

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.exceptions import ModelTrainingError

logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainer class
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = logger
        
        # Initialize model dictionary
        self.models = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'decision_tree': DecisionTreeClassifier,
            'naive_bayes': GaussianNB
        }
        
        # Create models directory if it doesn't exist
        os.makedirs(self.config['paths']['saved_models_dir'], exist_ok=True)
        
        # Initialize label encoders dictionary
        self.label_encoders = {}

    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by encoding categorical variables
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        try:
            X_processed = X.copy()
            
            # Identify categorical columns
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            # Encode each categorical column
            for col in categorical_columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col])
            
            return X_processed
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise ModelTrainingError(f"Failed to preprocess data: {str(e)}")

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train a model using the provided features and target
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Any: Trained model
        """
        try:
            # Get model configuration
            model_name = self.config['model_config']['model_name']
            model_params = self.config['model_config']['parameters']
            grid_search_params = self.config['model_config']['grid_search_params']
            
            # Initialize model
            if model_name == 'random_forest':
                model = RandomForestClassifier(**model_params)
            elif model_name == 'logistic_regression':
                model = LogisticRegression(**model_params)
            else:
                raise ValueError(f"Unsupported model type: {model_name}")
            
            # Perform grid search
            grid_search = GridSearchCV(
                model,
                param_grid=grid_search_params,
                cv=self.config['training']['cv_folds'],
                scoring=self.config['training']['scoring'],
                n_jobs=self.config['training']['n_jobs']
            )
            
            # Fit the model
            grid_search.fit(X, y)
            
            # Log best parameters
            self.logger.info(f"Best score: {grid_search.best_score_:.4f}")
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            
            # Save the model
            model_path = os.path.join(self.config['paths']['saved_models_dir'], f'{model_name}_model.pkl')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(grid_search.best_estimator_, f)
            
            # Save feature list
            feature_list_path = os.path.join(self.config['paths']['features_dir'], 'feature_list.txt')
            os.makedirs(os.path.dirname(feature_list_path), exist_ok=True)
            
            with open(feature_list_path, 'w') as f:
                f.write('\n'.join(X.columns.tolist()))
            
            self.logger.info(f"Model and feature list saved to {self.config['paths']['saved_models_dir']}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise ModelTrainingError(f"Failed to train model: {str(e)}")

    def _get_model_class(self, model_name: str) -> Any:
        """
        Get the model class based on model name
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Any: Model class
        """
        model_classes = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'neural_network': MLPClassifier,
            'svm': SVC,
            'decision_tree': DecisionTreeClassifier,
            'naive_bayes': GaussianNB
        }
        
        if model_name not in model_classes:
            raise ValueError(f"Unknown model type: {model_name}")
        
        return model_classes[model_name]

    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            Any: Loaded model
        """
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise ModelTrainingError(f"Failed to load model: {str(e)}")

def main():
    """Main function to test the ModelTrainer class"""
    try:
        # Example configuration
        config = {
            'model_config': {
                'model_name': 'random_forest',
                'parameters': {
                    'random_state': 42
                },
                'grid_search_params': {
                    'n_estimators': [100],
                    'max_depth': [5, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'training': {
                'cv_folds': 5,
                'scoring': 'accuracy',
                'n_jobs': -1
            },
            'paths': {
                'saved_models_dir': 'models/saved_models'
            }
        }
        
        # Load training data
        train_file = 'data/processed/train_data.csv'
        df = pd.read_csv(train_file)
        X = df.drop(columns=['qualified_status'])
        y = df['qualified_status']
        
        # Initialize and train model
        trainer = ModelTrainer(config)
        model = trainer.train_model(X, y)
        
        # Test prediction
        test_file = 'data/processed/test_data.csv'
        test_df = pd.read_csv(test_file)
        X_test = test_df.drop(columns=['qualified_status'])
        y_test = test_df['qualified_status']
        
        # Preprocess test data
        X_test_processed = trainer.preprocess_data(X_test)
        y_pred = model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 