import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.exceptions import ModelEvaluationError

logger = setup_logger(__name__)

class ModelEvaluator:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelEvaluator class
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = logger
        self.metrics = {}
        self.evaluation_dir = config['paths']['evaluation_dir']
        os.makedirs(self.evaluation_dir, exist_ok=True)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Data loaded successfully from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise ModelEvaluationError(f"Failed to load data: {str(e)}")

    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            Any: Loaded model
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise ModelEvaluationError(f"Failed to load model: {str(e)}")

    def load_encoders(self, encoders_path: str) -> Dict[str, Any]:
        """
        Load label encoders from disk
        
        Args:
            encoders_path (str): Path to the saved encoders
            
        Returns:
            Dict[str, Any]: Loaded encoders
        """
        try:
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)
            self.logger.info(f"Encoders loaded successfully from {encoders_path}")
            return encoders
        except Exception as e:
            self.logger.error(f"Error loading encoders: {str(e)}")
            raise ModelEvaluationError(f"Failed to load encoders: {str(e)}")

    def preprocess_data(self, X: pd.DataFrame, encoders: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess the data using loaded encoders
        
        Args:
            X (pd.DataFrame): Input features
            encoders (Dict[str, Any]): Label encoders
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        try:
            X_processed = X.copy()
            
            # Identify categorical columns
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            # Encode each categorical column using loaded encoders
            for col in categorical_columns:
                if col in encoders:
                    X_processed[col] = encoders[col].transform(X_processed[col])
            
            return X_processed
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise ModelEvaluationError(f"Failed to preprocess data: {str(e)}")

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate various performance metrics
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        try:
            metrics = {}
            
            if 'accuracy' in self.metrics:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
            if 'precision' in self.metrics:
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            if 'recall' in self.metrics:
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            if 'f1' in self.metrics:
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            self.logger.info("Performance metrics calculated successfully")
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise ModelEvaluationError(f"Failed to calculate metrics: {str(e)}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            plot_path = os.path.join(self.evaluation_dir, f'{model_name}_confusion_matrix.png')
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Confusion matrix plot saved for {model_name}")
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise ModelEvaluationError(f"Failed to plot confusion matrix: {str(e)}")

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str) -> None:
        """
        Plot ROC curve
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            
            # Save plot
            plot_path = os.path.join(self.evaluation_dir, f'{model_name}_roc_curve.png')
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"ROC curve plot saved for {model_name}")
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {str(e)}")
            raise ModelEvaluationError(f"Failed to plot ROC curve: {str(e)}")

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
        """
        Generate and save classification report
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
        """
        try:
            report = classification_report(y_true, y_pred)
            
            # Save report
            report_path = os.path.join(self.evaluation_dir, f'{model_name}_classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Classification report saved for {model_name}")
        except Exception as e:
            self.logger.error(f"Error generating classification report: {str(e)}")
            raise ModelEvaluationError(f"Failed to generate classification report: {str(e)}")

    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a model comprehensively
        
        Args:
            model (Any): Trained model to evaluate
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            Dict[str, Any]: Dictionary containing evaluation results
        """
        try:
            # Make predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            self.metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'classification_report': classification_report(y, y_pred)
            }
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            self.metrics['roc_auc'] = auc(fpr, tpr)
            
            # Log metrics
            self.logger.info("\nModel Evaluation Metrics:")
            for metric, value in self.metrics.items():
                if metric != 'classification_report':
                    self.logger.info(f"{metric}: {value:.4f}")
            
            self.logger.info("\nClassification Report:")
            self.logger.info(self.metrics['classification_report'])
            
            # Generate visualizations
            model_name = model.__class__.__name__
            
            # Plot confusion matrix
            if self.config['evaluation']['plot_confusion_matrix']:
                self.plot_confusion_matrix(y, y_pred, model_name)
            
            # Plot ROC curve
            if self.config['evaluation']['plot_roc_curve']:
                self.plot_roc_curve(y, y_pred_proba, model_name)
            
            # Generate classification report
            self.generate_classification_report(y, y_pred, model_name)
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise ModelEvaluationError(f"Failed to evaluate model: {str(e)}")

    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Create a bar plot comparing model performances
        
        Args:
            results (Dict[str, Dict[str, float]]): Dictionary of evaluation results
        """
        try:
            model_names = list(results.keys())
            x = np.arange(len(model_names))
            width = 0.2
            
            plt.figure(figsize=(15, 8))
            for i, metric in enumerate(self.metrics):
                values = [results[model][metric] for model in model_names]
                plt.bar(x + i*width, values, width, label=metric)
            
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x + width*1.5, model_names, rotation=45)
            plt.legend()
            
            plt.tight_layout()
            plot_path = os.path.join(self.evaluation_dir, 'model_comparison.png')
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info("Model comparison plot saved")
        except Exception as e:
            self.logger.error(f"Error plotting model comparison: {str(e)}")
            raise ModelEvaluationError(f"Failed to plot model comparison: {str(e)}")

    def save_evaluation_results(self) -> None:
        """
        Save evaluation results to JSON file
        """
        try:
            # Convert metrics to serializable format
            results = {
                metric: value for metric, value in self.metrics.items()
                if metric != 'classification_report'
            }
            
            # Save results
            results_path = os.path.join(self.evaluation_dir, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            self.logger.info(f"Evaluation results saved to {results_path}")
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {str(e)}")
            raise ModelEvaluationError(f"Failed to save evaluation results: {str(e)}")

def main():
    """Main function to test the ModelEvaluator class"""
    try:
        # Example configuration
        config = {
            'paths': {
                'evaluation_dir': 'models/evaluation',
                'saved_models_dir': 'models/saved_models'
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1'],
                'plot_confusion_matrix': True,
                'plot_roc_curve': True
            }
        }
        
        # Initialize evaluator
        evaluator = ModelEvaluator(config)
        
        # Load test data
        test_file = 'data/processed/test_data.csv'
        test_df = evaluator.load_data(test_file)
        X_test = test_df.drop(columns=['qualified_status'])
        y_test = test_df['qualified_status']
        
        # Load model and encoders
        model_path = os.path.join(config['paths']['saved_models_dir'], 'random_forest_model.pkl')
        encoders_path = os.path.join(config['paths']['saved_models_dir'], 'random_forest_encoders.pkl')
        
        model = evaluator.load_model(model_path)
        encoders = evaluator.load_encoders(encoders_path)
        
        # Preprocess test data
        X_test_processed = evaluator.preprocess_data(X_test, encoders)
        
        # Evaluate model
        metrics = evaluator.evaluate_model(model, X_test_processed, y_test)
        
        # Save evaluation results
        evaluator.save_evaluation_results()
        
        print("Model evaluation completed successfully")
        print(f"Metrics: {metrics}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 