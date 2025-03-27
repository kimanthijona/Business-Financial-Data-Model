import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List
import json
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.data.data_ingestion import DataIngestion
from src.data.data_validation import DataValidation
from src.data.data_transformation import DataTransformation
from src.data.feature_engineering import FeatureEngineering
from src.data.automated_feature_engineering import AutomatedFeatureEngineering
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluation import ModelEvaluator
from src.models.model_prediction import ModelPredictor
from src.utils.logger import setup_logger
from src.utils.exceptions import DataIngestionError, DataValidationError, DataTransformationError, ModelTrainingError, ModelEvaluationError

# Setup logging
logger = setup_logger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration"""
    log_dir = config['paths']['logs_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'pipeline_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def combine_features(
    survey_features: pd.DataFrame,
    automated_features: pd.DataFrame,
    target: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Combine survey features with automated features
    
    Args:
        survey_features (pd.DataFrame): Features from survey data transformation
        automated_features (pd.DataFrame): Features from automated feature engineering
        target (pd.Series): Target variable
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Combined features and target
    """
    # Remove any duplicate columns
    common_cols = set(survey_features.columns) & set(automated_features.columns)
    automated_features = automated_features.drop(columns=list(common_cols))
    
    # Combine features
    combined_features = pd.concat([survey_features, automated_features], axis=1)
    
    # Remove any columns that might be correlated with target
    target_corr = combined_features.corrwith(target).abs()
    high_corr_cols = target_corr[target_corr > 0.95].index
    combined_features = combined_features.drop(columns=high_corr_cols)
    
    return combined_features, target

def explain_predictions(
    model,
    X: pd.DataFrame,
    feature_names: List[str],
    plots_dir: str,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Generate SHAP and LIME explanations for model predictions
    
    Args:
        model: Trained model
        X (pd.DataFrame): Feature matrix
        feature_names (List[str]): List of feature names
        plots_dir (str): Directory to save plots
        sample_size (int): Number of samples to use for explanations
        
    Returns:
        Dict[str, Any]: Dictionary containing explanation results
    """
    logger.info("Generating model explanations")
    explanation_results = {}
    
    try:
        # Prepare data for explanations
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
            
        # SHAP Explanations
        logger.info("Generating SHAP explanations")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take positive class
            
        # SHAP Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'shap_summary.png'))
        plt.close()
        
        # SHAP Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'shap_importance.png'))
        plt.close()
        
        # Store SHAP values
        explanation_results['shap'] = {
            'mean_abs_shap': np.abs(shap_values).mean(0).tolist(),
            'feature_names': feature_names
        }
        
        # LIME Explanations
        logger.info("Generating LIME explanations")
        
        # Create LIME explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_sample.values,
            feature_names=feature_names,
            class_names=['Not Qualified', 'Qualified'],
            mode='classification'
        )
        
        # Generate LIME explanation for a sample instance
        sample_instance = X_sample.iloc[0].values
        lime_exp = lime_explainer.explain_instance(
            sample_instance,
            model.predict_proba,
            num_features=len(feature_names)
        )
        
        # Save LIME explanation plot
        lime_exp.save_to_file(os.path.join(plots_dir, 'lime_explanation.html'))
        
        # Store LIME feature importance
        feature_importance = dict(lime_exp.as_list())
        explanation_results['lime'] = {
            'feature_importance': feature_importance,
            'explained_instance_index': 0
        }
        
        logger.info("Model explanations generated successfully")
        return explanation_results
        
    except Exception as e:
        logger.error(f"Error generating model explanations: {str(e)}")
        raise

def main():
    """Main function to run the ML pipeline"""
    try:
        logger.info("Starting ML pipeline")
        
        # Load configuration
        config_path = os.path.join(project_root, 'config.yaml')
        config = load_config(config_path)
        
        # Update paths in config
        config['paths'] = {
            'raw_data': os.path.join(project_root, 'data/raw/survey_data.csv'),
            'processed_data': os.path.join(project_root, 'data/processed/processed_data.csv'),
            'features_dir': os.path.join(project_root, 'data/features'),
            'models_dir': os.path.join(project_root, 'models'),
            'saved_models_dir': os.path.join(project_root, 'models/saved_models'),
            'evaluation_dir': os.path.join(project_root, 'models/evaluation'),
            'predictions_dir': os.path.join(project_root, 'models/predictions'),
            'logs_dir': os.path.join(project_root, 'logs')
        }
        
        # Create necessary directories
        for path in config['paths'].values():
            if path.endswith(('.csv', '.yaml')):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            else:
                os.makedirs(path, exist_ok=True)
        
        # Data Ingestion
        logger.info("Starting data ingestion")
        data_ingestion = DataIngestion(config)
        data = data_ingestion.load_data()
        logger.info(f"Data loaded successfully with shape: {data.shape}")
        
        # Data Validation
        logger.info("Starting data validation")
        data_validation = DataValidation(config)
        data_validation.validate_data(data)
        logger.info("Data validation completed successfully")
        
        # Data Transformation
        logger.info("Starting data transformation")
        data_transformation = DataTransformation(config)
        transformed_data = data_transformation.transform_data(data)
        logger.info("Data transformation completed successfully")
        
        # Feature Engineering
        logger.info("Starting feature engineering")
        
        # Initialize feature engineering components
        feature_engineering = FeatureEngineering(config)
        auto_feature_engineering = AutomatedFeatureEngineering(max_features=50)
        
        # Create target variable
        logger.info("Creating target variable")
        transformed_data = auto_feature_engineering.create_target_variable(
            transformed_data,
            config['feature_engineering']['daily_sales_col'],
            config['feature_engineering']['target_threshold']
        )
        
        # Split features and target
        target = transformed_data['qualified_status']
        features = transformed_data.drop(columns=['qualified_status'])
        
        # Generate automated features
        logger.info("Generating automated features")
        automated_features = auto_feature_engineering.generate_features(features)
        logger.info(f"Generated {len(auto_feature_engineering.get_feature_list())} automated features")
        
        # Combine original and automated features
        logger.info("Combining original and automated features")
        combined_features = pd.concat([features, automated_features], axis=1)
        logger.info(f"Total number of features: {len(combined_features.columns)}")
        
        # Plot correlation matrix
        plots_dir = os.path.join(config['paths']['features_dir'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        auto_feature_engineering.plot_correlation_matrix(
            combined_features,
            os.path.join(plots_dir, 'correlation_matrix.png')
        )
        
        # Save feature list
        with open(os.path.join(config['paths']['features_dir'], 'feature_list.txt'), 'w') as f:
            f.write('\n'.join(combined_features.columns.tolist()))
        logger.info(f"Feature list saved to {config['paths']['features_dir']}/feature_list.txt")
        
        # Model Training
        logger.info("Starting model training")
        model_trainer = ModelTrainer(config)
        model = model_trainer.train_model(combined_features, target)
        logger.info("Model training completed successfully")
        
        # Generate model explanations
        explanation_results = explain_predictions(
            model,
            combined_features,
            combined_features.columns.tolist(),
            plots_dir
        )
        
        # Save explanation results
        with open(os.path.join(config['paths']['evaluation_dir'], 'explanation_results.json'), 'w') as f:
            json.dump(explanation_results, f, indent=4)
        
        # Plot feature importance after model training
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importances = pd.Series(model.feature_importances_, index=combined_features.columns)
            importances = importances.sort_values(ascending=False)
            importances.plot(kind='bar')
            plt.title('Feature Importances')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
            plt.close()
            logger.info("Feature importance plot saved successfully")
        
        # Model Evaluation
        logger.info("Starting model evaluation")
        model_evaluator = ModelEvaluator(config)
        
        # Load the trained model and encoders
        model_path = os.path.join(config['paths']['saved_models_dir'], 'random_forest_model.pkl')
        encoders_path = os.path.join(config['paths']['saved_models_dir'], 'random_forest_encoders.pkl')
        model = model_evaluator.load_model(model_path)
        encoders = model_evaluator.load_encoders(encoders_path)
        
        # Preprocess the evaluation data
        X_processed = model_evaluator.preprocess_data(combined_features, encoders)
        
        # Make predictions
        y_pred = model.predict(X_processed)
        y_pred_proba = model.predict_proba(X_processed)[:, 1]
        
        # Calculate metrics
        metrics = model_evaluator.calculate_metrics(target, y_pred)
        
        # Plot evaluation visualizations
        model_evaluator.plot_confusion_matrix(target, y_pred, 'random_forest')
        model_evaluator.plot_roc_curve(target, y_pred_proba, 'random_forest')
        
        # Save evaluation results
        evaluation_results = {
            'metrics': metrics,
            'model_name': 'random_forest',
            'n_samples': len(target),
            'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None,
            'feature_names': combined_features.columns.tolist(),
            'automated_features': auto_feature_engineering.get_feature_list(),
            'explanations': explanation_results
        }
        
        with open(os.path.join(config['paths']['evaluation_dir'], 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=4)
            
        logger.info("Model evaluation completed successfully")
        
        # Model Prediction
        logger.info("Starting model prediction")
        model_predictor = ModelPredictor(config)
        
        # Create sample data for prediction
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
        predictions = model_predictor.predict(sample_data)
        probabilities = model_predictor.predict_proba(sample_data)
        
        # Save predictions
        predictions_dir = config['paths']['predictions_dir']
        os.makedirs(predictions_dir, exist_ok=True)
        predictions_df = pd.DataFrame({
            'predicted_class': predictions,
            'probability': probabilities[:, 1]
        })
        predictions_df.to_csv(os.path.join(predictions_dir, 'sample_predictions.csv'), index=False)
        
        logger.info("Model prediction completed successfully")
        
        logger.info("ML pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 