import pandas as pd
import numpy as np
import featuretools as ft
from featuretools.primitives import (
    Mean, Sum, Std, Max, Min, Count, PercentTrue, NumUnique, Mode,
    Year, Month, Day, Hour, MultiplyNumeric, SubtractNumeric, AddNumeric,
    GreaterThan, LessThan
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from typing import List, Dict, Tuple
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.exceptions import DataTransformationError

logger = setup_logger(__name__)

class AutomatedFeatureEngineering:
    def __init__(self, max_features: int = 50):
        """
        Initialize AutomatedFeatureEngineering class
        
        Args:
            max_features (int): Maximum number of features to generate
        """
        self.logger = logger
        self.feature_list = []
        self.max_features = max_features
        self.feature_importances = None

    def create_entityset(self, df: pd.DataFrame) -> ft.EntitySet:
        """
        Create an EntitySet for featuretools
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            ft.EntitySet: EntitySet for feature generation
        """
        try:
            es = ft.EntitySet(id="survey_data")
            es.add_dataframe(
                dataframe_name="survey",
                dataframe=df,
                index="index",
                make_index=True
            )
            return es
        except Exception as e:
            self.logger.error(f"Error creating EntitySet: {str(e)}")
            raise DataTransformationError(f"Failed to create EntitySet: {str(e)}")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features automatically using featuretools
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with generated features
        """
        try:
            # Create EntitySet
            es = self.create_entityset(df)
            
            # Define aggregation primitives
            agg_primitives = [
                Mean,
                Sum,
                Std,
                Max,
                Min,
                Count,
                PercentTrue,
                NumUnique,
                Mode
            ]
            
            # Define transformation primitives
            trans_primitives = [
                Year,
                Month,
                Day,
                Hour,
                MultiplyNumeric,
                SubtractNumeric,
                AddNumeric,
                GreaterThan,
                LessThan
            ]
            
            # Generate features
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="survey",
                agg_primitives=agg_primitives,
                trans_primitives=trans_primitives,
                max_features=self.max_features
            )
            
            self.feature_list = feature_matrix.columns.tolist()
            self.logger.info(f"Generated {len(self.feature_list)} features automatically")
            
            return feature_matrix
        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            raise DataTransformationError(f"Failed to generate features: {str(e)}")

    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select most important features using mutual information
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            n_features (int): Number of features to select
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Selected features and their names
        """
        try:
            # Select features using mutual information
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Store feature importances
            self.feature_importances = dict(zip(selected_features, selector.scores_[selector.get_support()]))
            
            return pd.DataFrame(X_selected, columns=selected_features), selected_features
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            raise DataTransformationError(f"Failed to select features: {str(e)}")

    def plot_feature_importance(self, output_path: str) -> None:
        """
        Plot feature importance
        
        Args:
            output_path (str): Path to save the plot
        """
        try:
            if self.feature_importances is None:
                raise ValueError("Feature importances not calculated yet")
            
            # Sort features by importance
            sorted_features = dict(sorted(
                self.feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(sorted_features)), list(sorted_features.values()))
            plt.xticks(range(len(sorted_features)), list(sorted_features.keys()), rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
            self.logger.info(f"Feature importance plot saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")
            raise DataTransformationError(f"Failed to plot feature importance: {str(e)}")

    def plot_correlation_matrix(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Plot correlation matrix of features
        
        Args:
            df (pd.DataFrame): Feature matrix
            output_path (str): Path to save the plot
        """
        try:
            # Calculate correlation matrix
            corr = df.corr()
            
            # Create plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
            self.logger.info(f"Correlation matrix plot saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error plotting correlation matrix: {str(e)}")
            raise DataTransformationError(f"Failed to plot correlation matrix: {str(e)}")

    def plot_evaluation_metrics(self, metrics: Dict[str, float], output_path: str) -> None:
        """
        Plot evaluation metrics
        
        Args:
            metrics (Dict[str, float]): Dictionary of evaluation metrics
            output_path (str): Path to save the plot
        """
        try:
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.bar(metrics.keys(), metrics.values())
            plt.ylim(0, 1)
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title('Model Evaluation Metrics')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
            self.logger.info(f"Evaluation metrics plot saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error plotting evaluation metrics: {str(e)}")
            raise DataTransformationError(f"Failed to plot evaluation metrics: {str(e)}")

    def create_target_variable(self, df: pd.DataFrame, daily_sales_col: str, threshold: float = 5000) -> pd.DataFrame:
        """
        Create binary target variable based on daily sales
        
        Args:
            df (pd.DataFrame): Input dataframe
            daily_sales_col (str): Name of the daily sales column
            threshold (float): Threshold for classification
            
        Returns:
            pd.DataFrame: Dataframe with target variable
        """
        try:
            df['target'] = (df[daily_sales_col] >= threshold).astype(int)
            self.logger.info("Target variable created successfully")
            return df
        except Exception as e:
            self.logger.error(f"Error creating target variable: {str(e)}")
            raise DataTransformationError(f"Failed to create target variable: {str(e)}")

    def get_feature_list(self) -> List[str]:
        """
        Get list of generated features
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_list

def main():
    """Main function to test the AutomatedFeatureEngineering class"""
    try:
        # Example usage
        data = {
            "age": [25, 30, 35, 40],
            "income": [50000, 60000, 75000, 80000],
            "daily_sales": [4500, 5500, 6000, 7000],
            "gender": ["M", "F", "M", "F"],
            "education": ["BSc", "MSc", "PhD", "BSc"]
        }
        df = pd.DataFrame(data)
        
        # Initialize and run feature engineering
        auto_fe = AutomatedFeatureEngineering(max_features=50)
        df = auto_fe.create_target_variable(df, "daily_sales", threshold=5000)
        feature_matrix = auto_fe.generate_features(df)
        
        # Select best features
        X = feature_matrix.drop('target', axis=1)
        y = feature_matrix['target']
        X_selected, selected_features = auto_fe.select_features(X, y, n_features=20)
        
        # Create visualizations
        os.makedirs("visualizations", exist_ok=True)
        auto_fe.plot_feature_importance("visualizations/feature_importance.png")
        auto_fe.plot_correlation_matrix(X_selected, "visualizations/correlation_matrix.png")
        
        print("Feature engineering completed successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 