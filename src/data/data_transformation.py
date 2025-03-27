import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Union, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.exceptions import DataTransformationError

logger = setup_logger(__name__)

class DataTransformation:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataTransformation class
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.numeric_columns = config['data_transformation']['numeric_columns']
        self.categorical_columns = config['data_transformation']['categorical_columns']
        self.logger = logger
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer = SimpleImputer(strategy='mean')

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        try:
            # Handle numeric columns
            df[self.numeric_columns] = self.imputer.fit_transform(df[self.numeric_columns])
            
            # Handle categorical columns
            for col in self.categorical_columns:
                df[col] = df[col].fillna('missing')
            
            self.logger.info("Missing values handled successfully")
            return df
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise DataTransformationError(f"Failed to handle missing values: {str(e)}")

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        try:
            df_encoded = df.copy()
            
            for col in self.categorical_columns:
                if col in df.columns:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            
            self.logger.info("Categorical features encoded successfully")
            return df_encoded
        except Exception as e:
            self.logger.error(f"Error encoding categorical features: {str(e)}")
            raise DataTransformationError(f"Failed to encode categorical features: {str(e)}")

    def scale_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with scaled numeric features
        """
        try:
            df_scaled = df.copy()
            
            if self.numeric_columns:
                df_scaled[self.numeric_columns] = self.scaler.fit_transform(df[self.numeric_columns])
            
            self.logger.info("Numeric features scaled successfully")
            return df_scaled
        except Exception as e:
            self.logger.error(f"Error scaling numeric features: {str(e)}")
            raise DataTransformationError(f"Failed to scale numeric features: {str(e)}")

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by scaling numeric features and encoding categorical features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        try:
            # Create a copy of the dataframe to avoid modifying the original
            df_transformed = df.copy()
            
            # Handle missing values first
            df_transformed = self.handle_missing_values(df_transformed)
            
            # Scale numeric features
            if self.numeric_columns:
                df_transformed[self.numeric_columns] = self.scaler.fit_transform(df_transformed[self.numeric_columns])
                self.logger.info(f"Scaled {len(self.numeric_columns)} numeric features")
            
            # Encode categorical features
            for col in self.categorical_columns:
                if col in df.columns:
                    self.label_encoders[col] = LabelEncoder()
                    df_transformed[col] = self.label_encoders[col].fit_transform(df_transformed[col].astype(str))
                    self.logger.info(f"Encoded categorical feature: {col}")
            
            return df_transformed
        except Exception as e:
            self.logger.error(f"Error transforming data: {str(e)}")
            raise DataTransformationError(f"Failed to transform data: {str(e)}")

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data back to original scale
        
        Args:
            df (pd.DataFrame): Transformed dataframe
            
        Returns:
            pd.DataFrame: Dataframe in original scale
        """
        try:
            df_original = df.copy()
            
            # Inverse scale numeric features
            if self.numeric_columns:
                df_original[self.numeric_columns] = self.scaler.inverse_transform(df[self.numeric_columns])
            
            # Inverse encode categorical features
            for col in self.categorical_columns:
                if col in df.columns and col in self.label_encoders:
                    df_original[col] = self.label_encoders[col].inverse_transform(df[col])
            
            return df_original
        except Exception as e:
            self.logger.error(f"Error inverse transforming data: {str(e)}")
            raise DataTransformationError(f"Failed to inverse transform data: {str(e)}")

    def save_transformers(self, output_dir: str) -> None:
        """
        Save fitted transformers to disk
        
        Args:
            output_dir (str): Directory to save transformers
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save scaler
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            pd.to_pickle(self.scaler, scaler_path)
            
            # Save label encoders
            encoders_path = os.path.join(output_dir, "label_encoders.pkl")
            pd.to_pickle(self.label_encoders, encoders_path)
            
            self.logger.info(f"Transformers saved to {output_dir}")
        except Exception as e:
            self.logger.error(f"Error saving transformers: {str(e)}")
            raise DataTransformationError(f"Failed to save transformers: {str(e)}")

    def load_transformers(self, input_dir: str) -> None:
        """
        Load fitted transformers from disk
        
        Args:
            input_dir (str): Directory containing saved transformers
        """
        try:
            # Load scaler
            scaler_path = os.path.join(input_dir, "scaler.pkl")
            self.scaler = pd.read_pickle(scaler_path)
            
            # Load label encoders
            encoders_path = os.path.join(input_dir, "label_encoders.pkl")
            self.label_encoders = pd.read_pickle(encoders_path)
            
            self.logger.info(f"Transformers loaded from {input_dir}")
        except Exception as e:
            self.logger.error(f"Error loading transformers: {str(e)}")
            raise DataTransformationError(f"Failed to load transformers: {str(e)}")

def main():
    """Main function to test the DataTransformation class"""
    try:
        # Example configuration
        config = {
            'data_transformation': {
                'numeric_columns': ["age", "income"],
                'categorical_columns': ["gender", "education"]
            }
        }
        
        # Create sample data
        data = {
            "age": [25, 30, 35, 40],
            "income": [50000, 60000, 75000, 80000],
            "gender": ["M", "F", "M", "F"],
            "education": ["BSc", "MSc", "PhD", "BSc"]
        }
        df = pd.DataFrame(data)
        
        # Initialize and run transformation
        transformer = DataTransformation(config)
        df_transformed = transformer.transform_data(df)
        print("Data transformed successfully")
        
        # Save transformers
        transformer.save_transformers("transformers")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 