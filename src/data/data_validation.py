import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Union, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.exceptions import DataValidationError

logger = setup_logger(__name__)

class DataValidation:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataValidation class
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = logger
        
        # Define expected data types for each column
        self.numeric_columns = config['data_transformation']['numeric_columns']
        self.categorical_columns = config['data_transformation']['categorical_columns']
        self.expected_dtypes = {
            col: 'float64' for col in self.numeric_columns
        }
        self.expected_dtypes.update({
            col: 'object' for col in self.categorical_columns
        })

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the dataframe against expected schema
        
        Args:
            df (pd.DataFrame): Input dataframe to validate
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Check for required columns
            required_columns = self.numeric_columns + self.categorical_columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check data types and convert if necessary
            for col in self.numeric_columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception as e:
                    self.logger.error(f"Column {col} contains non-numeric values: {str(e)}")
                    return False
            
            # Check for null values
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                self.logger.warning(f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}")
            
            # Check for duplicate rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate rows")
            
            # Validate numeric ranges if specified in config
            if 'numeric_ranges' in self.config.get('data_validation', {}):
                for col, range_vals in self.config['data_validation']['numeric_ranges'].items():
                    if not self.validate_numeric_range(df, col, range_vals['min'], range_vals['max']):
                        return False
            
            # Validate categorical values if specified in config
            if 'categorical_values' in self.config.get('data_validation', {}):
                for col, allowed_values in self.config['data_validation']['categorical_values'].items():
                    if not self.validate_categorical_values(df, col, allowed_values):
                        return False
            
            self.logger.info("Data validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False

    def validate_numeric_range(self, df: pd.DataFrame, column: str, min_val: float, max_val: float) -> bool:
        """
        Validate that numeric values are within expected range
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to validate
            min_val (float): Minimum allowed value
            max_val (float): Maximum allowed value
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            if column not in df.columns:
                self.logger.error(f"Column {column} not found in dataframe")
                return False
            
            values = df[column]
            if not pd.api.types.is_numeric_dtype(values):
                self.logger.error(f"Column {column} is not numeric")
                return False
            
            out_of_range = values[(values < min_val) | (values > max_val)]
            if not out_of_range.empty:
                self.logger.error(
                    f"Column {column} contains values outside range [{min_val}, {max_val}]: {out_of_range.tolist()}"
                )
                return False
            
            self.logger.info(f"Range validation completed for column {column}")
            return True
            
        except Exception as e:
            self.logger.error(f"Range validation failed for column {column}: {str(e)}")
            return False

    def validate_categorical_values(self, df: pd.DataFrame, column: str, allowed_values: List[str]) -> bool:
        """
        Validate that categorical values are from allowed set
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to validate
            allowed_values (List[str]): List of allowed values
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            if column not in df.columns:
                self.logger.error(f"Column {column} not found in dataframe")
                return False
            
            values = df[column]
            invalid_values = [val for val in values.unique() if val not in allowed_values]
            if invalid_values:
                self.logger.error(
                    f"Column {column} contains invalid values: {invalid_values}. Allowed values: {allowed_values}"
                )
                return False
            
            self.logger.info(f"Categorical validation completed for column {column}")
            return True
            
        except Exception as e:
            self.logger.error(f"Categorical validation failed for column {column}: {str(e)}")
            return False

def main():
    """Main function to test the DataValidation class"""
    try:
        # Example configuration
        config = {
            'data_transformation': {
                'numeric_columns': ['age', 'income'],
                'categorical_columns': ['gender', 'education']
            },
            'data_validation': {
                'numeric_ranges': {
                    'age': {'min': 18, 'max': 100},
                    'income': {'min': 0, 'max': 1000000}
                },
                'categorical_values': {
                    'gender': ['M', 'F'],
                    'education': ['BSc', 'MSc', 'PhD']
                }
            }
        }
        
        # Create sample data
        data = {
            'age': [25, 30, 35, 40],
            'income': [50000.0, 60000.0, 75000.0, 80000.0],
            'gender': ['M', 'F', 'M', 'F'],
            'education': ['BSc', 'MSc', 'PhD', 'BSc']
        }
        df = pd.DataFrame(data)
        
        # Initialize and run validation
        validator = DataValidation(config)
        if validator.validate_data(df):
            print("Data validation completed successfully")
        else:
            print("Data validation failed")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 