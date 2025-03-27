import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.exceptions import DataIngestionError

logger = setup_logger(__name__)

class DataIngestion:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataIngestion class
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.data_path = os.path.join(project_root, config['data']['raw_data_path'])
        self.features_path = os.path.join(project_root, config['data']['features_path'])
        self.processed_data_path = os.path.join(project_root, config['data']['processed_data_path'])
        self.logger = logger

    def create_qualified_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create qualified_status target variable based on daily sales threshold
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with qualified_status target variable
        """
        try:
            # Create qualified_status based on daily_sales threshold of 5000
            df['qualified_status'] = (df['daily_sales'] >= 5000).astype(int)
            
            # Print data for verification
            print("\nDaily sales and qualified status:")
            print(df[['daily_sales', 'qualified_status']].head(10))
            print("\nQualified status distribution:")
            print(df['qualified_status'].value_counts())
            
            # Log the distribution of qualified_status
            status_counts = df['qualified_status'].value_counts()
            self.logger.info(f"Qualified status distribution:\n{status_counts}")
            
            return df
        except Exception as e:
            self.logger.error(f"Error creating qualified_status: {str(e)}")
            raise DataIngestionError(f"Failed to create qualified_status: {str(e)}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Load data
            df = pd.read_csv(self.data_path)
            
            # Strip whitespace from string columns
            object_columns = df.select_dtypes(include=['object']).columns
            for col in object_columns:
                df[col] = df[col].str.strip()
            
            # Create qualified_status target variable
            df = self.create_qualified_status(df)
            
            # Log the data shape and target variable info
            self.logger.info(f"Data loaded successfully from {self.data_path}")
            self.logger.info(f"Data shape: {df.shape}")
            self.logger.info(f"Target variable info:\n{df['qualified_status'].describe()}")
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise DataIngestionError(f"Failed to load data: {str(e)}")

    def load_feature_mappings(self) -> dict:
        """
        Load feature mappings from CSV file
        
        Returns:
            dict: Dictionary of feature mappings
        """
        try:
            if not os.path.exists(self.features_path):
                self.logger.warning("Features file not found, skipping feature mappings")
                return {}
            
            mappings = pd.read_csv(self.features_path)
            # The first column contains the old names and the second column contains the new names
            feature_dict = dict(zip(mappings.iloc[:, 0], mappings.iloc[:, 1]))
            self.logger.info(f"Feature mappings loaded successfully from {self.features_path}")
            return feature_dict
        except Exception as e:
            self.logger.error(f"Error loading feature mappings: {str(e)}")
            raise DataIngestionError(f"Failed to load feature mappings: {str(e)}")

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns based on feature mappings
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with renamed columns
        """
        try:
            feature_dict = self.load_feature_mappings()
            if feature_dict:
                # Only rename columns that exist in the dataframe
                existing_columns = {col for col in feature_dict.keys() if col in df.columns}
                rename_dict = {col: feature_dict[col] for col in existing_columns}
                
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    self.logger.info(f"Renamed {len(rename_dict)} columns successfully")
                else:
                    self.logger.warning("No matching columns found to rename")
            return df
        except Exception as e:
            self.logger.error(f"Error renaming columns: {str(e)}")
            raise DataIngestionError(f"Failed to rename columns: {str(e)}")

    def drop_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with null values
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with null values dropped
        """
        try:
            initial_rows = len(df)
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            self.logger.info(f"Dropped {dropped_rows} rows with null values")
            return df
        except Exception as e:
            self.logger.error(f"Error dropping null values: {str(e)}")
            raise DataIngestionError(f"Failed to drop null values: {str(e)}")

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training features, test features, training target, test target
        """
        try:
            target_column = self.config['data']['target_column']
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Get test size from config or use default
            test_size = self.config.get('training', {}).get('test_size', 0.2)
            random_state = self.config.get('training', {}).get('random_state', 42)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            self.logger.info(f"Data split into training and test sets:")
            self.logger.info(f"Training features shape: {X_train.shape}")
            self.logger.info(f"Test features shape: {X_test.shape}")
            self.logger.info(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
            self.logger.info(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise DataIngestionError(f"Failed to split data: {str(e)}")

    def save_data(self, df: pd.DataFrame) -> None:
        """
        Save processed data to CSV
        
        Args:
            df (pd.DataFrame): Data to save
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
            
            # Save data
            processed_file = os.path.join(self.processed_data_path, 'preprocessed_survey_data.csv')
            df.to_csv(processed_file, index=False)
            self.logger.info(f"Data saved to {processed_file}")
            
            # Print verification of saved data
            print("\nVerification of saved data:")
            saved_df = pd.read_csv(processed_file)
            print("\nQualified status distribution in saved data:")
            print(saved_df['qualified_status'].value_counts())
            print("\nSample of saved data (first 5 rows):")
            print(saved_df[['daily_sales', 'qualified_status']].head())
            
            # Split and save training and test sets
            X_train, X_test, y_train, y_test = self.split_data(saved_df)
            
            # Save training data
            train_file = os.path.join(self.processed_data_path, 'train_data.csv')
            pd.concat([X_train, y_train], axis=1).to_csv(train_file, index=False)
            self.logger.info(f"Training data saved to {train_file}")
            
            # Save test data
            test_file = os.path.join(self.processed_data_path, 'test_data.csv')
            pd.concat([X_test, y_test], axis=1).to_csv(test_file, index=False)
            self.logger.info(f"Test data saved to {test_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise DataIngestionError(f"Failed to save data: {str(e)}")

def main():
    """Main function to test the DataIngestion class"""
    try:
        # Example configuration
        config = {
            'data': {
                'raw_data_path': 'data/raw/survey_data.csv',
                'processed_data_path': 'data/processed/preprocessed_survey_data.csv',
                'features_path': 'features.csv',
                'target_column': 'target'
            }
        }
        
        # Initialize and run data ingestion
        data_ingestion = DataIngestion(config)
        df = data_ingestion.load_data()
        df = data_ingestion.rename_columns(df)
        df = data_ingestion.drop_null_values(df)
        X_train, X_test, y_train, y_test = data_ingestion.split_data(df)
        print("Data loaded and processed successfully")
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")
        print(f"Training target distribution:")
        print(y_train.value_counts(normalize=True))
        print(f"Test target distribution:")
        print(y_test.value_counts(normalize=True))
        
        # Save processed data
        data_ingestion.save_data(df)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 