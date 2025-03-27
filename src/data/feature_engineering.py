"""
Feature engineering module for creating additional features from survey data.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from src.utils.logger import setup_logger
from src.utils.exceptions import DataTransformationError

logger = setup_logger(__name__)

class FeatureEngineering:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureEngineering class
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = logger
        
    def calculate_age(self, df: pd.DataFrame, dob_col: str) -> pd.Series:
        """
        Calculate age from date of birth
        
        Args:
            df (pd.DataFrame): Input dataframe
            dob_col (str): Name of date of birth column
            
        Returns:
            pd.Series: Age in years
        """
        try:
            current_year = datetime.now().year
            df[dob_col] = pd.to_datetime(df[dob_col])
            age = current_year - df[dob_col].dt.year
            return age
        except Exception as e:
            self.logger.error(f"Error calculating age: {str(e)}")
            raise DataTransformationError(f"Failed to calculate age: {str(e)}")
    
    def calculate_business_operation_period(
        self,
        df: pd.DataFrame,
        start_date_col: str
    ) -> pd.Series:
        """
        Calculate business operation period in years
        
        Args:
            df (pd.DataFrame): Input dataframe
            start_date_col (str): Name of business start date column
            
        Returns:
            pd.Series: Business operation period in years
        """
        try:
            current_date = datetime.now()
            df[start_date_col] = pd.to_datetime(df[start_date_col])
            operation_period = (current_date - df[start_date_col]).dt.days / 365.25
            return operation_period
        except Exception as e:
            self.logger.error(f"Error calculating business operation period: {str(e)}")
            raise DataTransformationError(f"Failed to calculate business operation period: {str(e)}")
    
    def calculate_total_business_hours(
        self,
        df: pd.DataFrame,
        opening_time_col: str,
        closing_time_col: str,
        days_open_col: str
    ) -> pd.Series:
        """
        Calculate total business hours per week
        
        Args:
            df (pd.DataFrame): Input dataframe
            opening_time_col (str): Name of opening time column
            closing_time_col (str): Name of closing time column
            days_open_col (str): Name of days open column
            
        Returns:
            pd.Series: Total business hours per week
        """
        try:
            df[opening_time_col] = pd.to_datetime(df[opening_time_col]).dt.hour
            df[closing_time_col] = pd.to_datetime(df[closing_time_col]).dt.hour
            daily_hours = df[closing_time_col] - df[opening_time_col]
            total_hours = daily_hours * df[days_open_col]
            return total_hours
        except Exception as e:
            self.logger.error(f"Error calculating total business hours: {str(e)}")
            raise DataTransformationError(f"Failed to calculate total business hours: {str(e)}")
    
    def calculate_inventory_turnover(
        self,
        df: pd.DataFrame,
        sales_col: str,
        inventory_col: str
    ) -> pd.Series:
        """
        Calculate inventory turnover ratio
        
        Args:
            df (pd.DataFrame): Input dataframe
            sales_col (str): Name of sales column
            inventory_col (str): Name of inventory column
            
        Returns:
            pd.Series: Inventory turnover ratio
        """
        try:
            inventory_turnover = df[sales_col] / df[inventory_col]
            return inventory_turnover
        except Exception as e:
            self.logger.error(f"Error calculating inventory turnover: {str(e)}")
            raise DataTransformationError(f"Failed to calculate inventory turnover: {str(e)}")
    
    def calculate_customer_segments(
        self,
        df: pd.DataFrame,
        customer_type_col: str
    ) -> pd.DataFrame:
        """
        Calculate customer segment ratios
        
        Args:
            df (pd.DataFrame): Input dataframe
            customer_type_col (str): Name of customer type column
            
        Returns:
            pd.DataFrame: Customer segment ratios
        """
        try:
            segment_counts = pd.get_dummies(df[customer_type_col], prefix='customer_segment')
            segment_ratios = segment_counts.mean()
            return pd.DataFrame(segment_ratios).T
        except Exception as e:
            self.logger.error(f"Error calculating customer segments: {str(e)}")
            raise DataTransformationError(f"Failed to calculate customer segments: {str(e)}")
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable based on daily sales threshold
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with target variable
        """
        try:
            daily_sales_col = self.config['feature_engineering']['daily_sales_col']
            target_threshold = self.config['feature_engineering']['target_threshold']
            target_col = self.config['data']['target_column']
            
            df[target_col] = (df[daily_sales_col] > target_threshold).astype(int)
            self.logger.info(f"Created target variable '{target_col}' based on daily sales threshold {target_threshold}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating target variable: {str(e)}")
            raise DataTransformationError(f"Failed to create target variable: {str(e)}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        try:
            # Get feature engineering settings from config
            fe_config = self.config['feature_engineering']
            
            # Calculate age
            if 'dob_col' in fe_config:
                df['age'] = self.calculate_age(df, fe_config['dob_col'])
            
            # Calculate business operation period
            if 'start_date_col' in fe_config:
                df['business_operation_years'] = self.calculate_business_operation_period(
                    df,
                    fe_config['start_date_col']
                )
            
            # Calculate total business hours
            if all(col in fe_config for col in ['opening_time_col', 'closing_time_col', 'days_open_col']):
                df['total_business_hours'] = self.calculate_total_business_hours(
                    df,
                    fe_config['opening_time_col'],
                    fe_config['closing_time_col'],
                    fe_config['days_open_col']
                )
            
            # Calculate inventory turnover
            if all(col in fe_config for col in ['sales_col', 'inventory_col']):
                df['inventory_turnover'] = self.calculate_inventory_turnover(
                    df,
                    fe_config['sales_col'],
                    fe_config['inventory_col']
                )
            
            # Calculate customer segments
            if 'customer_type_col' in fe_config:
                customer_segments = self.calculate_customer_segments(
                    df,
                    fe_config['customer_type_col']
                )
                df = pd.concat([df, customer_segments], axis=1)
            
            # Create target variable
            df = self.create_target_variable(df)
            
            self.logger.info("Feature engineering completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise DataTransformationError(f"Failed to engineer features: {str(e)}")

def main():
    """Main function to test the FeatureEngineering class"""
    try:
        # Example configuration
        config = {
            'data': {
                'target_column': 'high_value_customer'
            },
            'feature_engineering': {
                'daily_sales_col': 'daily_sales',
                'target_threshold': 100000,
                'dob_col': 'date_of_birth',
                'start_date_col': 'business_start_date',
                'opening_time_col': 'opening_time',
                'closing_time_col': 'closing_time',
                'days_open_col': 'days_open',
                'sales_col': 'monthly_sales',
                'inventory_col': 'inventory_value',
                'customer_type_col': 'customer_segments'
            }
        }
        
        # Create sample data
        data = {
            'daily_sales': [80000, 120000, 90000, 150000],
            'date_of_birth': ['1980-01-01', '1985-02-15', '1990-03-30', '1975-12-25'],
            'business_start_date': ['2015-01-01', '2018-06-15', '2010-03-30', '2020-01-01'],
            'opening_time': ['08:00:00', '09:00:00', '07:30:00', '10:00:00'],
            'closing_time': ['17:00:00', '18:00:00', '16:30:00', '19:00:00'],
            'days_open': [6, 5, 7, 6],
            'monthly_sales': [2400000, 3600000, 2700000, 4500000],
            'inventory_value': [800000, 1200000, 900000, 1500000],
            'customer_segments': ['retail', 'wholesale', 'retail', 'service']
        }
        df = pd.DataFrame(data)
        
        # Initialize and run feature engineering
        fe = FeatureEngineering(config)
        df_engineered = fe.engineer_features(df)
        print("\nEngineered features:")
        print(df_engineered.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 