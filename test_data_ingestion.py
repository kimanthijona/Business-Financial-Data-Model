from src.data.data_ingestion import DataIngestion
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Test the data ingestion process"""
    try:
        # Example configuration
        config = {
            'data': {
                'raw_data_path': 'data/raw/survey_data.csv',
                'processed_data_path': 'data/processed/preprocessed_survey_data.csv',
                'features_path': 'features.csv',
                'target_column': 'qualified_status'
            }
        }
        
        # Initialize and run data ingestion
        data_ingestion = DataIngestion(config)
        df = data_ingestion.load_data()
        print("\nSample of processed data:")
        print(df[['daily_sales', 'qualified_status']].head())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 