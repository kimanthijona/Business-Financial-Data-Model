import os
import sys
from src.data.data_ingestion import DataIngestion

def main():
    config = {
        'data': {
            'raw_data_path': 'data/raw/survey_data.csv',
            'processed_data_path': 'data/processed',
            'features_path': 'data/features/features.csv',
            'target_column': 'qualified_status'
        },
        'training': {
            'test_size': 0.2,
            'random_state': 42
        }
    }
    
    try:
        data_ingestion = DataIngestion(config)
        df = data_ingestion.load_data()
        data_ingestion.save_data(df)
        
        # Verify the saved files
        print("\nVerifying saved files:")
        train_file = os.path.join(config['data']['processed_data_path'], 'train_data.csv')
        test_file = os.path.join(config['data']['processed_data_path'], 'test_data.csv')
        
        if os.path.exists(train_file):
            print(f"Training data saved successfully at {train_file}")
        if os.path.exists(test_file):
            print(f"Test data saved successfully at {test_file}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 