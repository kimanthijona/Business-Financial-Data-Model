import os
import sys
from src.models.model_evaluation import ModelEvaluator

def main():
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
    
    try:
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
        evaluator.evaluate_model(model, X_test_processed, y_test)
        
        # Save evaluation results
        evaluator.save_evaluation_results()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 