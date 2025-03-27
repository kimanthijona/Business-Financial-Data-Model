class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass

class DataTransformationError(Exception):
    """Raised when data transformation fails"""
    pass

class ModelTrainingError(Exception):
    """Raised when model training fails"""
    pass

class ModelEvaluationError(Exception):
    """Raised when model evaluation fails"""
    pass

class ModelPredictionError(Exception):
    """Raised when model prediction fails"""
    pass

class DataIngestionError(Exception):
    """Raised when data ingestion fails"""
    pass 