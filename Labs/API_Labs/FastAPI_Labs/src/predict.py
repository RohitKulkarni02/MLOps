import joblib
import os
from typing import Dict, Any

# Cache the model to avoid reloading on every request
_model_cache = None
_model_path = "../model/iris_model.pkl"

def _load_model():
    """Load and cache the model"""
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(_model_path):
            raise FileNotFoundError(f"Model file not found at {_model_path}")
        _model_cache = joblib.load(_model_path)
    return _model_cache

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (list or numpy.ndarray): Input data for which predictions are to be made.
                                   Can be a single sample or multiple samples.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = _load_model()
    y_pred = model.predict(X)
    return y_pred

def predict_proba_data(X):
    """
    Predict the class probabilities for the input data.
    Args:
        X (list or numpy.ndarray): Input data for which probabilities are to be computed.
                                   Can be a single sample or multiple samples.
    Returns:
        y_proba (numpy.ndarray): Predicted class probabilities for each class.
    """
    model = _load_model()
    if not hasattr(model, 'predict_proba'):
        raise AttributeError("Model does not support probability predictions")
    y_proba = model.predict_proba(X)
    return y_proba

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    Returns:
        dict: Dictionary containing model information.
    """
    try:
        model = _load_model()
        model_type = type(model).__name__
        
        info = {
            "model_type": model_type,
            "model_path": _model_path,
            "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "n_classes": len(model.classes_) if hasattr(model, 'classes_') else None,
            "has_predict_proba": hasattr(model, 'predict_proba')
        }
        
        # Add model-specific attributes if available
        if hasattr(model, 'max_depth'):
            info["max_depth"] = model.max_depth
        if hasattr(model, 'random_state'):
            info["random_state"] = model.random_state
            
        return info
    except Exception as e:
        return {
            "error": str(e),
            "model_type": "Unknown",
            "model_path": _model_path
        }
