from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from typing import List
from predict import predict_data, predict_proba_data, get_model_info

# API Version
API_VERSION = "v1"

app = FastAPI(
    title="Iris Classification API",
    description="Enhanced FastAPI service for Iris flower classification with batch prediction and probability support",
    version=API_VERSION
)

# Iris class names mapping
IRIS_CLASSES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Request Models
class IrisData(BaseModel):
    petal_length: float = Field(..., description="Petal length in cm", gt=0)
    sepal_length: float = Field(..., description="Sepal length in cm", gt=0)
    petal_width: float = Field(..., description="Petal width in cm", gt=0)
    sepal_width: float = Field(..., description="Sepal width in cm", gt=0)

class BatchIrisData(BaseModel):
    samples: List[IrisData] = Field(..., description="List of iris samples to predict", min_items=1, max_items=100)

# Response Models
class IrisResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class (0: setosa, 1: versicolor, 2: virginica)")
    class_name: str = Field(..., description="Predicted class name")

class IrisProbabilityResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class (0: setosa, 1: versicolor, 2: virginica)")
    class_name: str = Field(..., description="Predicted class name")
    probabilities: dict = Field(..., description="Probability for each class")
    confidence: float = Field(..., description="Confidence score of the prediction")

class BatchIrisResponse(BaseModel):
    predictions: List[IrisResponse] = Field(..., description="List of predictions for each sample")
    total_samples: int = Field(..., description="Total number of samples processed")

class ModelInfoResponse(BaseModel):
    model_type: str
    model_path: str
    classes: dict
    feature_names: List[str]
    api_version: str

# Endpoints
@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Health check endpoint"""
    return {"status": "healthy", "api_version": API_VERSION}

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Detailed health check endpoint"""
    try:
        model_info = get_model_info()
        return {
            "status": "healthy",
            "api_version": API_VERSION,
            "model_loaded": True,
            "model_type": model_info.get("model_type", "Unknown")
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info_endpoint():
    """Get information about the loaded model"""
    try:
        info = get_model_info()
        return ModelInfoResponse(
            model_type=info.get("model_type", "DecisionTreeClassifier"),
            model_path=info.get("model_path", "../model/iris_model.pkl"),
            classes=IRIS_CLASSES,
            feature_names=info.get("feature_names", ["sepal_length", "sepal_width", "petal_length", "petal_width"]),
            api_version=API_VERSION
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    """
    Predict the iris class for a single sample.
    
    Returns the predicted class number and class name.
    """
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]

        prediction = predict_data(features)
        predicted_class = int(prediction[0])
        
        return IrisResponse(
            prediction=predicted_class,
            class_name=IRIS_CLASSES[predicted_class]
        )
    
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid prediction class: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/probabilities", response_model=IrisProbabilityResponse)
async def predict_iris_with_probabilities(iris_features: IrisData):
    """
    Predict the iris class with probability scores for a single sample.
    
    Returns the predicted class, class name, probabilities for all classes, and confidence score.
    """
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]

        prediction = predict_data(features)
        probabilities = predict_proba_data(features)
        
        predicted_class = int(prediction[0])
        prob_dict = {
            IRIS_CLASSES[i]: float(prob) 
            for i, prob in enumerate(probabilities[0])
        }
        confidence = float(probabilities[0][predicted_class])
        
        return IrisProbabilityResponse(
            prediction=predicted_class,
            class_name=IRIS_CLASSES[predicted_class],
            probabilities=prob_dict,
            confidence=confidence
        )
    
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid prediction class: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchIrisResponse)
async def predict_iris_batch(batch_data: BatchIrisData):
    """
    Predict iris classes for multiple samples in a single request.
    
    Accepts up to 100 samples at once for efficient batch processing.
    """
    try:
        if len(batch_data.samples) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one sample is required"
            )
        
        if len(batch_data.samples) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 samples allowed per batch"
            )
        
        # Prepare features array
        features = []
        for sample in batch_data.samples:
            features.append([
                sample.sepal_length,
                sample.sepal_width,
                sample.petal_length,
                sample.petal_width
            ])
        
        # Get predictions
        predictions = predict_data(features)
        
        # Format responses
        prediction_responses = []
        for pred in predictions:
            predicted_class = int(pred)
            prediction_responses.append(
                IrisResponse(
                    prediction=predicted_class,
                    class_name=IRIS_CLASSES[predicted_class]
                )
            )
        
        return BatchIrisResponse(
            predictions=prediction_responses,
            total_samples=len(prediction_responses)
        )
    
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid prediction class: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/version")
async def get_api_version():
    """Get the current API version"""
    return {"version": API_VERSION}
