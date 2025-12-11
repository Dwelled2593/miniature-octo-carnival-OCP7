"""
Pydantic models for request/response validation
"""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ClientFeatures(BaseModel):
    """
    Input features for credit scoring prediction.
    All features from the trained model.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and their values",
        json_schema_extra={
            "example": {
                "EXT_SOURCE_2": 0.5,
                "EXT_SOURCE_3": 0.6,
                "DAYS_BIRTH": -15000,
                "AMT_CREDIT": 500000
            }
        }
    )
    client_id: Optional[str] = Field(
        None,
        description="Optional client identifier for tracking"
    )

    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """
    Response model for credit scoring prediction
    """
    client_id: Optional[str] = Field(
        None,
        description="Client identifier if provided"
    )
    probability_default: float = Field(
        ...,
        description="Probability of default (0-1)",
        ge=0.0,
        le=1.0
    )
    probability_no_default: float = Field(
        ...,
        description="Probability of no default (0-1)",
        ge=0.0,
        le=1.0
    )
    prediction: int = Field(
        ...,
        description="Binary prediction: 0 (no default) or 1 (default)",
        ge=0,
        le=1
    )
    decision: str = Field(
        ...,
        description="Business decision: APPROVED or REJECTED"
    )
    threshold_used: float = Field(
        ...,
        description="Decision threshold used for classification"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "client_id": "12345",
                "probability_default": 0.23,
                "probability_no_default": 0.77,
                "prediction": 0,
                "decision": "APPROVED",
                "threshold_used": 0.48
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch predictions
    """
    clients: List[ClientFeatures] = Field(
        ...,
        description="List of clients to predict"
    )


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions
    """
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each client"
    )
    total_clients: int = Field(
        ...,
        description="Total number of clients processed"
    )
    approved_count: int = Field(
        ...,
        description="Number of approved credits"
    )
    rejected_count: int = Field(
        ...,
        description="Number of rejected credits"
    )


class FeatureImportanceResponse(BaseModel):
    """
    Response model for SHAP feature importance
    """
    client_id: Optional[str] = Field(
        None,
        description="Client identifier if provided"
    )
    shap_values: Dict[str, float] = Field(
        ...,
        description="SHAP values for each feature"
    )
    top_positive_features: List[Dict[str, Any]] = Field(
        ...,
        description="Top features contributing positively (towards approval)"
    )
    top_negative_features: List[Dict[str, Any]] = Field(
        ...,
        description="Top features contributing negatively (towards rejection)"
    )
    base_value: float = Field(
        ...,
        description="Base prediction value (expected value)"
    )
    prediction_value: float = Field(
        ...,
        description="Final prediction value after SHAP contributions"
    )


class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str = Field(
        ...,
        description="API status"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }
    )


class ErrorResponse(BaseModel):
    """
    Error response model
    """
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )