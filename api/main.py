"""
FastAPI Credit Scoring API
Main application file
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    ALLOWED_ORIGINS,
    LOG_LEVEL
)
from api.models import (
    ClientFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureImportanceResponse,
    HealthResponse,
    ErrorResponse
)
from api.predictor import get_predictor

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    try:
        logger.info("Starting up API...")
        predictor = get_predictor()
        logger.info(f"Predictor loaded successfully. Threshold: {predictor.get_threshold()}")
    except Exception as e:
        logger.error(f"Failed to load predictor: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check if the API is healthy and the model is loaded
    
    Returns
    -------
    HealthResponse
        Status information about the API
    """
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="healthy",
            model_loaded=predictor.is_loaded(),
            version=API_VERSION
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict credit score for a client",
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict(client: ClientFeatures):
    """
    Predict credit score and decision for a client
    
    Parameters
    ----------
    client : ClientFeatures
        Client features for prediction
        
    Returns
    -------
    PredictionResponse
        Prediction results including probability and decision
        
    Raises
    ------
    HTTPException
        If prediction fails
    """
    try:
        logger.info(f"Prediction request for client: {client.client_id}")
        
        predictor = get_predictor()
        
        # Get probabilities
        proba_no_default, proba_default = predictor.predict_proba(client.features)
        
        # Get prediction and decision
        prediction, decision = predictor.predict(client.features)
        
        response = PredictionResponse(
            client_id=client.client_id,
            probability_default=proba_default,
            probability_no_default=proba_no_default,
            prediction=prediction,
            decision=decision,
            threshold_used=predictor.get_threshold()
        )
        
        logger.info(f"Prediction completed: {decision} (proba: {proba_default:.4f})")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict credit scores for multiple clients",
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict credit scores for multiple clients
    
    Parameters
    ----------
    request : BatchPredictionRequest
        List of clients to predict
        
    Returns
    -------
    BatchPredictionResponse
        Batch prediction results with statistics
        
    Raises
    ------
    HTTPException
        If batch prediction fails
    """
    try:
        logger.info(f"Batch prediction request for {len(request.clients)} clients")
        
        predictor = get_predictor()
        predictions = []
        approved_count = 0
        rejected_count = 0
        
        for client in request.clients:
            # Get probabilities
            proba_no_default, proba_default = predictor.predict_proba(client.features)
            
            # Get prediction and decision
            prediction, decision = predictor.predict(client.features)
            
            predictions.append(PredictionResponse(
                client_id=client.client_id,
                probability_default=proba_default,
                probability_no_default=proba_no_default,
                prediction=prediction,
                decision=decision,
                threshold_used=predictor.get_threshold()
            ))
            
            if decision == "APPROVED":
                approved_count += 1
            else:
                rejected_count += 1
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_clients=len(predictions),
            approved_count=approved_count,
            rejected_count=rejected_count
        )
        
        logger.info(f"Batch prediction completed: {approved_count} approved, {rejected_count} rejected")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch prediction"
        )


@app.post(
    "/feature-importance",
    response_model=FeatureImportanceResponse,
    tags=["Explainability"],
    summary="Get SHAP feature importance for a client",
    responses={
        200: {"description": "Successful feature importance calculation"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def feature_importance(client: ClientFeatures):
    """
    Get SHAP feature importance values for a client's prediction
    
    This endpoint provides explainability by showing which features
    contributed most to the prediction decision.
    
    Parameters
    ----------
    client : ClientFeatures
        Client features for analysis
        
    Returns
    -------
    FeatureImportanceResponse
        SHAP values and top contributing features
        
    Raises
    ------
    HTTPException
        If feature importance calculation fails
    """
    try:
        logger.info(f"Feature importance request for client: {client.client_id}")
        
        predictor = get_predictor()
        importance = predictor.get_feature_importance(client.features)
        
        response = FeatureImportanceResponse(
            client_id=client.client_id,
            shap_values=importance["shap_values"],
            top_positive_features=importance["top_positive_features"],
            top_negative_features=importance["top_negative_features"],
            base_value=importance["base_value"],
            prediction_value=importance["prediction_value"]
        )
        
        logger.info(f"Feature importance calculated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during feature importance calculation"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )