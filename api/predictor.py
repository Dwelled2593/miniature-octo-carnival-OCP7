"""
Credit Score Predictor
Handles model loading and predictions
"""

import pickle
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from api.config import (
    MODEL_PATH,
    EXPLAINER_PATH,
    FEATURE_NAMES_PATH,
    THRESHOLD_PATH,
    DEFAULT_THRESHOLD
)

logger = logging.getLogger(__name__)


class CreditScorePredictor:
    """
    Credit scoring predictor with SHAP explainability
    """
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.threshold = DEFAULT_THRESHOLD
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, explainer, feature names, and threshold"""
        try:
            # Load model
            logger.info(f"Loading model from {MODEL_PATH}")
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
            
            # Load explainer
            logger.info(f"Loading explainer from {EXPLAINER_PATH}")
            with open(EXPLAINER_PATH, 'rb') as f:
                self.explainer = pickle.load(f)
            logger.info("Explainer loaded successfully")
            
            # Load feature names
            logger.info(f"Loading feature names from {FEATURE_NAMES_PATH}")
            with open(FEATURE_NAMES_PATH, 'rb') as f:
                self.feature_names = pickle.load(f)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load optimal threshold
            if Path(THRESHOLD_PATH).exists():
                logger.info(f"Loading threshold from {THRESHOLD_PATH}")
                with open(THRESHOLD_PATH, 'r') as f:
                    threshold_data = json.load(f)
                    self.threshold = threshold_data.get('threshold', DEFAULT_THRESHOLD)
                logger.info(f"Using optimal threshold: {self.threshold}")
            else:
                logger.warning(f"Threshold file not found, using default: {DEFAULT_THRESHOLD}")
                
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise
    
    def _prepare_features(self, features_dict: Dict[str, float]) -> np.ndarray:
        """
        Prepare features in the correct order for the model
        
        Parameters
        ----------
        features_dict : Dict[str, float]
            Dictionary of feature names and values
            
        Returns
        -------
        np.ndarray
            Features array in correct order
        """
        # Create array with correct order
        features_array = np.zeros(len(self.feature_names))
        
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in features_dict:
                features_array[i] = features_dict[feature_name]
            else:
                # Missing feature - will be handled by model's imputer
                features_array[i] = np.nan
        
        return features_array.reshape(1, -1)
    
    def predict_proba(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict probability of default
        
        Parameters
        ----------
        features : Dict[str, float]
            Client features
            
        Returns
        -------
        Tuple[float, float]
            (probability_no_default, probability_default)
        """
        try:
            X = self._prepare_features(features)
            probas = self.model.predict_proba(X)[0]
            return float(probas[0]), float(probas[1])
        except Exception as e:
            logger.error(f"Error in predict_proba: {str(e)}")
            raise
    
    def predict(
        self,
        features: Dict[str, float],
        threshold: Optional[float] = None
    ) -> Tuple[int, str]:
        """
        Predict credit decision
        
        Parameters
        ----------
        features : Dict[str, float]
            Client features
        threshold : Optional[float]
            Custom threshold (uses optimal if None)
            
        Returns
        -------
        Tuple[int, str]
            (prediction, decision)
            prediction: 0 (no default) or 1 (default)
            decision: "APPROVED" or "REJECTED"
        """
        try:
            _, proba_default = self.predict_proba(features)
            
            # Use custom threshold or optimal threshold
            thresh = threshold if threshold is not None else self.threshold
            
            # Predict based on threshold
            prediction = 1 if proba_default > thresh else 0
            
            # Business decision (inverted: default=1 means REJECTED)
            decision = "REJECTED" if prediction == 1 else "APPROVED"
            
            return prediction, decision
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise
    
    def get_feature_importance(
        self,
        features: Dict[str, float],
        top_n: int = 10
    ) -> Dict:
        """
        Get SHAP feature importance for a prediction
        
        Parameters
        ----------
        features : Dict[str, float]
            Client features
        top_n : int
            Number of top features to return
            
        Returns
        -------
        Dict
            Dictionary with SHAP values and top features
        """
        try:
            X = self._prepare_features(features)
            
            # Get SHAP values
            shap_values = self.explainer(X)
            
            # Extract values
            shap_vals = shap_values.values[0]
            base_value = shap_values.base_values[0]
            
            # Create dictionary of feature: shap_value
            shap_dict = {
                feature: float(value)
                for feature, value in zip(self.feature_names, shap_vals)
            }
            
            # Sort by absolute value to get most important
            sorted_features = sorted(
                shap_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Separate positive and negative contributions
            positive_features = [
                {"feature": f, "value": v}
                for f, v in sorted_features if v > 0
            ][:top_n]
            
            negative_features = [
                {"feature": f, "value": v}
                for f, v in sorted_features if v < 0
            ][:top_n]
            
            # Calculate prediction value
            prediction_value = base_value + sum(shap_vals)
            
            return {
                "shap_values": shap_dict,
                "top_positive_features": positive_features,
                "top_negative_features": negative_features,
                "base_value": float(base_value),
                "prediction_value": float(prediction_value)
            }
        except Exception as e:
            logger.error(f"Error in get_feature_importance: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_threshold(self) -> float:
        """Get current threshold"""
        return self.threshold


# Global predictor instance
_predictor: Optional[CreditScorePredictor] = None


def get_predictor() -> CreditScorePredictor:
    """
    Get or create the global predictor instance
    
    Returns
    -------
    CreditScorePredictor
        The predictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = CreditScorePredictor()
    return _predictor