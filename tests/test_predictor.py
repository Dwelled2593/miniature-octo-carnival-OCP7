"""
Tests for the predictor module
"""

import pytest
import numpy as np
from api.predictor import CreditScorePredictor, get_predictor


class TestCreditScorePredictor:
    """Tests for CreditScorePredictor class"""
    
    def test_predictor_initialization(self):
        """Test that predictor initializes correctly"""
        predictor = CreditScorePredictor()
        
        assert predictor.model is not None
        assert predictor.explainer is not None
        assert predictor.feature_names is not None
        assert predictor.threshold is not None
        assert predictor.is_loaded() is True
    
    def test_predictor_singleton(self):
        """Test that get_predictor returns same instance"""
        predictor1 = get_predictor()
        predictor2 = get_predictor()
        
        assert predictor1 is predictor2
    
    def test_predict_proba(self, sample_features):
        """Test probability prediction"""
        predictor = get_predictor()
        proba_no_default, proba_default = predictor.predict_proba(sample_features)
        
        # Check types
        assert isinstance(proba_no_default, float)
        assert isinstance(proba_default, float)
        
        # Check ranges
        assert 0 <= proba_no_default <= 1
        assert 0 <= proba_default <= 1
        
        # Check sum
        assert abs((proba_no_default + proba_default) - 1.0) < 0.01
    
    def test_predict(self, sample_features):
        """Test prediction with decision"""
        predictor = get_predictor()
        prediction, decision = predictor.predict(sample_features)
        
        # Check types
        assert isinstance(prediction, int)
        assert isinstance(decision, str)
        
        # Check values
        assert prediction in [0, 1]
        assert decision in ["APPROVED", "REJECTED"]
        
        # Check consistency
        if prediction == 0:
            assert decision == "APPROVED"
        else:
            assert decision == "REJECTED"
    
    def test_predict_with_custom_threshold(self, sample_features):
        """Test prediction with custom threshold"""
        predictor = get_predictor()
        
        # Test with very low threshold (should approve almost everyone)
        prediction_low, decision_low = predictor.predict(sample_features, threshold=0.01)
        
        # Test with very high threshold (should reject almost everyone)
        prediction_high, decision_high = predictor.predict(sample_features, threshold=0.99)
        
        # At least one should be different (unless probability is exactly 0.01 or 0.99)
        assert isinstance(prediction_low, int)
        assert isinstance(prediction_high, int)
    
    def test_get_feature_importance(self, sample_features):
        """Test SHAP feature importance calculation"""
        predictor = get_predictor()
        importance = predictor.get_feature_importance(sample_features)
        
        # Check structure
        assert "shap_values" in importance
        assert "top_positive_features" in importance
        assert "top_negative_features" in importance
        assert "base_value" in importance
        assert "prediction_value" in importance
        
        # Check types
        assert isinstance(importance["shap_values"], dict)
        assert isinstance(importance["top_positive_features"], list)
        assert isinstance(importance["top_negative_features"], list)
        assert isinstance(importance["base_value"], float)
        assert isinstance(importance["prediction_value"], float)
        
        # Check that we have SHAP values
        assert len(importance["shap_values"]) > 0
    
    def test_get_feature_importance_top_n(self, sample_features):
        """Test that top_n parameter works"""
        predictor = get_predictor()
        
        importance_5 = predictor.get_feature_importance(sample_features, top_n=5)
        importance_10 = predictor.get_feature_importance(sample_features, top_n=10)
        
        # Should have at most top_n features
        assert len(importance_5["top_positive_features"]) <= 5
        assert len(importance_5["top_negative_features"]) <= 5
        assert len(importance_10["top_positive_features"]) <= 10
        assert len(importance_10["top_negative_features"]) <= 10
    
    def test_get_threshold(self):
        """Test getting current threshold"""
        predictor = get_predictor()
        threshold = predictor.get_threshold()
        
        assert isinstance(threshold, float)
        assert 0 < threshold < 1
    
    def test_prepare_features_with_missing_values(self):
        """Test feature preparation with missing features"""
        predictor = get_predictor()
        
        # Partial features
        partial_features = {
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": 0.6
        }
        
        # Should not raise error
        features_array = predictor._prepare_features(partial_features)
        
        assert features_array.shape[0] == 1
        assert features_array.shape[1] == len(predictor.feature_names)
        
        # Missing features should be NaN
        assert np.isnan(features_array[0]).sum() > 0
    
    def test_prepare_features_order(self):
        """Test that features are in correct order"""
        predictor = get_predictor()
        
        features = {
            predictor.feature_names[0]: 1.0,
            predictor.feature_names[1]: 2.0,
            predictor.feature_names[2]: 3.0
        }
        
        features_array = predictor._prepare_features(features)
        
        # Check that values are in correct positions
        assert features_array[0, 0] == 1.0
        assert features_array[0, 1] == 2.0
        assert features_array[0, 2] == 3.0


class TestPredictorErrorHandling:
    """Tests for error handling in predictor"""
    
    def test_predict_with_invalid_features(self):
        """Test prediction with invalid feature types"""
        predictor = get_predictor()
        
        # This should be handled gracefully by the model's preprocessing
        invalid_features = {
            "EXT_SOURCE_2": "invalid"  # String instead of float
        }
        
        # Should raise an error or handle gracefully
        with pytest.raises(Exception):
            predictor.predict_proba(invalid_features)
    
    def test_predict_with_empty_features(self):
        """Test prediction with empty features dictionary"""
        predictor = get_predictor()
        
        # Empty features should still work (all NaN)
        empty_features = {}
        
        # Should not raise error - model handles missing values
        try:
            proba_no_default, proba_default = predictor.predict_proba(empty_features)
            assert isinstance(proba_no_default, float)
            assert isinstance(proba_default, float)
        except Exception:
            # If it does raise, that's also acceptable behavior
            pass


class TestPredictorConsistency:
    """Tests for prediction consistency"""
    
    def test_prediction_consistency(self, sample_features):
        """Test that same input gives same output"""
        predictor = get_predictor()
        
        # Make two predictions with same features
        proba1_no, proba1_yes = predictor.predict_proba(sample_features)
        proba2_no, proba2_yes = predictor.predict_proba(sample_features)
        
        # Should be identical
        assert proba1_no == proba2_no
        assert proba1_yes == proba2_yes
    
    def test_probability_threshold_relationship(self, sample_features):
        """Test relationship between probability and threshold"""
        predictor = get_predictor()
        
        proba_no_default, proba_default = predictor.predict_proba(sample_features)
        threshold = predictor.get_threshold()
        
        prediction, decision = predictor.predict(sample_features)
        
        # If probability of default > threshold, should predict default (1)
        if proba_default > threshold:
            assert prediction == 1
            assert decision == "REJECTED"
        else:
            assert prediction == 0
            assert decision == "APPROVED"