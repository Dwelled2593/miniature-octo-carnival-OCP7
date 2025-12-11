"""
Tests for API endpoints
"""

import pytest
from fastapi import status


class TestHealthEndpoint:
    """Tests for /health endpoint"""
    
    def test_health_check_success(self, client):
        """Test that health check returns 200 and correct structure"""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"
    
    def test_health_check_structure(self, client):
        """Test health check response structure"""
        response = client.get("/health")
        data = response.json()
        
        # Verify all required fields are present
        required_fields = ["status", "model_loaded", "version"]
        for field in required_fields:
            assert field in data


class TestPredictEndpoint:
    """Tests for /predict endpoint"""
    
    def test_predict_success(self, client, sample_client_request):
        """Test successful prediction"""
        response = client.post("/predict", json=sample_client_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify response structure
        assert "client_id" in data
        assert "probability_default" in data
        assert "probability_no_default" in data
        assert "prediction" in data
        assert "decision" in data
        assert "threshold_used" in data
        
        # Verify data types and ranges
        assert isinstance(data["probability_default"], float)
        assert isinstance(data["probability_no_default"], float)
        assert 0 <= data["probability_default"] <= 1
        assert 0 <= data["probability_no_default"] <= 1
        assert data["prediction"] in [0, 1]
        assert data["decision"] in ["APPROVED", "REJECTED"]
        
        # Verify probabilities sum to 1 (approximately)
        prob_sum = data["probability_default"] + data["probability_no_default"]
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_predict_with_client_id(self, client, sample_client_request):
        """Test that client_id is returned in response"""
        response = client.post("/predict", json=sample_client_request)
        data = response.json()
        
        assert data["client_id"] == sample_client_request["client_id"]
    
    def test_predict_without_client_id(self, client, sample_features):
        """Test prediction without client_id"""
        request = {"features": sample_features}
        response = client.post("/predict", json=request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["client_id"] is None
    
    def test_predict_empty_features(self, client):
        """Test prediction with empty features"""
        request = {"features": {}}
        response = client.post("/predict", json=request)
        
        # Should return 422 for validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_missing_features_field(self, client):
        """Test prediction without features field"""
        request = {"client_id": "TEST"}
        response = client.post("/predict", json=request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_feature_type(self, client):
        """Test prediction with invalid feature type"""
        request = {
            "features": {
                "EXT_SOURCE_2": "invalid_string"  # Should be float
            }
        }
        response = client.post("/predict", json=request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_decision_logic(self, client, sample_features):
        """Test that decision logic is correct"""
        request = {"features": sample_features}
        response = client.post("/predict", json=request)
        data = response.json()
        
        # If prediction is 0 (no default), decision should be APPROVED
        # If prediction is 1 (default), decision should be REJECTED
        if data["prediction"] == 0:
            assert data["decision"] == "APPROVED"
        else:
            assert data["decision"] == "REJECTED"


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint"""
    
    def test_batch_predict_success(self, client, sample_batch_request):
        """Test successful batch prediction"""
        response = client.post("/predict/batch", json=sample_batch_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify response structure
        assert "predictions" in data
        assert "total_clients" in data
        assert "approved_count" in data
        assert "rejected_count" in data
        
        # Verify counts
        assert data["total_clients"] == len(sample_batch_request["clients"])
        assert len(data["predictions"]) == data["total_clients"]
        assert data["approved_count"] + data["rejected_count"] == data["total_clients"]
    
    def test_batch_predict_individual_predictions(self, client, sample_batch_request):
        """Test that each prediction in batch has correct structure"""
        response = client.post("/predict/batch", json=sample_batch_request)
        data = response.json()
        
        for prediction in data["predictions"]:
            assert "client_id" in prediction
            assert "probability_default" in prediction
            assert "probability_no_default" in prediction
            assert "prediction" in prediction
            assert "decision" in prediction
            assert "threshold_used" in prediction
    
    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty client list"""
        request = {"clients": []}
        response = client.post("/predict/batch", json=request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_clients"] == 0
        assert data["approved_count"] == 0
        assert data["rejected_count"] == 0
    
    def test_batch_predict_single_client(self, client, sample_client_request):
        """Test batch prediction with single client"""
        request = {"clients": [sample_client_request]}
        response = client.post("/predict/batch", json=request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_clients"] == 1


class TestFeatureImportanceEndpoint:
    """Tests for /feature-importance endpoint"""
    
    def test_feature_importance_success(self, client, sample_client_request):
        """Test successful feature importance calculation"""
        response = client.post("/feature-importance", json=sample_client_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify response structure
        assert "client_id" in data
        assert "shap_values" in data
        assert "top_positive_features" in data
        assert "top_negative_features" in data
        assert "base_value" in data
        assert "prediction_value" in data
    
    def test_feature_importance_shap_values(self, client, sample_client_request):
        """Test that SHAP values are returned correctly"""
        response = client.post("/feature-importance", json=sample_client_request)
        data = response.json()
        
        # SHAP values should be a dictionary
        assert isinstance(data["shap_values"], dict)
        assert len(data["shap_values"]) > 0
        
        # All values should be floats
        for feature, value in data["shap_values"].items():
            assert isinstance(feature, str)
            assert isinstance(value, (int, float))
    
    def test_feature_importance_top_features(self, client, sample_client_request):
        """Test that top features are returned"""
        response = client.post("/feature-importance", json=sample_client_request)
        data = response.json()
        
        # Should have lists of top features
        assert isinstance(data["top_positive_features"], list)
        assert isinstance(data["top_negative_features"], list)
        
        # Each feature should have name and value
        for feature in data["top_positive_features"]:
            assert "feature" in feature
            assert "value" in feature
            assert feature["value"] > 0  # Positive features
        
        for feature in data["top_negative_features"]:
            assert "feature" in feature
            assert "value" in feature
            assert feature["value"] < 0  # Negative features
    
    def test_feature_importance_base_value(self, client, sample_client_request):
        """Test that base value is returned"""
        response = client.post("/feature-importance", json=sample_client_request)
        data = response.json()
        
        assert isinstance(data["base_value"], (int, float))
        assert isinstance(data["prediction_value"], (int, float))


class TestAPIDocumentation:
    """Tests for API documentation"""
    
    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available"""
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_docs_available(self, client):
        """Test that Swagger UI docs are available"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
    
    def test_redoc_available(self, client):
        """Test that ReDoc documentation is available"""
        response = client.get("/redoc")
        assert response.status_code == status.HTTP_200_OK


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_invalid_endpoint(self, client):
        """Test that invalid endpoint returns 404"""
        response = client.get("/invalid-endpoint")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_invalid_method(self, client):
        """Test that invalid HTTP method returns 405"""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED