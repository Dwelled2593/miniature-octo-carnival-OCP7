"""
Pytest configuration and fixtures
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """
    Create a test client for the FastAPI app
    
    Returns
    -------
    TestClient
        FastAPI test client
    """
    return TestClient(app)


@pytest.fixture
def sample_features():
    """
    Sample client features for testing
    
    Returns
    -------
    dict
        Dictionary with sample features
    """
    return {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.6,
        "DAYS_BIRTH": -15000,
        "AMT_CREDIT": 500000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "DAYS_EMPLOYED": -3000,
        "DAYS_ID_PUBLISH": -2000,
        "REGION_POPULATION_RELATIVE": 0.02,
        "DAYS_LAST_PHONE_CHANGE": -1000
    }


@pytest.fixture
def sample_client_request(sample_features):
    """
    Sample client request for testing
    
    Parameters
    ----------
    sample_features : dict
        Sample features fixture
        
    Returns
    -------
    dict
        Complete client request
    """
    return {
        "features": sample_features,
        "client_id": "TEST_12345"
    }


@pytest.fixture
def sample_batch_request(sample_features):
    """
    Sample batch request for testing
    
    Parameters
    ----------
    sample_features : dict
        Sample features fixture
        
    Returns
    -------
    dict
        Batch request with multiple clients
    """
    return {
        "clients": [
            {
                "features": sample_features,
                "client_id": "TEST_001"
            },
            {
                "features": {**sample_features, "EXT_SOURCE_2": 0.3},
                "client_id": "TEST_002"
            },
            {
                "features": {**sample_features, "EXT_SOURCE_3": 0.8},
                "client_id": "TEST_003"
            }
        ]
    }