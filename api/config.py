"""
Configuration for the Credit Scoring API
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "selected_model.sav"))
EXPLAINER_PATH = os.getenv("EXPLAINER_PATH", str(BASE_DIR / "explainer.sav"))
FEATURE_NAMES_PATH = os.getenv("FEATURE_NAMES_PATH", str(BASE_DIR / "feature_names.sav"))
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", str(BASE_DIR / "optimal_threshold.json"))

# API Configuration
API_TITLE = "Credit Scoring API"
API_DESCRIPTION = """
API de prédiction de score crédit pour Prêt à dépenser.

Cette API permet de :
* Prédire la probabilité de défaut de paiement d'un client
* Obtenir une décision d'acceptation ou de refus de crédit
* Analyser l'importance des features (SHAP values)

Développé dans le cadre du projet OpenClassroom OCP7.
"""
API_VERSION = "1.0.0"

# Business logic
DEFAULT_THRESHOLD = 0.5  # Will be overridden by optimal_threshold.json
FN_COST = 1  # False Negative cost
FP_COST = 10  # False Positive cost (loan to bad client)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# CORS
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:8501",  # Streamlit default port
    "http://localhost:3000",
    "*"  # Allow all in development (restrict in production)
]