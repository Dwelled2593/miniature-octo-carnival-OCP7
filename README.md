# 🏦 Credit Scoring API - Prêt à dépenser

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8-orange.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

API de prédiction de score crédit utilisant Machine Learning pour évaluer la probabilité de défaut de paiement des clients.

**Projet OpenClassroom - OCP7 - Implémentez un modèle de scoring**

---

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [API Documentation](#-api-documentation)
- [Tests](#-tests)
- [Déploiement](#-déploiement)
- [MLOps](#-mlops)
- [Structure du projet](#-structure-du-projet)
- [Contribuer](#-contribuer)

---

## 🎯 Vue d'ensemble

Cette API permet de prédire la probabilité qu'un client fasse défaut sur son crédit. Elle utilise un modèle LightGBM entraîné sur des données historiques et optimisé avec un seuil de décision métier qui prend en compte le coût différentiel entre faux positifs et faux négatifs.

### Contexte métier

- **Faux Négatif (FN)** : Coût = 1 (client solvable refusé, manque à gagner)
- **Faux Positif (FP)** : Coût = 10 (client insolvable accepté, perte en capital)
- **Seuil optimal** : Calculé pour minimiser le coût métier total

### Performances du modèle

- **ROC AUC** : ~0.76
- **Business Cost** : Optimisé avec seuil personnalisé
- **Explicabilité** : SHAP values pour chaque prédiction

---

## ✨ Fonctionnalités

### API Endpoints

- ✅ **GET /health** - Vérification de l'état de l'API
- ✅ **POST /predict** - Prédiction pour un client
- ✅ **POST /predict/batch** - Prédictions en batch
- ✅ **POST /feature-importance** - Analyse SHAP des features

### Capacités

- 🔮 Prédiction de probabilité de défaut
- 📊 Décision automatique (APPROVED/REJECTED)
- 🔍 Explicabilité via SHAP values
- 📈 Traitement batch pour plusieurs clients
- 🎯 Seuil de décision optimisé métier
- 📝 Documentation interactive (Swagger UI)

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│  FastAPI     │─────▶│  LightGBM   │
│ (Streamlit) │      │     API      │      │    Model    │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │     SHAP     │
                     │  Explainer   │
                     └──────────────┘
```

### Technologies utilisées

- **API Framework** : FastAPI
- **ML Model** : LightGBM avec pipeline scikit-learn
- **Explicabilité** : SHAP
- **Tracking** : MLflow
- **Data Drift** : Evidently
- **Tests** : Pytest
- **Déploiement** : Docker + Google Cloud Run
- **CI/CD** : GitHub Actions

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- pip
- Git

### Installation locale

```bash
# Cloner le repository
git clone https://github.com/votre-username/miniature-octo-carnival-OCP7.git
cd miniature-octo-carnival-OCP7

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Fichiers requis

Assurez-vous que les fichiers suivants sont présents :

- `selected_model.sav` - Modèle LightGBM entraîné
- `explainer.sav` - SHAP explainer
- `feature_names.sav` - Liste des features
- `optimal_threshold.json` - Seuil de décision optimal

---

## 💻 Utilisation

### Lancer l'API localement

```bash
# Méthode 1 : Uvicorn direct
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Méthode 2 : Python
python -m api.main
```

L'API sera accessible sur : `http://localhost:8080`

### Documentation interactive

- **Swagger UI** : http://localhost:8080/
- **ReDoc** : http://localhost:8080/redoc
- **OpenAPI Schema** : http://localhost:8080/openapi.json

### Exemples d'utilisation

#### Python

```python
import requests

# Health check
response = requests.get("http://localhost:8080/health")
print(response.json())

# Prédiction
client_data = {
    "features": {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.6,
        "DAYS_BIRTH": -15000,
        "AMT_CREDIT": 500000
    },
    "client_id": "12345"
}

response = requests.post(
    "http://localhost:8080/predict",
    json=client_data
)
print(response.json())
```

#### cURL

```bash
# Health check
curl http://localhost:8080/health

# Prédiction
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "EXT_SOURCE_2": 0.5,
      "EXT_SOURCE_3": 0.6,
      "DAYS_BIRTH": -15000,
      "AMT_CREDIT": 500000
    },
    "client_id": "12345"
  }'
```

---

## 📚 API Documentation

### POST /predict

Prédit le score crédit pour un client.

**Request Body:**
```json
{
  "features": {
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.6,
    "DAYS_BIRTH": -15000,
    "AMT_CREDIT": 500000
  },
  "client_id": "12345"
}
```

**Response:**
```json
{
  "client_id": "12345",
  "probability_default": 0.23,
  "probability_no_default": 0.77,
  "prediction": 0,
  "decision": "APPROVED",
  "threshold_used": 0.48
}
```

### POST /feature-importance

Analyse l'importance des features pour une prédiction.

**Response:**
```json
{
  "client_id": "12345",
  "shap_values": {
    "EXT_SOURCE_2": -0.15,
    "EXT_SOURCE_3": -0.12,
    "DAYS_BIRTH": 0.08
  },
  "top_positive_features": [...],
  "top_negative_features": [...],
  "base_value": 0.5,
  "prediction_value": 0.23
}
```

---

## 🧪 Tests

### Lancer les tests

```bash
# Tous les tests
pytest

# Avec couverture
pytest --cov=api --cov-report=html

# Tests spécifiques
pytest tests/test_api.py
pytest tests/test_predictor.py

# Mode verbose
pytest -v
```

### Couverture des tests

Les tests couvrent :
- ✅ Tous les endpoints API
- ✅ Validation des données
- ✅ Logique de prédiction
- ✅ Calcul SHAP
- ✅ Gestion d'erreurs
- ✅ Documentation API

---

## 🌐 Déploiement

### Docker

```bash
# Build l'image
docker build -t credit-scoring-api .

# Lancer le container
docker run -p 8080:8080 credit-scoring-api
```

### Google Cloud Run

```bash
# Authentification
gcloud auth login

# Configuration du projet
gcloud config set project YOUR_PROJECT_ID

# Build et push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/credit-api

# Déploiement
gcloud run deploy credit-api \
  --image gcr.io/YOUR_PROJECT_ID/credit-api \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

### Variables d'environnement

```bash
MODEL_PATH=/path/to/selected_model.sav
EXPLAINER_PATH=/path/to/explainer.sav
FEATURE_NAMES_PATH=/path/to/feature_names.sav
THRESHOLD_PATH=/path/to/optimal_threshold.json
LOG_LEVEL=INFO
```

---

## 🔄 MLOps

### MLflow Tracking

Le projet utilise MLflow pour tracker les expérimentations :

```bash
# Lancer MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Accéder à l'interface
# http://localhost:5000
```

### Data Drift Monitoring

Analyse du data drift avec Evidently :

```bash
# Générer le rapport
jupyter notebook drift.ipynb

# Ouvrir le rapport HTML
open data_drift.html
```

### CI/CD Pipeline

GitHub Actions automatise :
1. ✅ Tests unitaires
2. ✅ Vérification du code (linting)
3. ✅ Build Docker
4. ✅ Déploiement Cloud Run

---

## 📁 Structure du projet

```
miniature-octo-carnival-OCP7/
├── api/                        # Code de l'API
│   ├── __init__.py
│   ├── main.py                # Application FastAPI
│   ├── models.py              # Modèles Pydantic
│   ├── predictor.py           # Logique de prédiction
│   └── config.py              # Configuration
├── tests/                     # Tests unitaires
│   ├── __init__.py
│   ├── conftest.py           # Fixtures pytest
│   ├── test_api.py           # Tests API
│   └── test_predictor.py     # Tests prédicteur
├── notebooks/                 # Notebooks Jupyter
│   ├── modeling.ipynb        # Modélisation + MLflow
│   ├── drift.ipynb           # Analyse data drift
│   ├── exploration.ipynb     # Exploration données
│   └── test_api.ipynb        # Tests API
├── streamlit/                 # Dashboard Streamlit
│   └── app.py
├── .github/                   # GitHub Actions
│   └── workflows/
│       ├── test.yml
│       └── deploy.yml
├── models/                    # Modèles sauvegardés
│   ├── selected_model.sav
│   ├── explainer.sav
│   ├── feature_names.sav
│   └── optimal_threshold.json
├── Dockerfile                 # Configuration Docker
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce fichier
└── .gitignore                # Fichiers ignorés
```

---

## 👥 Contribuer

Les contributions sont les bienvenues !

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 📧 Contact

**Projet OpenClassroom OCP7**

- Repository: [https://github.com/votre-username/miniature-octo-carnival-OCP7](https://github.com/votre-username/miniature-octo-carnival-OCP7)
- Documentation: [API Docs](http://localhost:8080/)

---

## 🙏 Remerciements

- OpenClassroom pour le projet
- Kaggle pour les données
- La communauté open-source pour les outils utilisés

---

**Made with ❤️ for OpenClassroom OCP7**
