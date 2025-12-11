"""
Streamlit Dashboard for Credit Scoring API Testing
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional

# Configuration
st.set_page_config(
    page_title="Credit Scoring API - Test Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .approved {
        color: #28a745;
        font-weight: bold;
    }
    .rejected {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class APIClient:
    """Client for interacting with the Credit Scoring API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict:
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict(self, features: Dict, client_id: Optional[str] = None) -> Dict:
        """Get prediction for a client"""
        try:
            payload = {"features": features}
            if client_id:
                payload["client_id"] = client_id
            
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_feature_importance(self, features: Dict, client_id: Optional[str] = None) -> Dict:
        """Get SHAP feature importance"""
        try:
            payload = {"features": features}
            if client_id:
                payload["client_id"] = client_id
            
            response = requests.post(
                f"{self.base_url}/feature-importance",
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def create_gauge_chart(probability: float, threshold: float) -> go.Figure:
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de Défaut (%)", 'font': {'size': 20}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': '#90EE90'},
                {'range': [threshold * 100, 100], 'color': '#FFB6C1'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_shap_waterfall(shap_data: Dict) -> go.Figure:
    """Create waterfall chart for SHAP values"""
    # Get top positive and negative features
    top_positive = shap_data.get('top_positive_features', [])[:10]
    top_negative = shap_data.get('top_negative_features', [])[:10]
    
    # Combine and sort
    all_features = top_positive + top_negative
    all_features.sort(key=lambda x: abs(x['value']), reverse=True)
    
    if not all_features:
        return go.Figure()
    
    features = [f['feature'] for f in all_features]
    values = [f['value'] for f in all_features]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.4f}" for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Top Features Impact (SHAP Values)",
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=200, r=20, t=50, b=20)
    )
    
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🏦 Credit Scoring API - Test Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_url = st.text_input(
            "API URL",
            value="http://localhost:8080",
            help="URL de l'API (local ou Cloud Run)"
        )
        
        client = APIClient(api_url)
        
        # Health check
        st.subheader("🏥 API Health")
        if st.button("Check Health"):
            with st.spinner("Checking..."):
                health = client.health_check()
                if "error" in health:
                    st.error(f"❌ Error: {health['error']}")
                else:
                    st.success("✅ API is healthy")
                    st.json(health)
        
        st.divider()
        
        # Sample data
        st.subheader("📊 Sample Features")
        use_sample = st.checkbox("Use sample data", value=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📈 Batch Predictions", "📚 Documentation"])
    
    with tab1:
        st.header("Single Client Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Client Features")
            
            client_id = st.text_input("Client ID (optional)", value="TEST_001")
            
            if use_sample:
                features = {
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
            else:
                features = {}
            
            # Feature input
            st.write("Enter feature values:")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                features["EXT_SOURCE_2"] = st.number_input(
                    "EXT_SOURCE_2", 
                    value=features.get("EXT_SOURCE_2", 0.5),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01
                )
                features["EXT_SOURCE_3"] = st.number_input(
                    "EXT_SOURCE_3",
                    value=features.get("EXT_SOURCE_3", 0.6),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01
                )
                features["DAYS_BIRTH"] = st.number_input(
                    "DAYS_BIRTH",
                    value=features.get("DAYS_BIRTH", -15000),
                    step=100
                )
                features["AMT_CREDIT"] = st.number_input(
                    "AMT_CREDIT",
                    value=features.get("AMT_CREDIT", 500000.0),
                    step=1000.0
                )
                features["AMT_ANNUITY"] = st.number_input(
                    "AMT_ANNUITY",
                    value=features.get("AMT_ANNUITY", 25000.0),
                    step=100.0
                )
            
            with col_b:
                features["AMT_GOODS_PRICE"] = st.number_input(
                    "AMT_GOODS_PRICE",
                    value=features.get("AMT_GOODS_PRICE", 450000.0),
                    step=1000.0
                )
                features["DAYS_EMPLOYED"] = st.number_input(
                    "DAYS_EMPLOYED",
                    value=features.get("DAYS_EMPLOYED", -3000),
                    step=100
                )
                features["DAYS_ID_PUBLISH"] = st.number_input(
                    "DAYS_ID_PUBLISH",
                    value=features.get("DAYS_ID_PUBLISH", -2000),
                    step=100
                )
                features["REGION_POPULATION_RELATIVE"] = st.number_input(
                    "REGION_POPULATION_RELATIVE",
                    value=features.get("REGION_POPULATION_RELATIVE", 0.02),
                    step=0.001,
                    format="%.4f"
                )
                features["DAYS_LAST_PHONE_CHANGE"] = st.number_input(
                    "DAYS_LAST_PHONE_CHANGE",
                    value=features.get("DAYS_LAST_PHONE_CHANGE", -1000),
                    step=100
                )
            
            # Predict button
            if st.button("🔮 Get Prediction", type="primary", use_container_width=True):
                with st.spinner("Predicting..."):
                    # Get prediction
                    prediction = client.predict(features, client_id)
                    
                    if "error" in prediction:
                        st.error(f"❌ Error: {prediction['error']}")
                    else:
                        # Store in session state
                        st.session_state['last_prediction'] = prediction
                        st.session_state['last_features'] = features
                        st.session_state['last_client_id'] = client_id
        
        with col2:
            st.subheader("Prediction Result")
            
            if 'last_prediction' in st.session_state:
                pred = st.session_state['last_prediction']
                
                # Decision
                decision = pred.get('decision', 'UNKNOWN')
                decision_class = "approved" if decision == "APPROVED" else "rejected"
                st.markdown(f'<h2 class="{decision_class}">{decision}</h2>', 
                           unsafe_allow_html=True)
                
                # Metrics
                st.metric("Probability Default", f"{pred.get('probability_default', 0):.2%}")
                st.metric("Probability No Default", f"{pred.get('probability_no_default', 0):.2%}")
                st.metric("Threshold Used", f"{pred.get('threshold_used', 0):.2%}")
                
                # Gauge chart
                st.plotly_chart(
                    create_gauge_chart(
                        pred.get('probability_default', 0),
                        pred.get('threshold_used', 0.5)
                    ),
                    use_container_width=True
                )
                
                # Get SHAP values
                if st.button("📊 Show Feature Importance"):
                    with st.spinner("Calculating SHAP values..."):
                        shap_data = client.get_feature_importance(
                            st.session_state['last_features'],
                            st.session_state.get('last_client_id')
                        )
                        
                        if "error" not in shap_data:
                            st.session_state['last_shap'] = shap_data
        
        # SHAP visualization
        if 'last_shap' in st.session_state:
            st.divider()
            st.subheader("🔍 Feature Importance Analysis (SHAP)")
            
            shap_data = st.session_state['last_shap']
            
            col_shap1, col_shap2 = st.columns([2, 1])
            
            with col_shap1:
                st.plotly_chart(
                    create_shap_waterfall(shap_data),
                    use_container_width=True
                )
            
            with col_shap2:
                st.write("**Base Value:**", f"{shap_data.get('base_value', 0):.4f}")
                st.write("**Prediction Value:**", f"{shap_data.get('prediction_value', 0):.4f}")
                
                st.write("**Top Positive Features:**")
                for feat in shap_data.get('top_positive_features', [])[:5]:
                    st.write(f"- {feat['feature']}: {feat['value']:.4f}")
                
                st.write("**Top Negative Features:**")
                for feat in shap_data.get('top_negative_features', [])[:5]:
                    st.write(f"- {feat['feature']}: {feat['value']:.4f}")
    
    with tab2:
        st.header("Batch Predictions")
        st.info("📝 Upload a CSV file with client features for batch predictions")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:", df.head())
            
            if st.button("Run Batch Predictions"):
                st.warning("Batch prediction endpoint to be implemented")
    
    with tab3:
        st.header("📚 API Documentation")
        
        st.markdown(f"""
        ### API Endpoints
        
        **Base URL:** `{api_url}`
        
        #### GET /health
        Check API health status
        
        #### POST /predict
        Get credit score prediction for a single client
        
        **Request:**
        ```json
        {{
          "features": {{
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": 0.6,
            ...
          }},
          "client_id": "optional_id"
        }}
        ```
        
        **Response:**
        ```json
        {{
          "client_id": "optional_id",
          "probability_default": 0.23,
          "probability_no_default": 0.77,
          "prediction": 0,
          "decision": "APPROVED",
          "threshold_used": 0.48
        }}
        ```
        
        #### POST /feature-importance
        Get SHAP feature importance values
        
        ### Quick Links
        - [Interactive API Docs]({api_url}/)
        - [ReDoc Documentation]({api_url}/redoc)
        - [OpenAPI Schema]({api_url}/openapi.json)
        """)


if __name__ == "__main__":
    main()