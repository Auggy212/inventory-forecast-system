"""
Intelligent Inventory & Demand Forecasting System
Premium Edition - Professional UI/UX Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import calendar
import warnings
warnings.filterwarnings('ignore')

# Import forecasting libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Additional imports
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import time
from math import sqrt
import importlib

# Optional Gemini integration
genai_spec = importlib.util.find_spec("google.generativeai")
if genai_spec is not None:
    genai = importlib.import_module("google.generativeai")
    GEMINI_AVAILABLE = True
else:
    genai = None
    GEMINI_AVAILABLE = False

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="IntelliStock AI - Inventory Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS with animations and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 0;
        background: #f8fafc;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar text color - white */
    section[data-testid="stSidebar"] {
        color: white !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Premium Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 0 0 30px 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 15s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.3; }
    }
    
    .header-content {
        position: relative;
        z-index: 1;
        text-align: center;
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
    }
    
    .sub-title {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        font-weight: 400;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Animation Keyframes */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Premium Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: black;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        font-weight: 500;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .metric-delta.positive {
        background: #d4f8e8;
        color: #047857;
    }
    
    .metric-delta.negative {
        background: #fee2e2;
        color: #dc2626;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        cursor: pointer;
        text-align: center;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }
    
    .feature-icon {
        width: 60px;
        height: 60px;
        margin: 0 auto 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    
    /* Navigation Tabs */
    .custom-tabs {
        background: white;
        padding: 0.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        display: flex;
        gap: 0.5rem;
        overflow-x: auto;
    }
    
    .tab-button {
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        background: transparent;
        color: #4a5568;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        white-space: nowrap;
    }
    
    .tab-button:hover {
        background: #f7fafc;
    }
    
    .tab-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Action Buttons */
    .action-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        display: inline-block;
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Model Selection Buttons */
    .model-button-container {
        position: relative;
    }
    
    .model-button-selected {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: 3px solid #10b981 !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5) !important;
        transform: scale(1.05);
        font-weight: 700 !important;
    }
    
    .model-button-selected::after {
        content: "✓";
        position: absolute;
        top: 5px;
        right: 10px;
        background: white;
        color: #10b981;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
    }
    
    .model-selection-banner {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: slideIn 0.5s ease-out;
    }
    
    .model-selection-banner-icon {
        font-size: 2rem;
    }
    
    .model-selection-banner-text {
        flex: 1;
    }
    
    .model-selection-banner-title {
        font-size: 1.125rem;
        font-weight: 700;
        color: #e6e6e6;
        margin: 0;
    }
    
    .model-selection-banner-subtitle {
        font-size: 0.875rem;
        color: #e6e6e6;
        margin: 0.25rem 0 0 0;
    }
    
    /* Section Cards */
    .section-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        animation: slideIn 0.6s ease-out;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .status-indicator.success {
        background: #10b981;
    }
    
    .status-indicator.warning {
        background: #f59e0b;
    }
    
    .status-indicator.error {
        background: #ef4444;
    }
    
    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e5e7ff 100%);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-card h4 {
        color: #667eea;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .info-card p {
        color: #4a5568;
        margin: 0;
        font-size: 0.875rem;
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .chart-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1a202c;
    }
    
    /* Loading Animation */
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f4f6;
        border-top: 5px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Streamlit Overrides */
    /* Targeted heading colors: dark headings on white cards, white headings on colored headers */

    /* Headings that live in white/neutral cards and chart containers -> dark */
    .section-card h1,
    .section-card h2,
    .section-card h3,
    .section-card h4,
    .chart-container .chart-title,
    .chart-header .chart-title,
    .info-card h4,
    .feature-card h3,
    div[data-testid="stMarkdownContainer"] .section-card h1,
    div[data-testid="stMarkdownContainer"] .section-card h2,
    div[data-testid="stMarkdownContainer"] .section-card h3 {
        color: #1a202c !important;
        -webkit-text-fill-color: #1a202c !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }

    /* Headings that sit on the main gradient header / banner -> keep white */
    .main-header .main-title,
    .main-header h1,
    .main-header h2,
    .main-header h3,
    .header-content h1,
    .header-content .sub-title,
    .model-selection-banner-title,
    .model-selection-banner-subtitle {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }

    /* Fallback: ensure any Streamlit-generated wrapper class headings are dark unless in main header */
    html body .stApp [class*="css-"] h1,
    html body .stApp [class*="css-"] h2,
    html body .stApp [class*="css-"] h3 {
        color: #1a202c !important;
        -webkit-text-fill-color: #1a202c !important;
    }

    /* Preserve white title on gradient header */
    .main-header .main-title, .main-header h1 {
        color: #ffffff !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
        position: relative;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Selected model button styling */
    div[data-testid="stButton"]:has(button[aria-pressed="true"]) button,
    .model-selected-button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: 3px solid #10b981 !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5) !important;
        font-weight: 700 !important;
    }
    
    /* Add checkmark to selected button via CSS */
    .model-selected-button::before {
        content: "✓ Selected";
        position: absolute;
        top: 5px;
        right: 10px;
        background: rgba(255, 255, 255, 0.3);
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border-color: #e2e8f0;
    }
    
    .stTextInput > div > div {
        border-radius: 10px;
        border-color: #e2e8f0;
    }
    
    /* Progress Bar */
    .progress-container {
        background: #f3f4f6;
        border-radius: 10px;
        padding: 0.25rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .feature-card {
            margin-bottom: 1rem;
        }
        
        .custom-tabs {
            flex-wrap: wrap;
        }
            
    }
   
            
            
</style>
""", unsafe_allow_html=True)


# Initialize session state with more structure
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'data' not in st.session_state:
    st.session_state.data = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'inventory_recommendations' not in st.session_state:
    st.session_state.inventory_recommendations = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'forecast_metrics' not in st.session_state:
    st.session_state.forecast_metrics = None
if 'backtest_metrics' not in st.session_state:
    st.session_state.backtest_metrics = None

# Utility functions for UI components
def create_metric_card(label, value, delta=None, delta_type="positive", icon="📊"):
    """Create a premium metric card"""
    delta_html = ""
    if delta:
        delta_class = "positive" if delta_type == "positive" else "negative"
        delta_symbol = "↑" if delta_type == "positive" else "↓"
        delta_html = f'<div class="metric-delta {delta_class}">{delta_symbol} {delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <div class="metric-label">{label}</div>
        </div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

def create_feature_card(title, description, icon, action_text="Explore"):
    """Create a feature card"""
    return f"""
    <div class="feature-card">
        <div class="feature-icon">
            <span style="color: white; font-size: 2rem;">{icon}</span>
        </div>
        <h3 style="margin-bottom: 0.5rem; color: #1a202c;">{title}</h3>
        <p style="color: #718096; margin-bottom: 1.5rem;">{description}</p>
    </div>
    """

def configure_integration_settings():
    """Sidebar configuration for API keys and integrations."""
    with st.sidebar:
        st.markdown("### 🔐 Integration Settings")
        st.markdown("Provide your Gemini API key to activate GenAI-powered insights. Keys are stored in-memory for this session only.")
        api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.get('gemini_api_key', ""),
            type="password",
            help="Paste your Google Gemini (free tier) API key. It will not be persisted."
        )
        if api_key != st.session_state.get('gemini_api_key', ""):
            st.session_state.gemini_api_key = api_key.strip()
            st.success("API key updated for this session.")
        
        if GEMINI_AVAILABLE and st.session_state.gemini_api_key:
            try:
                genai.configure(api_key=st.session_state.gemini_api_key)
                st.caption("✅ Gemini client configured.")
            except Exception as e:
                st.error(f"Gemini configuration failed: {str(e)}")
        # Removed install message - user can install package if needed

def compute_backtest_metrics(df, model_name=None, test_window=None):
    """Run a hold-out backtest using the selected forecasting model."""
    if df is None or 'sales' not in df.columns or len(df) < 30:
        return None
    
    selected_model = model_name or st.session_state.get('selected_model') or 'Prophet'
    cached = st.session_state.get('backtest_metrics')
    if cached and cached.get('model') == selected_model:
        if not test_window or cached.get('test_days') == test_window:
            return cached
    
    try:
        forecaster = DemandForecaster(df)
        metrics = forecaster.backtest(model_name=selected_model, test_window=test_window)
        st.session_state.backtest_metrics = metrics
        return metrics
    except Exception as exc:
        st.warning(f"Backtest unavailable: {exc}")
        return None

def analyze_inventory_challenges(df, inventory_col=None, inventory_metrics=None):
    """Derive headline inventory risks and opportunities"""
    insights = {
        'overstock_days': None,
        'stockout_days': None,
        'seasonality_peak': None,
        'seasonality_trough': None,
        'demand_volatility': None,
        'current_gap': None
    }
    
    if df is None or 'sales' not in df.columns:
        return insights
    
    data = df.copy()
    data = data.sort_values('date')
    
    if inventory_col and inventory_col in data.columns:
        inventory_series = pd.to_numeric(data[inventory_col], errors='coerce').fillna(0)
        demand_series = data['sales']
        insights['stockout_days'] = int((inventory_series <= demand_series * 0.1).sum())
        insights['overstock_days'] = int((inventory_series >= demand_series * 2).sum())
    elif inventory_metrics and inventory_metrics.get('reorder_point') is not None:
        current_inventory = inventory_metrics.get('current_inventory', 0)
        reorder_point = inventory_metrics.get('reorder_point', 0)
        recommended_max = inventory_metrics.get('recommended_max_inventory', reorder_point * 1.4)
        insights['current_gap'] = current_inventory - reorder_point
        if current_inventory > recommended_max:
            insights['overstock_days'] = max(int((current_inventory - recommended_max) / max(inventory_metrics.get('avg_daily_demand', 1), 1)), 0)
        if current_inventory < reorder_point:
            insights['stockout_days'] = max(int(abs(current_inventory - reorder_point) / max(inventory_metrics.get('avg_daily_demand', 1), 1)), 0)
    
    data['month'] = data['date'].dt.month
    monthly = data.groupby('month')['sales'].mean()
    if len(monthly) >= 2:
        peak_month = monthly.idxmax()
        trough_month = monthly.idxmin()
        insights['seasonality_peak'] = peak_month
        insights['seasonality_trough'] = trough_month
    
    if data['sales'].mean() > 0:
        insights['demand_volatility'] = float(data['sales'].std() / data['sales'].mean() * 100)
    
    return insights

def evaluate_data_sources(df, column_mapping):
    """Summarize availability of recommended data assets"""
    sources = []
    available_cols = set(df.columns) if df is not None else set()
    inventory_col = column_mapping.get('inventory') if column_mapping else None
    
    sources.append({
        'name': 'Historical Sales',
        'status': 'Available ✅' if 'sales' in available_cols else 'Missing ⚠️',
        'detail': 'Daily transaction history used to train forecasting models.'
    })
    sources.append({
        'name': 'Inventory Positions',
        'status': 'Available ✅' if inventory_col and inventory_col in available_cols else 'Optional',
        'detail': 'On-hand and on-order balances to detect stockouts and overstock risk.'
    })
    
    promo_cols = [c for c in available_cols if 'promo' in c.lower() or 'campaign' in c.lower()]
    sources.append({
        'name': 'Promotions & Price Events',
        'status': 'Available ✅' if promo_cols else 'Recommended',
        'detail': 'Marketing levers that impact demand uplift.'
    })
    
    external_cols = [c for c in available_cols if any(keyword in c.lower() for keyword in ['weather', 'google', 'macro', 'holiday'])]
    sources.append({
        'name': 'External Drivers',
        'status': 'Available ✅' if external_cols else 'Recommended',
        'detail': 'Signals such as holidays, weather, or macro trends to boost accuracy.'
    })
    
    return sources

def show_loading_animation(text="Processing..."):
    """Show a premium loading animation"""
    return st.markdown(f"""
    <div class="loading-animation">
        <div style="text-align: center;">
            <div class="loading-spinner"></div>
            <p style="margin-top: 1rem; color: #718096;">{text}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Header Component
def show_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <h1 class="main-title">🎯 IntelliStock AI</h1>
            <p class="sub-title">Transform your inventory management with AI-powered insights</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Navigation Component
def show_navigation():
    """Show custom navigation tabs"""
    tabs = {
        'home': {'label': '🏠 Home', 'key': 'home'},
        'upload': {'label': '📤 Data Upload', 'key': 'upload'},
        'forecast': {'label': '📊 Forecasting', 'key': 'forecast'},
        'inventory': {'label': '📦 Inventory', 'key': 'inventory'},
        'analytics': {'label': '💰 Analytics', 'key': 'analytics'},
        'boardroom': {'label': '🏢 Boardroom', 'key': 'boardroom'},
        'reports': {'label': '📑 Reports', 'key': 'reports'}
    }
    
    cols = st.columns(len(tabs))
    for idx, (key, tab) in enumerate(tabs.items()):
        with cols[idx]:
            if st.button(tab['label'], key=f"nav_{key}", use_container_width=True):
                st.session_state.current_page = key

# Data processing classes with flexible column detection
class DataProcessor:
    """Handles data loading and preprocessing with intelligent column detection"""
    
    # Common column name patterns for auto-detection
    DATE_PATTERNS = [
        'date', 'time', 'timestamp', 'day', 'datetime', 'period', 
        'month', 'year', 'week', 'quarter', 'created_at', 'updated_at',
        'transaction_date', 'order_date', 'ship_date', 'delivery_date'
    ]
    
    DEMAND_PATTERNS = [
        'sales', 'demand', 'quantity', 'qty', 'units', 'volume', 
        'orders', 'order_qty', 'ordered', 'sold', 'consumption',
        'usage', 'withdrawal', 'issue', 'delivery', 'shipped',
        'requested', 'required', 'needed'
    ]
    
    INVENTORY_PATTERNS = [
        'inventory', 'stock', 'on_hand', 'onhand', 'available', 
        'quantity_on_hand', 'qoh', 'stock_level', 'current_stock',
        'balance', 'ending_inventory', 'beginning_inventory', 'on_hand_qty'
    ]
    
    @staticmethod
    def detect_date_column(df):
        """Auto-detect date column from various patterns"""
        # Check for exact matches first
        for pattern in DataProcessor.DATE_PATTERNS:
            matches = [col for col in df.columns if pattern in str(col).lower()]
            if matches:
                # Try to convert to datetime
                for col in matches:
                    try:
                        sample = df[col].dropna().head(100)
                        if len(sample) > 0:
                            pd.to_datetime(sample)
                            return col
                    except:
                        continue
        
        # Check by data type
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
            # Try converting first few rows
            try:
                sample = df[col].dropna().head(50)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    return col
            except:
                continue
        
        return None
    
    @staticmethod
    def detect_demand_column(df, exclude_cols=None):
        """Auto-detect demand/sales column"""
        if exclude_cols is None:
            exclude_cols = []
        
        # Check for exact matches
        for pattern in DataProcessor.DEMAND_PATTERNS:
            matches = [col for col in df.columns 
                      if pattern in str(col).lower() and col not in exclude_cols]
            if matches:
                # Prefer numeric columns
                for col in matches:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        return col
                # Return first match if no numeric found
                if matches:
                    return matches[0]
        
        # Check numeric columns that might be demand
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        if len(numeric_cols) > 0:
            # Prefer columns with positive values and reasonable range
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0 and (col_data >= 0).all():
                    return col
            return numeric_cols[0]
        
        return None
    
    @staticmethod
    def detect_inventory_column(df, exclude_cols=None):
        """Auto-detect inventory/stock column"""
        if exclude_cols is None:
            exclude_cols = []
        
        for pattern in DataProcessor.INVENTORY_PATTERNS:
            matches = [col for col in df.columns 
                      if pattern in str(col).lower() and col not in exclude_cols]
            if matches:
                for col in matches:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        return col
                if matches:
                    return matches[0]
        
        return None
    
    @staticmethod
    def _robust_parse_dates(series):
        """Attempt multiple strategies to parse dates and return best result"""
        s = series.copy()
        
        # Trim whitespace if object dtype
        if pd.api.types.is_object_dtype(s):
            try:
                s = s.astype(str).str.strip().replace({'': np.nan})
            except Exception:
                pass
        
        candidates = []
        
        # Strategy 1: default parser
        try:
            dt = pd.to_datetime(s, errors='coerce')
            candidates.append(dt)
        except Exception:
            candidates.append(pd.Series([pd.NaT] * len(s)))
        
        # Strategy 2: dayfirst
        try:
            dt = pd.to_datetime(s, errors='coerce', dayfirst=True)
            candidates.append(dt)
        except Exception:
            candidates.append(pd.Series([pd.NaT] * len(s)))
        
        # Strategy 3: common explicit formats
        common_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']
        for fmt in common_formats:
            try:
                dt = pd.to_datetime(s, errors='coerce', format=fmt)
                candidates.append(dt)
            except Exception:
                candidates.append(pd.Series([pd.NaT] * len(s)))
        
        # Strategy 4: Excel serial numbers
        try:
            if pd.api.types.is_numeric_dtype(s) or (pd.api.types.is_object_dtype(s) and s.astype(str).str.replace('.', '', regex=False).str.isnumeric().mean() > 0.5):
                numeric = pd.to_numeric(s, errors='coerce')
                # Heuristic range for Excel serials
                looks_like_excel = numeric.between(20000, 60000).mean() > 0.5
                if looks_like_excel:
                    base = pd.Timestamp('1899-12-30')
                    dt = base + pd.to_timedelta(numeric, unit='D')
                    candidates.append(dt)
        except Exception:
            candidates.append(pd.Series([pd.NaT] * len(s)))
        
        # Pick candidate with highest success rate
        best = max(candidates, key=lambda x: x.notna().mean() if isinstance(x, pd.Series) else -1)
        success_rate = best.notna().mean() if isinstance(best, pd.Series) else 0.0
        return best, success_rate
    
    @staticmethod
    def load_data(uploaded_file, date_col=None, demand_col=None, inventory_col=None):
        """Load data from uploaded file with flexible column detection"""
        try:
            # Read file with error handling for different encodings
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                return None, "Unsupported file format. Please use CSV or Excel files."
            
            if df.empty:
                return None, "The uploaded file is empty."
            
            # Store original column names
            original_columns = df.columns.tolist()
            
            # Auto-detect columns if not provided
            if date_col is None or date_col not in df.columns:
                date_col = DataProcessor.detect_date_column(df)
            
            if demand_col is None or demand_col not in df.columns:
                demand_col = DataProcessor.detect_demand_column(df, exclude_cols=[date_col] if date_col else [])
            
            if inventory_col is None or inventory_col not in df.columns:
                inventory_col = DataProcessor.detect_inventory_column(df, exclude_cols=[date_col, demand_col] if date_col and demand_col else [date_col] if date_col else [])
            
            # Validate we have at least a date column
            if date_col is None:
                return None, "Could not detect a date column. Please ensure your data has a date/time column."
            
            # Create processed dataframe
            df_processed = df.copy()
            
            # Convert and rename date column
            if date_col in df_processed.columns:
                parsed_dates, success = DataProcessor._robust_parse_dates(df_processed[date_col])
                df_processed['date'] = parsed_dates
                
                # If still poor success, provide clearer guidance but keep trying a relaxed threshold
                initial_len = len(df_processed)
                df_processed = df_processed.dropna(subset=['date'])
                
                # Require at least 80% success if there are enough rows, otherwise 50% for very small datasets
                min_success = 0.8 if initial_len >= 30 else 0.5
                if (len(df_processed) / max(initial_len, 1)) < min_success:
                    return None, f"Too many rows failed date conversion. Detected column '{date_col}' contains mixed or unrecognized formats. Try selecting the correct date column or reformatting dates (e.g., YYYY-MM-DD)."
            else:
                return None, f"Date column '{date_col}' not found in the data."
            
            # Handle demand/sales column
            if demand_col and demand_col in df_processed.columns:
                df_processed['sales'] = pd.to_numeric(df_processed[demand_col], errors='coerce')
            elif inventory_col and inventory_col in df_processed.columns:
                # Use inventory as proxy for demand if no demand column
                df_processed['sales'] = pd.to_numeric(df_processed[inventory_col], errors='coerce')
            else:
                # Try to find any numeric column
                numeric_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                               if col != date_col]
                if len(numeric_cols) > 0:
                    df_processed['sales'] = pd.to_numeric(df_processed[numeric_cols[0]], errors='coerce')
                else:
                    return None, "Could not detect a numeric column for sales/demand. Please ensure your data has numeric values."
            
            # Sort by date
            df_processed = df_processed.sort_values('date').reset_index(drop=True)
            
            # Handle missing values in sales column
            df_processed['sales'] = df_processed['sales'].ffill().bfill().fillna(0)
            
            # Ensure sales is non-negative
            df_processed['sales'] = df_processed['sales'].clip(lower=0)
            
            # Store column mapping info
            if 'column_mapping' not in st.session_state:
                st.session_state.column_mapping = {}
            
            st.session_state.column_mapping = {
                'date': date_col,
                'demand': demand_col if demand_col else inventory_col,
                'inventory': inventory_col,
                'original_columns': original_columns,
                'all_columns': df.columns.tolist(),
                'detected': {
                    'date_auto': date_col is not None,
                    'demand_auto': demand_col is not None,
                    'inventory_auto': inventory_col is not None
                }
            }
            
            # Keep all original columns in the dataframe for reference
            for col in original_columns:
                if col not in df_processed.columns or col not in ['date', 'sales']:
                    if col in df.columns:
                        df_processed[col] = df[col].values[:len(df_processed)]
            
            return df_processed, None
            
        except Exception as e:
            import traceback
            error_msg = f"Error loading data: {str(e)}"
            if 'column_mapping' in locals():
                error_msg += f"\nDetected columns - Date: {date_col}, Demand: {demand_col}, Inventory: {inventory_col}"
            return None, error_msg
    
    @staticmethod
    def prepare_features(df, target_col='sales', include_external=False):
        """Prepare features for modeling"""
        df = df.copy()
        
        # Ensure date column exists
        if 'date' not in df.columns:
            return df
        
        # Time-based features
        if pd.api.types.is_datetime64_any_dtype(df['date']):
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            try:
                df['weekofyear'] = df['date'].dt.isocalendar().week
            except:
                df['weekofyear'] = df['date'].dt.week
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Lag features (only if target column exists and is numeric)
        if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            for lag in [1, 7, 14, 30]:
                if len(df) > lag:
                    df[f'lag_{lag}'] = df[target_col].shift(lag)
            
            # Rolling statistics
            for window in [7, 14, 30]:
                if len(df) > window:
                    df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
                    df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
        
        # Don't drop all NaN rows - just ensure critical columns exist
        critical_cols = ['date']
        if target_col in df.columns:
            critical_cols.append(target_col)
        
        df = df.dropna(subset=critical_cols)
        
        return df

# [Include the DemandForecaster and InventoryOptimizer classes from the previous code]

# Page Components
def show_home_page():
    """Display the home page"""
    
    # Welcome Section
    st.markdown("""
    <div class="section-card">
        <h2 style="text-align: center; margin-bottom: 2rem;">Welcome to IntelliStock AI</h2>
        <p style="text-align: center; font-size: 1.1rem; color: black; margin-bottom: 3rem;">
            Your intelligent companion for inventory optimization and demand forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            create_feature_card(
                "Smart Forecasting",
                "Leverage AI models to predict future demand with high accuracy",
                "📈",
                "Start Forecasting"
            ),
            unsafe_allow_html=True
        )
        if st.button("Start Forecasting →", key="go_forecast", use_container_width=True):
            st.session_state.current_page = 'forecast'
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    
    with col2:
        st.markdown(
            create_feature_card(
                "Inventory Optimization",
                "Get intelligent recommendations for stock levels and reorder points",
                "📦",
                "Optimize Now"
            ),
            unsafe_allow_html=True
        )
        if st.button("Optimize Now →", key="go_inventory", use_container_width=True):
            st.session_state.current_page = 'inventory'
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    
    with col3:
        st.markdown(
            create_feature_card(
                "Cost Analytics",
                "Analyze and reduce inventory costs with data-driven insights",
                "💰",
                "View Analytics"
            ),
            unsafe_allow_html=True
        )
        if st.button("View Analytics →", key="go_analytics", use_container_width=True):
            st.session_state.current_page = 'analytics'
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    
    # Quick Stats (if data is loaded)
    if st.session_state.data is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-card">
            <h3 style="margin-bottom: 1.5rem;">📊 Quick Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_sales = st.session_state.data['sales'].mean()
            recent_mean = st.session_state.data['sales'].tail(7).mean()
            prev_mean = st.session_state.data['sales'].iloc[-14:-7].mean() if len(st.session_state.data) >= 14 else None
            if prev_mean is not None and pd.notna(prev_mean) and prev_mean > 0:
                avg_delta_pct = (recent_mean - prev_mean) / prev_mean * 100
                avg_delta = f"{abs(avg_delta_pct):.1f}% vs prior 7 days"
                avg_delta_type = "positive" if avg_delta_pct >= 0 else "negative"
            else:
                avg_delta = None
                avg_delta_type = "positive"
            st.markdown(
                create_metric_card(
                    "Average Daily Sales",
                    f"{avg_sales:,.0f}",
                    avg_delta,
                    avg_delta_type,
                    "📊"
                ),
                unsafe_allow_html=True
            )
        
        with col2:
            total_sales = st.session_state.data['sales'].sum()
            if len(st.session_state.data) >= 60:
                recent_total = st.session_state.data.tail(30)['sales'].sum()
                prev_total = st.session_state.data.iloc[-60:-30]['sales'].sum()
                if pd.notna(prev_total) and prev_total > 0:
                    total_delta_pct = (recent_total - prev_total) / prev_total * 100
                    total_delta = f"{abs(total_delta_pct):.1f}% last 30 days"
                    total_delta_type = "positive" if total_delta_pct >= 0 else "negative"
                else:
                    total_delta = None
                    total_delta_type = "positive"
            else:
                total_delta = None
                total_delta_type = "positive"
            st.markdown(
                create_metric_card(
                    "Total Sales",
                    f"{total_sales:,.0f}",
                    total_delta,
                    total_delta_type,
                    "💰"
                ),
                unsafe_allow_html=True
            )
        
        with col3:
            days = len(st.session_state.data)
            st.markdown(
                create_metric_card(
                    "Days Analyzed",
                    f"{days}",
                    None,
                    "positive",
                    "📅"
                ),
                unsafe_allow_html=True
            )
        
        with col4:
            if len(st.session_state.data) >= 14:
                recent_avg = st.session_state.data['sales'].tail(7).mean()
                previous_avg = st.session_state.data['sales'].iloc[-14:-7].mean()
                trend_delta_pct = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg is not None and pd.notna(previous_avg) and previous_avg > 0 else 0
            else:
                recent_avg = st.session_state.data['sales'].tail(7).mean()
                previous_avg = avg_sales
                trend_delta_pct = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg is not None and pd.notna(previous_avg) and previous_avg > 0 else 0
            trend_icon = "📈" if trend_delta_pct >= 0 else "📉"
            trend_strength = f"{abs(trend_delta_pct):.1f}% change"
            st.markdown(
                create_metric_card(
                    "Current Trend",
                    trend_icon,
                    trend_strength,
                    "positive" if trend_delta_pct >= 0 else "negative",
                    "📊"
                ),
                unsafe_allow_html=True
            )

def show_upload_page():
    """Display the data upload page with flexible column detection"""
    st.markdown("""
    <div class="section-card">
        <h2 style="color: white;">📤 Upload Your Data</h2>
        <p style="color: #718096;">Import any inventory dataset - we'll automatically detect columns and formats</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader with custom styling
        uploaded_file = st.file_uploader(
            "",
            type=['csv', 'xlsx', 'xls'],
            help="Upload any inventory dataset (CSV or Excel). We'll automatically detect date and quantity columns.",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # First, read the file to show column options
            try:
                if uploaded_file.name.endswith('.csv'):
                    try:
                        preview_df = pd.read_csv(uploaded_file, nrows=5, encoding='utf-8')
                    except UnicodeDecodeError:
                        preview_df = pd.read_csv(uploaded_file, nrows=5, encoding='latin-1')
                else:
                    preview_df = pd.read_excel(uploaded_file, nrows=5)
                
                # Auto-detect columns first
                detected_date = DataProcessor.detect_date_column(preview_df)
                detected_demand = DataProcessor.detect_demand_column(preview_df, exclude_cols=[detected_date] if detected_date else [])
                detected_inventory = DataProcessor.detect_inventory_column(preview_df, exclude_cols=[detected_date, detected_demand] if detected_date and detected_demand else [detected_date] if detected_date else [])
                
                # Show detected columns and allow manual override
                st.markdown('<h3 style="color: white;">🔍 Column Detection</h3>', unsafe_allow_html=True)
                if detected_date or detected_demand:
                    detection_msg = "✅ Auto-detected columns: "
                    if detected_date:
                        detection_msg += f"Date={detected_date} "
                    if detected_demand:
                        detection_msg += f"Demand={detected_demand} "
                    if detected_inventory:
                        detection_msg += f"Inventory={detected_inventory}"
                    st.success(detection_msg)
                else:
                    st.info("💡 Please manually select columns from your dataset.")
                
                col_date, col_demand, col_inv = st.columns(3)
                
                all_columns = preview_df.columns.tolist()
                
                with col_date:
                    # Find index of detected date column
                    date_index = 0
                    if detected_date and detected_date in all_columns:
                        date_index = all_columns.index(detected_date)
                    
                    date_col = st.selectbox(
                        "📅 Date Column",
                        options=all_columns,
                        index=date_index if all_columns else 0,
                        help="Select the column containing dates/timestamps"
                    )
                
                with col_demand:
                    # Find index of detected demand column
                    demand_options = ['Auto-detect'] + all_columns
                    demand_index = 0
                    if detected_demand and detected_demand in all_columns:
                        demand_index = all_columns.index(detected_demand) + 1
                    
                    demand_col = st.selectbox(
                        "📊 Demand/Quantity Column",
                        options=demand_options,
                        index=demand_index,
                        help="Select the column containing sales/demand/quantity (or auto-detect)"
                    )
                    if demand_col == 'Auto-detect':
                        demand_col = detected_demand  # Use auto-detected value
                
                with col_inv:
                    # Find index of detected inventory column
                    inv_options = ['None'] + all_columns
                    inv_index = 0
                    if detected_inventory and detected_inventory in all_columns:
                        inv_index = all_columns.index(detected_inventory) + 1
                    
                    inventory_col = st.selectbox(
                        "📦 Inventory Column (Optional)",
                        options=inv_options,
                        index=inv_index,
                        help="Select the column containing inventory/stock levels (optional)"
                    )
                    if inventory_col == 'None':
                        inventory_col = None
                
                if st.button("🔄 Process Data", type="primary", use_container_width=True):
                    with st.spinner("Processing your data..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        data, error = DataProcessor.load_data(
                            uploaded_file, 
                            date_col=date_col if date_col else None,
                            demand_col=demand_col if demand_col else None,
                            inventory_col=inventory_col if inventory_col else None
                        )
                        
                        if error:
                            st.error(f"❌ Error: {error}")
                            st.info("💡 Tip: Make sure your date column is in a recognizable format (YYYY-MM-DD, MM/DD/YYYY, etc.)")
                        else:
                            st.session_state.data = data
                            st.success("✅ Data uploaded and processed successfully!")
                            st.session_state.forecast_results = None
                            st.session_state.forecast_metrics = None
                            st.session_state.backtest_metrics = None
                            st.session_state.inventory_recommendations = None
                            st.session_state.analysis_complete = False
                            
                            # Show column mapping info
                            if 'column_mapping' in st.session_state:
                                mapping = st.session_state.column_mapping
                                st.info("""
                                **📋 Detected Column Mapping:**
                                
                                - **Date:** {}
                                - **Demand/Quantity:** {}
                                - **Inventory:** {}
                                """.format(
                                    mapping.get('date', 'N/A'),
                                    mapping.get('demand', 'N/A'),
                                    mapping.get('inventory', 'N/A (not detected)')
                                ))
                            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("💡 Please ensure your file is a valid CSV or Excel file.")
        
        # Show data preview and visualization if data is loaded
        if st.session_state.data is not None:
            data = st.session_state.data
            
            st.markdown("---")
            st.markdown('<h3 style="color: white;">📊 Data Overview</h3>', unsafe_allow_html=True)
            
            # Data statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Total Records", len(data))
            with col_stat2:
                if 'date' in data.columns:
                    date_range = (data['date'].max() - data['date'].min()).days
                    st.metric("Date Range", f"{date_range} days")
            with col_stat3:
                if 'sales' in data.columns:
                    st.metric("Avg Daily Value", f"{data['sales'].mean():.2f}")
            with col_stat4:
                if 'sales' in data.columns:
                    st.metric("Total Value", f"{data['sales'].sum():,.0f}")
            
            # Show data preview
            st.markdown('<h3 style="color: white;">📋 Data Preview</h3>', unsafe_allow_html=True)
            st.dataframe(
                data.head(20).style.set_properties(**{
                    'background-color': '#f8fafc',
                    'color': '#1a202c',
                    'border-color': '#e2e8f0'
                }),
                use_container_width=True,
                height=400
            )
            
            # Visualize the data if we have date and sales columns
            if 'date' in data.columns and 'sales' in data.columns:
                st.markdown('<h3 style="color: white;">📈 Trend Visualization</h3>', unsafe_allow_html=True)
                fig = px.line(
                    data,
                    x='date',
                    y='sales',
                    title='',
                    line_shape='spline',
                    labels={'date': 'Date', 'sales': 'Quantity/Demand'}
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(0,0,0,0.05)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(0,0,0,0.05)'
                    ),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                fig.update_traces(
                    line=dict(color='#667eea', width=3),
                    mode='lines'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Info card with updated requirements
        with st.container():
            st.markdown('<h3 style="color: white;">📋 Supported Formats</h3>', unsafe_allow_html=True)
            
            st.markdown("""
            **File Types:**
            - CSV files
            - Excel files (.xlsx, .xls)
            
            **Auto-Detected Columns:**
            - Date/Time columns
            - Sales/Demand/Quantity
            - Inventory/Stock levels
            
            **Supported Date Formats:**
            - YYYY-MM-DD
            - MM/DD/YYYY
            - DD-MM-YYYY
            - And many more!
            
            **Optional Columns:**
            - Promotion flags
            - Holiday indicators
            - External factors
            - Product categories
            """)
        
        st.markdown("---")
        
        # Sample data download
        st.markdown('<h3 style="color: white;">📥 Sample Data</h3>', unsafe_allow_html=True)
        
        sample_type = st.selectbox(
            "Sample Dataset Type",
            ["Sales Data", "Inventory Levels", "Order History"],
            label_visibility="collapsed"
        )
        
        if sample_type == "Sales Data":
            sample_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=365, freq='D'),
                'sales': np.random.normal(1000, 200, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 300
            })
            filename = 'sample_sales_data.csv'
        elif sample_type == "Inventory Levels":
            sample_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=365, freq='D'),
                'inventory_on_hand': np.random.normal(5000, 500, 365),
                'orders_received': np.random.poisson(100, 365)
            })
            filename = 'sample_inventory_data.csv'
        else:
            sample_data = pd.DataFrame({
                'order_date': pd.date_range('2023-01-01', periods=365, freq='D'),
                'quantity_ordered': np.random.poisson(50, 365),
                'order_value': np.random.normal(5000, 1000, 365)
            })
            filename = 'sample_orders_data.csv'
        
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {sample_type} Sample",
            data=csv,
            file_name=filename,
            mime='text/csv',
            use_container_width=True
        )

def show_forecast_page():
    """Display the forecasting page"""
    if st.session_state.data is None:
        st.markdown("""
        <div class="section-card" style="text-align: center;">
            <h3>📊 No Data Available</h3>
            <p style="color: #718096;">Please upload your data first to begin forecasting</p>
            <br>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Data Upload", use_container_width=True):
            st.session_state.current_page = 'upload'
        return
    
    st.markdown("""
    <div class="section-card">
        <h2>🔮 Demand Forecasting</h2>
        <p style="color: #718096;">Choose your preferred AI model for accurate demand predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection cards
    col1, col2, col3, col4 = st.columns(4)
    
    models = {
        'ARIMA': {'icon': '📈', 'desc': 'Time series classic', 'full_desc': 'ARIMA (AutoRegressive Integrated Moving Average) is a classic time series forecasting method that captures trends and patterns in historical data.'},
        'Prophet': {'icon': '🔮', 'desc': 'Facebook\'s algorithm', 'full_desc': 'Prophet is Facebook\'s robust forecasting tool that handles seasonality, holidays, and changepoints automatically.'},
        'XGBoost': {'icon': '🚀', 'desc': 'Machine learning power', 'full_desc': 'XGBoost is a powerful gradient boosting machine learning algorithm that can capture complex non-linear patterns.'},
        'Ensemble': {'icon': '🎯', 'desc': 'Combined intelligence', 'full_desc': 'Ensemble combines predictions from multiple models to provide the most accurate and robust forecasts.'}
    }
    
    # Handle model selection
    selected_model = st.session_state.selected_model
    
    # Add custom CSS for selected button styling
    if selected_model:
        st.markdown(f"""
        <style>
            button[data-testid="baseButton-secondary"][aria-label*="{selected_model}"] {{
                background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
                color: black !important;
                border: 3px solid #10b981 !important;
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5) !important;
                font-weight: 700 !important;
            }}
        </style>
        """, unsafe_allow_html=True)
    
    for idx, (model, info) in enumerate(models.items()):
        with [col1, col2, col3, col4][idx]:
            # Determine if this model is selected
            is_selected = (selected_model == model)
            button_label = f"{info['icon']} {model}\n{info['desc']}"
            
            # Add visual indicator for selected model in button text
            if is_selected:
                button_label = f"✅ {button_label}\n[SELECTED]"
            
            # Use primary button type for selected, secondary for others
            button_type = "primary" if is_selected else "secondary"
            
            if st.button(
                button_label,
                key=f"model_{model}",
                use_container_width=True,
                type=button_type
            ):
                st.session_state.selected_model = model
                selected_model = model
                st.rerun()
    
    # Display selected model indicator
    if st.session_state.selected_model:
        selected_info = models[st.session_state.selected_model]
        st.markdown(f"""
        <div class="model-selection-banner">
            <div class="model-selection-banner-icon">{selected_info['icon']}</div>
            <div class="model-selection-banner-text">
                <div class="model-selection-banner-title">Selected Model: {st.session_state.selected_model}</div>
                <div class="model-selection-banner-subtitle">{selected_info['full_desc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ **Please select a forecasting model above to proceed.** Each model has unique strengths for different types of data patterns.")
    
    # Forecast configuration
    with st.expander("⚙️ Advanced Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
            confidence_level = st.select_slider("Confidence Level", [0.90, 0.95, 0.99], 0.95)
        with col2:
            include_seasonality = st.checkbox("Include Seasonality", True)
            include_holidays = st.checkbox("Include Holiday Effects", False)
    
    # Run forecast button
    if st.session_state.selected_model:
        forecast_button_label = f"🚀 Generate Forecast using {st.session_state.selected_model}"
    else:
        forecast_button_label = "🚀 Generate Forecast (Please select a model first)"
    
    if st.button(forecast_button_label, use_container_width=True, disabled=(st.session_state.selected_model is None)):
        if st.session_state.selected_model is None:
            st.warning("⚠️ Please select a forecasting model first!")
        else:
            try:
                with st.spinner(f"Running {st.session_state.selected_model} model - analyzing your data..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text(f"Initializing {st.session_state.selected_model} model...")
                    for i in range(30):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)
                    
                    status_text.text(f"Training {st.session_state.selected_model} model...")
                    for i in range(30, 70):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)
                    
                    status_text.text(f"Generating forecasts with {st.session_state.selected_model}...")
                    for i in range(70, 100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.005)
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    forecaster = DemandForecaster(st.session_state.data)
                    forecast_output = forecaster.forecast(
                        model_name=st.session_state.selected_model,
                        horizon=forecast_horizon,
                        confidence_level=confidence_level,
                        include_seasonality=include_seasonality,
                        include_holidays=include_holidays
                    )
                
                st.session_state.forecast_results = forecast_output.get('forecast')
                st.session_state.forecast_metrics = forecast_output.get('metrics')
                st.session_state.backtest_metrics = forecast_output.get('backtest')
                st.session_state.forecast_config = {
                    'horizon': forecast_horizon,
                    'confidence_level': confidence_level,
                    'include_seasonality': include_seasonality,
                    'include_holidays': include_holidays,
                    'model': st.session_state.selected_model
                }
                st.session_state.analysis_complete = True
                st.success(f"✅ Forecasting completed successfully using {st.session_state.selected_model}!")
            except Exception as err:
                st.error(f"Forecasting failed: {err}")
                st.stop()

    forecast_data = st.session_state.get('forecast_results')
    forecast_metrics = st.session_state.get('forecast_metrics')
    
    if forecast_data is not None and forecast_metrics is not None:
        if not isinstance(forecast_data, pd.DataFrame):
            forecast_df = pd.DataFrame(forecast_data)
        else:
            forecast_df = forecast_data.copy()
        
        if 'date' in forecast_df.columns:
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        forecast_df = forecast_df.sort_values('date')
        
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.data['date'],
            y=st.session_state.data['sales'],
            name='Historical',
            line=dict(color='#4a5568', width=2),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            name='Forecast',
            line=dict(color='#667eea', width=3),
            mode='lines+markers',
            marker=dict(size=4)
        ))
        if {'lower_bound', 'upper_bound'}.issubset(forecast_df.columns):
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='Confidence'
            ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.05)',
                title=''
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.05)',
                title='Sales'
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        if st.session_state.selected_model:
            st.markdown(f"""
            <div class="info-card" style="margin-bottom: 1.5rem;">
                <h4>🤖 Model Used: {st.session_state.selected_model}</h4>
                <p>This forecast leverages <strong>{st.session_state.selected_model}</strong>. {models[st.session_state.selected_model]['full_desc']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">
                <div class="chart-title">AI-Powered Demand Forecast</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            model_icon = models.get(st.session_state.selected_model, {}).get('icon', '🎯') if st.session_state.selected_model else '🎯'
            st.markdown(
                create_metric_card(
                    "Forecast Model",
                    st.session_state.selected_model if st.session_state.selected_model else "N/A",
                    delta=f"Horizon: {st.session_state.get('forecast_config', {}).get('horizon', len(forecast_df))} days",
                    delta_type="positive",
                    icon=model_icon
                ),
                unsafe_allow_html=True
            )
        with col2:
            mape_delta = f"{forecast_metrics.get('mape_improvement', 0):.1f} pts vs naive" if forecast_metrics.get('mape_improvement') is not None else None
            mape_delta_type = "positive" if forecast_metrics.get('mape_improvement', 0) >= 0 else "negative"
            st.markdown(
                create_metric_card(
                    "MAPE",
                    f"{forecast_metrics.get('ai_mape', 0):.1f}%",
                    delta=mape_delta,
                    delta_type=mape_delta_type,
                    icon="🎯"
                ),
                unsafe_allow_html=True
            )
        with col3:
            wape_delta = f"{forecast_metrics.get('wape_improvement', 0):.1f} pts vs naive" if forecast_metrics.get('wape_improvement') is not None else None
            wape_delta_type = "positive" if forecast_metrics.get('wape_improvement', 0) >= 0 else "negative"
            st.markdown(
                create_metric_card(
                    "WAPE",
                    f"{forecast_metrics.get('ai_wape', 0):.1f}%",
                    delta=wape_delta,
                    delta_type=wape_delta_type,
                    icon="📉"
                ),
                unsafe_allow_html=True
            )
        with col4:
            peak_idx = forecast_df['forecast'].idxmax()
            peak_date = forecast_df.loc[peak_idx, 'date'] if peak_idx is not None else None
            peak_text = peak_date.strftime('%A') if isinstance(peak_date, pd.Timestamp) else "—"
            st.markdown(
                create_metric_card(
                    "Peak Demand",
                    f"{forecast_df['forecast'].max():,.0f}",
                    delta=peak_text,
                    delta_type="positive",
                    icon="🔝"
                ),
                unsafe_allow_html=True
            )

def show_inventory_page():
    """Display the inventory optimization page"""
    
    if st.session_state.data is None:
        st.markdown("""
        <div class="section-card" style="text-align: center;">
            <h3>📦 No Data Available</h3>
            <p style="color: #718096;">Please upload your data first to optimize inventory</p>
            <br>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Data Upload", use_container_width=True):
            st.session_state.current_page = 'upload'
        return
    
    st.markdown("""
    <div class="section-card">
        <h2>📦 Smart Inventory Optimization</h2>
        <p style="color: #718096;">Get AI-driven recommendations for optimal stock levels</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Current inventory input with beautiful design
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3 style="margin-bottom: 1.5rem;">Current Status</h3>
        """, unsafe_allow_html=True)
        
        current_inventory = st.number_input(
            "Current Inventory Level",
            min_value=0,
            value=1000,
            step=100,
            help="Enter your current stock quantity"
        )
        
        lead_time = st.number_input(
            "Lead Time (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="Average time to receive new inventory"
        )
        
        service_level = st.select_slider(
            "Target Service Level",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{int(x*100)}%"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3 style="margin-bottom: 1.5rem;">Cost Parameters</h3>
        """, unsafe_allow_html=True)
        
        holding_cost = st.number_input(
            "Holding Cost (per unit/day)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.2f"
        )
        
        stockout_cost = st.number_input(
            "Stockout Cost (per unit)",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            format="%.2f"
        )
        
        ordering_cost = st.number_input(
            "Ordering Cost (per order)",
            min_value=10.0,
            max_value=500.0,
            value=50.0,
            step=10.0,
            format="%.2f"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Optimize button
    if st.button("🎯 Optimize Inventory", use_container_width=True):
        with st.spinner("AI is calculating optimal inventory levels..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            # Calculate recommendations (simplified for demo)
            avg_demand = st.session_state.data['sales'].mean()
            std_demand = st.session_state.data['sales'].std()
            
            # Safety stock calculation
            z_score = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}[service_level]
            safety_stock = z_score * std_demand * np.sqrt(lead_time)
            
            # Reorder point
            reorder_point = (avg_demand * lead_time) + safety_stock
            
            # Economic Order Quantity
            eoq = np.sqrt((2 * avg_demand * ordering_cost) / holding_cost)
            
            recommended_max_inventory = reorder_point + safety_stock
            stockout_risk = 0.0
            if reorder_point > 0:
                stockout_risk = max((reorder_point - current_inventory) / reorder_point * 100, 0)
            overstock_gap_units = max(current_inventory - recommended_max_inventory, 0)
            days_of_stock = current_inventory / avg_demand if avg_demand > 0 else None
            
            st.session_state.inventory_recommendations = {
                'avg_daily_demand': float(avg_demand),
                'safety_stock': float(safety_stock),
                'reorder_point': float(reorder_point),
                'economic_order_quantity': float(eoq),
                'current_inventory': float(current_inventory),
                'recommended_max_inventory': float(recommended_max_inventory),
                'stockout_risk': float(stockout_risk),
                'overstock_gap_units': float(overstock_gap_units),
                'service_level': float(service_level * 100),
                'lead_time': int(lead_time),
                'holding_cost': float(holding_cost),
                'ordering_cost': float(ordering_cost),
                'stockout_cost': float(stockout_cost),
                'days_of_stock': float(days_of_stock) if days_of_stock is not None else None
            }
            
            st.success("✅ Optimization complete!")
            
            # Display results in beautiful cards
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="section-card">
                <h3 style="margin-bottom: 1.5rem;">🎯 Optimization Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    create_metric_card(
                        "Safety Stock",
                        f"{safety_stock:,.0f}",
                        "units",
                        "positive",
                        "🛡️"
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    create_metric_card(
                        "Reorder Point",
                        f"{reorder_point:,.0f}",
                        "units",
                        "positive",
                        "🔄"
                    ),
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    create_metric_card(
                        "Optimal Order Qty",
                        f"{eoq:,.0f}",
                        "units",
                        "positive",
                        "📦"
                    ),
                    unsafe_allow_html=True
                )
            
            with col4:
                days_of_stock = current_inventory / avg_demand if avg_demand > 0 else 0
                st.markdown(
                    create_metric_card(
                        "Days of Stock",
                        f"{days_of_stock:.0f}",
                        "days",
                        "positive" if days_of_stock > lead_time else "negative",
                        "📅"
                    ),
                    unsafe_allow_html=True
                )
            
            # Visual representation
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Create inventory level visualization
            fig = go.Figure()
            
            # Current level
            fig.add_trace(go.Bar(
                x=['Current'],
                y=[current_inventory],
                name='Current Inventory',
                marker_color='#4a5568'
            ))
            
            # Recommended levels
            fig.add_trace(go.Bar(
                x=['Safety Stock', 'Reorder Point', 'Order Quantity'],
                y=[safety_stock, reorder_point, eoq],
                name='Recommended',
                marker_color='#667eea'
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                yaxis_title='Units',
                xaxis_title='',
                barmode='group',
                height=400
            )
            
            st.markdown("""
            <div class="chart-container">
                <div class="chart-header">
                    <div class="chart-title">Inventory Level Comparison</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Action recommendations
            st.markdown("""
            <div class="section-card">
                <h3 style="margin-bottom: 1rem;">📋 Action Recommendations</h3>
            """, unsafe_allow_html=True)
            
            if current_inventory < reorder_point:
                st.error(f"""
                🚨 **Immediate Action Required**
                - Your inventory is below the reorder point
                - Place an order for {eoq:,.0f} units immediately
                - Consider expedited shipping to avoid stockouts
                """)
            elif current_inventory < reorder_point * 1.2:
                st.warning(f"""
                ⚠️ **Prepare to Reorder**
                - Inventory approaching reorder point
                - Prepare to order {eoq:,.0f} units soon
                - Monitor daily sales closely
                """)
            else:
                st.success(f"""
                ✅ **Inventory Levels Optimal**
                - Current stock levels are healthy
                - Next order: {eoq:,.0f} units when reaching {reorder_point:,.0f}
                - Continue monitoring trends
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)

def show_analytics_page():
    """Display the analytics page"""
    
    if st.session_state.data is None:
        st.markdown("""
        <div class="section-card" style="text-align: center;">
            <h3>💰 No Data Available</h3>
            <p style="color: #718096;">Please upload your data first to view analytics</p>
            <br>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Data Upload", use_container_width=True):
            st.session_state.current_page = 'upload'
        return
    
    st.markdown("""
    <div class="section-card">
        <h2>💰 Cost Analytics & Insights</h2>
        <p style="color: #718096;">Understand your inventory costs and discover savings opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data
    data = st.session_state.data
    inventory_metrics = st.session_state.inventory_recommendations
    
    # Cost parameters input section
    st.markdown("### 💲 Cost Parameters")
  


    st.info("💡 Enter your cost parameters to calculate accurate inventory costs based on your data.")
    
    cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
    
    with cost_col1:
        item_cost = st.number_input(
            "Item Cost ($)",
            min_value=0.01,
            value=10.0,
            step=0.1,
            help="Cost per unit of inventory item"
        )
    
    with cost_col2:
        holding_cost_rate = st.number_input(
            "Holding Cost Rate (%/year)",
            min_value=0.1,
            max_value=50.0,
            value=20.0,
            step=0.5,
            help="Annual holding cost as percentage of item value"
        ) / 100
    
    with cost_col3:
        ordering_cost = st.number_input(
            "Ordering Cost ($/order)",
            min_value=1.0,
            value=50.0,
            step=5.0,
            help="Fixed cost per order placed"
        )
    
    with cost_col4:
        stockout_cost = st.number_input(
            "Stockout Cost ($/unit)",
            min_value=1.0,
            value=25.0,
            step=1.0,
            help="Cost per unit when stockout occurs"
        )
    
    # Calculate costs based on actual data
    if 'sales' in data.columns:
        # Calculate from actual data
        avg_daily_demand = data['sales'].mean()
        total_demand = data['sales'].sum()
        days_of_data = len(data)
        
        # Current inventory level (use from inventory recommendations if available, otherwise estimate)
        if inventory_metrics and 'current_inventory' in inventory_metrics:
            current_inventory = inventory_metrics['current_inventory']
        else:
            # Estimate based on average demand and a buffer
            current_inventory = avg_daily_demand * 30  # Assume 30 days of stock
        
        # Inventory value
        current_inventory_value = current_inventory * item_cost
        
        # Monthly holding cost
        monthly_holding_cost = (current_inventory_value * holding_cost_rate) / 12
        
        # Ordering costs (estimate based on demand frequency)
        # Assume orders are placed when inventory reaches reorder point
        if inventory_metrics and 'economic_order_quantity' in inventory_metrics:
            eoq = inventory_metrics['economic_order_quantity']
            orders_per_year = (avg_daily_demand * 365) / eoq if eoq > 0 else 12
        else:
            # Estimate: order monthly
            orders_per_year = 12
        
        monthly_ordering_cost = (orders_per_year * ordering_cost) / 12
        
        # Stockout losses (estimate based on demand variability)
        if inventory_metrics and 'stockout_risk' in inventory_metrics:
            stockout_probability = inventory_metrics['stockout_risk'] / 100
        else:
            # Estimate 5% stockout risk if no metrics available
            stockout_probability = 0.05
        
        # Estimate monthly stockout cost
        monthly_demand = avg_daily_demand * 30
        monthly_stockout_losses = monthly_demand * stockout_probability * stockout_cost
        
        # Calculate trend from actual data
        if len(data) >= 30:
            # Calculate monthly trends from data
            data['month'] = data['date'].dt.to_period('M')
            monthly_data = data.groupby('month')['sales'].sum().reset_index()
            monthly_data['month'] = monthly_data['month'].astype(str)
            
            # Calculate cost trend
            monthly_costs = []
            for month_sales in monthly_data['sales']:
                month_inv = month_sales * 1.5  # Assume inventory is 1.5x monthly sales
                month_holding = (month_inv * item_cost * holding_cost_rate) / 12
                # Estimate orders: assume orders are placed based on demand
                days_in_month = 30
                month_daily_avg = month_sales / days_in_month if days_in_month > 0 else avg_daily_demand
                if inventory_metrics and 'economic_order_quantity' in inventory_metrics and inventory_metrics['economic_order_quantity'] > 0:
                    month_orders = (month_sales / inventory_metrics['economic_order_quantity'])
                else:
                    month_orders = max(1, month_sales / (avg_daily_demand * 30)) if avg_daily_demand > 0 else 1
                month_ordering = month_orders * ordering_cost
                month_stockout = month_sales * stockout_probability * stockout_cost
                monthly_costs.append(month_holding + month_ordering + month_stockout)
        else:
            # Use simulated trend if not enough data
            monthly_costs = None
    else:
        # Fallback to sample data if no sales column
        current_inventory = 5000
        current_inventory_value = 50000
        monthly_holding_cost = 2500
        monthly_ordering_cost = 1500
        monthly_stockout_losses = 3000
        orders_per_year = 12
        stockout_probability = 0.05
        avg_daily_demand = 100
        monthly_costs = None
        st.warning("⚠️ Unable to calculate costs from data. Using sample values.")
    
    # Cost overview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            create_metric_card(
                "Inventory Value",
                f"${current_inventory_value:,.0f}",
                f"{current_inventory:,.0f} units",
                "positive",
                "💵"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        annual_holding = monthly_holding_cost * 12
        st.markdown(
            create_metric_card(
                "Monthly Holding Cost",
                f"${monthly_holding_cost:,.0f}",
                f"${annual_holding:,.0f}/year",
                "negative",
                "📦"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        annual_ordering = monthly_ordering_cost * 12
        st.markdown(
            create_metric_card(
                "Monthly Ordering Cost",
                f"${monthly_ordering_cost:,.0f}",
                f"{orders_per_year:.1f} orders/year",
                "positive",
                "🚚"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        annual_stockout = monthly_stockout_losses * 12
        stockout_risk_pct = stockout_probability * 100
        st.markdown(
            create_metric_card(
                "Monthly Stockout Cost",
                f"${monthly_stockout_losses:,.0f}",
                f"{stockout_risk_pct:.1f}% risk",
                "negative",
                "⚠️"
            ),
            unsafe_allow_html=True
        )
    
    # Cost breakdown visualization
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Pie chart for cost distribution
        total_monthly_cost = monthly_holding_cost + monthly_ordering_cost + monthly_stockout_losses
        cost_data = pd.DataFrame({
            'Category': ['Holding', 'Ordering', 'Stockouts'],
            'Amount': [monthly_holding_cost, monthly_ordering_cost, monthly_stockout_losses]
        })
        
        fig = px.pie(
            cost_data,
            values='Amount',
            names='Category',
            color_discrete_map={
                'Holding': '#667eea',
                'Ordering': '#764ba2',
                'Stockouts': '#f59e0b'
            },
            hole=0.4
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='%{label}: $%{value:,.0f}<extra></extra>'
        )
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=300
        )
        
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">
                <div class="chart-title">Cost Distribution</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cost trend chart based on actual data
        if monthly_costs and len(monthly_costs) > 0:
            # Use actual monthly data
            trend_data = pd.DataFrame({
                'Month': monthly_data['month'],
                'Total Cost': monthly_costs
            })
            chart_title = f"Cost Trend ({len(monthly_costs)} Months)"
        else:
            # Fallback: show estimated trend
            if 'date' in data.columns and len(data) > 0:
                # Create monthly trend from available data
                data_copy = data.copy()
                data_copy['month'] = data_copy['date'].dt.to_period('M')
                monthly_sales = data_copy.groupby('month')['sales'].sum().reset_index()
                
                # Estimate costs for each month
                monthly_costs_est = []
                for sales in monthly_sales['sales']:
                    month_inv = sales * 1.5
                    month_holding = (month_inv * item_cost * holding_cost_rate) / 12
                    month_ordering = ordering_cost * (sales / avg_daily_demand / 30) if avg_daily_demand > 0 else ordering_cost
                    month_stockout = sales * stockout_probability * stockout_cost
                    monthly_costs_est.append(month_holding + month_ordering + month_stockout)
                
                trend_data = pd.DataFrame({
                    'Month': monthly_sales['month'].astype(str),
                    'Total Cost': monthly_costs_est
                })
                chart_title = f"Estimated Cost Trend ({len(monthly_costs_est)} Months)"
            else:
                # Last resort: show current month only
                trend_data = pd.DataFrame({
                    'Month': [datetime.now().strftime('%Y-%m')],
                    'Total Cost': [total_monthly_cost]
                })
                chart_title = "Current Month Cost"
        
        fig = px.area(
            trend_data,
            x='Month',
            y='Total Cost',
            line_shape='spline',
            labels={'Total Cost': 'Total Cost ($)', 'Month': 'Month'}
        )
        
        fig.update_traces(
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='#667eea', width=3)
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        )
        
        st.markdown(f"""
        <div class="chart-container">
            <div class="chart-header">
                <div class="chart-title">{chart_title}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Calculate savings opportunities based on actual data
    st.markdown("""
    <div class="section-card">
        <h3 style="margin-bottom: 1.5rem;">💡 Savings Opportunities</h3>
    """, unsafe_allow_html=True)
    
    # Calculate potential savings based on actual data
    if 'sales' in data.columns and avg_daily_demand > 0:
        # Optimize order frequency savings (using EOQ formula)
        try:
            # Calculate optimal EOQ
            annual_demand = avg_daily_demand * 365
            annual_holding_cost_per_unit = item_cost * holding_cost_rate
            if ordering_cost > 0 and annual_holding_cost_per_unit > 0:
                optimal_eoq = np.sqrt((2 * annual_demand * ordering_cost) / annual_holding_cost_per_unit)
                optimal_orders_per_year = annual_demand / optimal_eoq if optimal_eoq > 0 else orders_per_year
                
                # Calculate savings from optimizing order frequency
                if optimal_orders_per_year > 0 and optimal_orders_per_year < orders_per_year:
                    order_freq_savings = (orders_per_year - optimal_orders_per_year) * ordering_cost / 12
                    order_freq_savings = max(0, order_freq_savings)
                else:
                    # Estimate 20% savings from better ordering
                    order_freq_savings = monthly_ordering_cost * 0.2
            else:
                order_freq_savings = monthly_ordering_cost * 0.2
        except:
            order_freq_savings = monthly_ordering_cost * 0.2
        
        # Reduce safety stock savings (if inventory metrics available)
        if inventory_metrics and isinstance(inventory_metrics, dict) and 'safety_stock' in inventory_metrics:
            safety_stock_reduction = inventory_metrics['safety_stock'] * 0.2  # 20% reduction
            safety_stock_savings = (safety_stock_reduction * item_cost * holding_cost_rate) / 12
        else:
            safety_stock_savings = monthly_holding_cost * 0.15  # 15% savings estimate
        
        # Prevent stockouts savings
        if stockout_probability > 0.05:  # If stockout risk is high
            stockout_reduction = monthly_stockout_losses * 0.5  # 50% reduction possible
        else:
            stockout_reduction = monthly_stockout_losses * 0.2  # 20% reduction
        
        savings_opportunities = [
            {
                'title': 'Optimize Order Frequency',
                'savings': f'${order_freq_savings:,.0f}/month',
                'difficulty': 'Easy',
                'impact': 'High' if order_freq_savings > monthly_ordering_cost * 0.3 else 'Medium',
                'description': f'Reduce ordering costs by optimizing order frequency. Potential annual savings: ${order_freq_savings * 12:,.0f}'
            },
            {
                'title': 'Optimize Safety Stock',
                'savings': f'${safety_stock_savings:,.0f}/month',
                'difficulty': 'Medium',
                'impact': 'High' if safety_stock_savings > monthly_holding_cost * 0.2 else 'Medium',
                'description': f'Lower holding costs with improved demand forecasting. Potential annual savings: ${safety_stock_savings * 12:,.0f}'
            },
            {
                'title': 'Reduce Stockout Risk',
                'savings': f'${stockout_reduction:,.0f}/month',
                'difficulty': 'Easy',
                'impact': 'High' if stockout_probability > 0.1 else 'Medium',
                'description': f'Improve inventory management to reduce stockout losses. Potential annual savings: ${stockout_reduction * 12:,.0f}'
            }
        ]
    else:
        # Fallback to sample savings opportunities
        savings_opportunities = [
            {
                'title': 'Optimize Order Frequency',
                'savings': '$3,200/month',
                'difficulty': 'Easy',
                'impact': 'High',
                'description': 'Reduce ordering costs by 40% through batch optimization'
            },
            {
                'title': 'Reduce Safety Stock',
                'savings': '$1,800/month',
                'difficulty': 'Medium',
                'impact': 'Medium',
                'description': 'Lower holding costs with improved demand forecasting'
            },
            {
                'title': 'Prevent Stockouts',
                'savings': '$2,500/month',
                'difficulty': 'Easy',
                'impact': 'High',
                'description': 'Eliminate lost sales with better reorder timing'
            }
        ]
    for opp in savings_opportunities:
                impact_color = {'High': '#10b981', 'Medium': '#f59e0b', 'Low': '#ef4444'}[opp['impact']]

            # Difficulty color-coding
                diff = opp['difficulty'].lower()
                if 'easy' in diff:
                    diff_color = "#047857"      # green text
                    diff_bg = "#d1fae5"         # soft green background
                elif 'medium' in diff:
                    diff_color = "#b45309"      # amber text
                    diff_bg = "#fef3c7"         # pale amber background
                elif 'hard' in diff:
                    diff_color = "#b91c1c"      # red text
                    diff_bg = "#fee2e2"         # light red background
                else:
                    diff_color = "#1a202c"      # fallback dark gray
                    diff_bg = "#e5e7eb"         # neutral bg

                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid {impact_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #1a202c;">{opp['title']}</h4>
                        <span style="font-size: 1.25rem; font-weight: 700; color: #10b981;">{opp['savings']}</span>
                    </div>
                    <p style="color: #718096; margin: 0.5rem 0;">{opp['description']}</p>
                    <div style="display: flex; gap: 1rem; margin-top: 0.75rem;">
                        <span style="background: {diff_bg}; color: {diff_color}; padding: 0.25rem 0.75rem;
                                    border-radius: 20px; font-size: 0.875rem; font-weight: 600;">
                            Difficulty: {opp['difficulty']}
                        </span>
                        <span style="background: {impact_color}22; color: {impact_color};
                                    padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem;
                                    font-weight: 600;">
                            Impact: {opp['impact']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # for opp in savings_opportunities:
    #     impact_color = {'High': '#10b981', 'Medium': '#f59e0b', 'Low': '#ef4444'}[opp['impact']]
        
    #     st.markdown(f"""
    #     <div style="background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid {impact_color};">
    #         <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
    #             <h4 style="margin: 0; color: #1a202c;">{opp['title']}</h4>
    #             <span style="font-size: 1.25rem; font-weight: 700; color: #10b981;">{opp['savings']}</span>
    #         </div>
    #         <p style="color: #718096; margin: 0.5rem 0;">{opp['description']}</p>
    #         <div style="display: flex; gap: 1rem; margin-top: 0.75rem;">
    #             <span style="background: #f3f4f6; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem;">
    #                 Difficulty: {opp['difficulty']}
    #             </span>
    #             <span style="background: {impact_color}22; color: {impact_color}; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem;">
    #                 Impact: {opp['impact']}
    #             </span>
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # st.markdown("</div>", unsafe_allow_html=True)


def show_boardroom_page():
    """Display an executive-ready dashboard answering key boardroom questions"""
    st.markdown("""
    <div class="section-card">
        <h2>🏢 Boardroom Command Center</h2>
        <p style="color: #718096;">
            Executive-level storyline that maps AI capabilities to your supply chain strategy. Each tab aligns to the questions we must answer for the C-suite.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    data = st.session_state.data
    column_mapping = st.session_state.get('column_mapping', {})
    inventory_metrics = st.session_state.inventory_recommendations
    inventory_col = column_mapping.get('inventory') if column_mapping else None
    
    if data is None:
        st.info("Upload data first on the Data Upload page to activate the Boardroom insights.")
        return
    
    backtest_metrics = compute_backtest_metrics(data)
    challenge_insights = analyze_inventory_challenges(
        data,
        inventory_col=inventory_col,
        inventory_metrics=inventory_metrics
    )
    data_sources = evaluate_data_sources(data, column_mapping)
    
    tabs = st.tabs([
        "Inventory Challenges",
        "Predictive Accuracy",
        "Critical Data Sources",
        "Best-Fit AI Techniques",
        "MAPE & WAPE",
        "Real-time Visibility",
        "Sustainability & Cost"
    ])
    
    with tabs[0]:
        st.markdown("#### What inventory challenges can AI solve?")
        
        overstock_value = (
            f"{challenge_insights['overstock_days']} days"
            if challenge_insights['overstock_days'] is not None else "Monitor"
        )
        stockout_value = (
            f"{challenge_insights['stockout_days']} days"
            if challenge_insights['stockout_days'] is not None else "Monitor"
        )
        volatility_value = (
            f"{challenge_insights['demand_volatility']:.1f}%"
            if challenge_insights['demand_volatility'] is not None else "—"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            delta = None
            delta_type = "positive"
            if inventory_metrics and inventory_metrics.get('stockout_risk') is not None:
                delta = f"{inventory_metrics['stockout_risk']:.1f}% risk"
                delta_type = "negative" if inventory_metrics['stockout_risk'] > 15 else "positive"
            st.markdown(
                create_metric_card(
                    "Stockout Signals",
                    stockout_value,
                    delta=delta,
                    delta_type=delta_type,
                    icon="🚨"
                ),
                unsafe_allow_html=True
            )
        with col2:
            overstock_delta = None
            delta_type = "negative"
            if inventory_metrics and inventory_metrics.get('overstock_gap_units') is not None:
                gap_units = inventory_metrics['overstock_gap_units']
                if gap_units > 0:
                    overstock_delta = f"{gap_units:,.0f} excess units"
                    delta_type = "negative"
                else:
                    overstock_delta = "Optimized"
                    delta_type = "positive"
            st.markdown(
                create_metric_card(
                    "Overstock Exposure",
                    overstock_value,
                    delta=overstock_delta,
                    delta_type=delta_type,
                    icon="📦"
                ),
                unsafe_allow_html=True
            )
        with col3:
            peak_month = challenge_insights.get('seasonality_peak')
            trough_month = challenge_insights.get('seasonality_trough')
            seasonality_note = None
            if peak_month and trough_month:
                seasonality_note = f"Peak: {calendar.month_name[int(peak_month)]}, Low: {calendar.month_name[int(trough_month)]}"
            st.markdown(
                create_metric_card(
                    "Demand Volatility",
                    volatility_value,
                    delta=seasonality_note,
                    delta_type="positive",
                    icon="🌐"
                ),
                unsafe_allow_html=True
            )
        
        st.markdown("""
- Detect stockouts before they happen by monitoring low stock days and lead-time coverage.
- Flag costly overstock by benchmarking on-hand units against AI-calculated reorder and max thresholds.
- Quantify seasonal swings so planners can dial up or down safety stock ahead of peak periods.
        """)
    
    with tabs[1]:
        st.markdown("#### How does predictive analytics improve forecasting accuracy?")
        if backtest_metrics:
            improvement = backtest_metrics['mape_improvement']
            improvement_wape = backtest_metrics['wape_improvement']
            mape_delta_type = "positive" if improvement > 0 else "negative"
            wape_delta_type = "positive" if improvement_wape > 0 else "negative"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    create_metric_card(
                        "AI Forecast MAPE",
                        f"{backtest_metrics['ai_mape']:.1f}%",
                        delta=f"{improvement:.1f} pts vs naive",
                        delta_type=mape_delta_type,
                        icon="🎯"
                    ),
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    create_metric_card(
                        "AI Forecast WAPE",
                        f"{backtest_metrics['ai_wape']:.1f}%",
                        delta=f"{improvement_wape:.1f} pts improvement",
                        delta_type=wape_delta_type,
                        icon="📉"
                    ),
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    create_metric_card(
                        "Backtest Window",
                        f"{backtest_metrics['test_days']} days",
                        delta="Rolling hold-out sample",
                        delta_type="positive",
                        icon="⏱️"
                    ),
                    unsafe_allow_html=True
                )
            
            fig = go.Figure()
            test_data = backtest_metrics['test_actuals']
            fig.add_trace(go.Scatter(
                x=test_data['date'],
                y=test_data['sales'],
                name='Actuals',
                line=dict(color='#4a5568', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=test_data['date'],
                y=backtest_metrics['test_forecast'],
                name='AI Trend Forecast',
                line=dict(color='#667eea', width=3, dash='dot')
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 60 days of history to evidence accuracy gains. Upload more data to unlock the backtest view.")
        
        st.markdown("""
- Rolling backtests show tangible lift versus the naive “last value” approach.
- Executive KPI: Highlight the accuracy gain (MAPE/WAPE delta) to quantify ROI from predictive analytics.
- Use the chart to brief leaders on forecast confidence bands and planning readiness.
        """)
    
    with tabs[2]:
        st.markdown("#### What data sources are essential?")
        source_df = pd.DataFrame(data_sources)
        st.dataframe(
            source_df[['name', 'status', 'detail']],
            use_container_width=True,
            hide_index=True
        )
        st.markdown("""
- Confirm baseline datasets: sales history is mandatory, inventory is highly recommended.
- Layer promotional, pricing, and external signals (weather, macro, regional events) for further accuracy lift.
- Highlight gaps to procurement/marketing so teams can align on data-sharing roadmaps.
        """)
    
    with tabs[3]:
        st.markdown("#### Which AI techniques should we deploy?")
        technique_cards = [
            ("ARIMA", "Great for stable, stationary demand – quick to deploy for single-SKU forecasting.", "📈"),
            ("Prophet", "Captures trend shifts, seasonality, and holiday effects with minimal tuning.", "🔮"),
            ("XGBoost", "Machine learning workhorse that fuses internal and external drivers for complex demand.", "🚀"),
            ("Ensemble", "Blends statistical + ML models to hedge risk and deliver robust forecasts.", "🎯"),
        ]
        col1, col2 = st.columns(2)
        for idx, (name, desc, icon) in enumerate(technique_cards):
            container = col1 if idx % 2 == 0 else col2
            with container:
                st.markdown(
                    create_feature_card(
                        name,
                        desc,
                        icon,
                        action_text="Details"
                    ),
                    unsafe_allow_html=True
                )
        st.markdown("""
- Start with ARIMA/Prophet for rapid pilots, then graduate to XGBoost as more features become available.
- Ensembles are ideal for executive confidence, blending strengths of each technique.
- Align technique selection with SKU criticality, data richness, and latency requirements.
        """)
    
    with tabs[4]:
        st.markdown("#### How do MAPE and WAPE frame performance?")
        if backtest_metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    create_metric_card(
                        "Mean Absolute Percentage Error",
                        f"{backtest_metrics['ai_mape']:.1f}%",
                        delta="Lower is better",
                        delta_type="positive",
                        icon="📐"
                    ),
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    create_metric_card(
                        "Weighted Absolute Percentage Error",
                        f"{backtest_metrics['ai_wape']:.1f}%",
                        delta="Revenue-weighted",
                        delta_type="positive",
                        icon="⚖️"
                    ),
                    unsafe_allow_html=True
                )
        st.markdown("""
- **MAPE**: Percentage error per day/SKU – perfect for benchmarking planner accuracy.
- **WAPE**: Scales error by volume/revenue – ideal for CFO briefings on revenue protection.
- Track both monthly and by product family to pinpoint where to invest in data or model refinements.
        """)
    
    with tabs[5]:
        st.markdown("#### Why real-time visibility and automated replenishment matter?")
        if inventory_metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    create_metric_card(
                        "Current Inventory",
                        f"{inventory_metrics.get('current_inventory', 0):,.0f} units",
                        delta=f"Reorder @ {inventory_metrics.get('reorder_point', 0):,.0f}",
                        delta_type="negative" if inventory_metrics.get('current_inventory', 0) < inventory_metrics.get('reorder_point', 0) else "positive",
                        icon="📦"
                    ),
                    unsafe_allow_html=True
                )
            with col2:
                days_of_stock = inventory_metrics.get('days_of_stock')
                st.markdown(
                    create_metric_card(
                        "Days of Cover",
                        f"{days_of_stock:.0f} days" if days_of_stock is not None else "—",
                        delta=f"Lead time: {inventory_metrics.get('lead_time', 0)} days",
                        delta_type="positive",
                        icon="🕒"
                    ),
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    create_metric_card(
                        "Safety Stock",
                        f"{inventory_metrics.get('safety_stock', 0):,.0f} units",
                        delta="Auto-adjusted from demand volatility",
                        delta_type="positive",
                        icon="🛡️"
                    ),
                    unsafe_allow_html=True
                )
        else:
            st.info("Run the Inventory Optimization workflow to populate live replenishment metrics.")
        
        st.markdown("""
- Always-on telemetry triggers replenishment orders when real inventory crosses AI-calculated thresholds.
- Automating replenishment lowers manual intervention, shortens cycle times, and boosts service levels.
- Use these metrics in S&OP meetings to align operations, merchandising, and finance on a single source of truth.
        """)
    
    with tabs[6]:
        st.markdown("#### How does AI forecasting support sustainability & cost reduction?")
        if inventory_metrics:
            carrying_delta = None
            if inventory_metrics.get('holding_cost') is not None and inventory_metrics.get('overstock_gap_units') is not None:
                carrying_delta = inventory_metrics['overstock_gap_units'] * inventory_metrics['holding_cost']
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    create_metric_card(
                        "Reduced Carrying Cost",
                        f"${carrying_delta:,.0f}/cycle" if carrying_delta and carrying_delta > 0 else "On target",
                        delta="Driven by lower safety stock",
                        delta_type="positive",
                        icon="💰"
                    ),
                    unsafe_allow_html=True
                )
            with col2:
                reorder_qty = inventory_metrics.get('economic_order_quantity')
                ordering_cost = inventory_metrics.get('ordering_cost')
                if reorder_qty and ordering_cost:
                    order_savings = ordering_cost / max(reorder_qty, 1) * 100
                else:
                    order_savings = None
                st.markdown(
                    create_metric_card(
                        "Order Efficiency",
                        f"{reorder_qty:,.0f} EOQ" if reorder_qty else "—",
                        delta=f"Cost per unit ↓ ~{order_savings:.1f}%" if order_savings else "Balanced batches",
                        delta_type="positive",
                        icon="♻️"
                    ),
                    unsafe_allow_html=True
                )
        st.markdown("""
- Leaner safety stock cuts energy, storage space, and waste – contributing to Scope 3 reductions.
- Smarter EOQ and lead-time alignment lowers rush shipments, packaging, and carbon-intensive expedites.
- Align AI-driven savings to ESG scorecards and cost-to-serve KPIs for executive sign-off.
        """)

def generate_pdf_report(report_type="Full Report", date_range=None, include_charts=True, include_recommendations=True):
    """Generate a proper PDF report with available data"""
    if not REPORTLAB_AVAILABLE:
        # Fallback: create a simple text-based PDF-like content
        return None
    
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph(f"<b>Inventory Forecasting Report - {report_type}</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Date and metadata
        date_text = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        story.append(date_text)
        if date_range:
            if isinstance(date_range, tuple) and len(date_range) == 2:
                period_text = Paragraph(f"Report Period: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}", styles['Normal'])
                story.append(period_text)
        story.append(Spacer(1, 0.3*inch))
        
        # Data Overview Section
        if st.session_state.data is not None:
            data_title = Paragraph("<b>Data Overview</b>", styles['Heading2'])
            story.append(data_title)
            story.append(Spacer(1, 0.1*inch))
            
            df = st.session_state.data
            data_overview = [
                ['Metric', 'Value'],
                ['Total Records', str(len(df))],
                ['Date Range', f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"],
                ['Average Sales', f"{df['sales'].mean():.2f}"],
                ['Total Sales', f"{df['sales'].sum():,.0f}"],
                ['Min Sales', f"{df['sales'].min():.2f}"],
                ['Max Sales', f"{df['sales'].max():.2f}"],
                ['Std Deviation', f"{df['sales'].std():.2f}"]
            ]
            
            data_table = Table(data_overview, colWidths=[3*inch, 3*inch])
            data_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(data_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Forecast Results Section
        forecast_results = st.session_state.forecast_results
        if forecast_results is not None:
            # Check if it's not empty (handle both dict and DataFrame)
            is_empty = False
            if isinstance(forecast_results, dict):
                is_empty = len(forecast_results) == 0
            elif hasattr(forecast_results, '__len__'):
                try:
                    is_empty = len(forecast_results) == 0
                except:
                    is_empty = False
            
            if not is_empty:
                forecast_title = Paragraph("<b>Forecast Results</b>", styles['Heading2'])
                story.append(forecast_title)
                story.append(Spacer(1, 0.1*inch))
                
                # Get forecast data - handle different data structures
                if isinstance(forecast_results, dict):
                    # Handle dict structure
                    forecast_text = Paragraph("Forecast data available (dict format)", styles['Normal'])
                    story.append(forecast_text)
                elif hasattr(forecast_results, 'to_dict') or hasattr(forecast_results, 'columns'):
                    # It's a DataFrame
                    try:
                        forecast_df = pd.DataFrame(forecast_results) if not isinstance(forecast_results, pd.DataFrame) else forecast_results
                        if 'forecast' in forecast_df.columns:
                            forecast_summary = [
                                ['Metric', 'Value'],
                                ['Forecast Period', f"{len(forecast_df)} days"],
                                ['Average Forecasted Demand', f"{forecast_df['forecast'].mean():.2f}"],
                                ['Total Forecasted Demand', f"{forecast_df['forecast'].sum():,.0f}"],
                                ['Min Forecast', f"{forecast_df['forecast'].min():.2f}"],
                                ['Max Forecast', f"{forecast_df['forecast'].max():.2f}"]
                            ]
                            
                            forecast_table = Table(forecast_summary, colWidths=[3*inch, 3*inch])
                            forecast_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 12),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ]))
                            story.append(forecast_table)
                            story.append(Spacer(1, 0.3*inch))
                    except Exception as e:
                        forecast_text = Paragraph(f"Forecast data available (error processing: {str(e)})", styles['Normal'])
                        story.append(forecast_text)
                
                # Add selected model info if available
                if st.session_state.selected_model:
                    model_info = Paragraph(f"<b>Forecast Model Used:</b> {st.session_state.selected_model}", styles['Normal'])
                    story.append(model_info)
                    story.append(Spacer(1, 0.2*inch))
        
        # Inventory Recommendations Section
        inv_recommendations = st.session_state.inventory_recommendations
        if inv_recommendations is not None:
            # Check if it's not empty
            is_empty = False
            if isinstance(inv_recommendations, dict):
                is_empty = len(inv_recommendations) == 0
            elif hasattr(inv_recommendations, '__len__'):
                try:
                    is_empty = len(inv_recommendations) == 0
                except:
                    is_empty = False
            
            if not is_empty:
                inv_title = Paragraph("<b>Inventory Recommendations</b>", styles['Heading2'])
                story.append(inv_title)
                story.append(Spacer(1, 0.1*inch))
                
                inv_metrics = inv_recommendations
                if isinstance(inv_metrics, dict):
                    inv_data = [['Metric', 'Value']]
                    
                    # Add available metrics
                    metric_labels = {
                        'avg_daily_demand': 'Average Daily Demand',
                        'safety_stock': 'Safety Stock',
                        'reorder_point': 'Reorder Point',
                        'economic_order_quantity': 'Economic Order Quantity',
                        'service_level': 'Service Level',
                        'stockout_risk': 'Stockout Risk',
                        'current_inventory': 'Current Inventory',
                        'recommended_max_inventory': 'Recommended Max Inventory'
                    }
                    
                    for key, label in metric_labels.items():
                        if key in inv_metrics:
                            value = inv_metrics[key]
                            if 'level' in key or 'risk' in key:
                                inv_data.append([label, f"{value:.2f}%"])
                            elif 'inventory' in key or 'stock' in key or 'point' in key or 'quantity' in key or 'demand' in key:
                                inv_data.append([label, f"{value:.2f} units"])
                            else:
                                inv_data.append([label, f"{value:.2f}"])
                    
                    if len(inv_data) > 1:  # More than just header
                        inv_table = Table(inv_data, colWidths=[3*inch, 3*inch])
                        inv_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 12),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ]))
                        story.append(inv_table)
                        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer = Paragraph("<i>Generated by IntelliStock AI - Inventory Intelligence Platform</i>", styles['Normal'])
        story.append(footer)
        
        # Build PDF - ensure we have at least some content
        if len(story) == 0:
            # Add minimal content if nothing was added
            story.append(Paragraph("<b>No data available for report generation</b>", styles['Title']))
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Please load data and generate forecasts before creating a report.", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    
    except Exception as e:
        import traceback
        error_msg = f"Error generating PDF: {str(e)}\n{traceback.format_exc()}"
        # Don't call st.error here as this function might be called outside streamlit context
        print(error_msg)
        return None

def show_reports_page():
    """Display the reports page"""
    
    st.markdown("""
    <div class="section-card">
        <h2>📑 Reports & Exports</h2>
        <p style="color: #718096;">Generate comprehensive reports and export your analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3 style="margin-bottom: 1.5rem;">📊 Quick Reports</h3>
        """, unsafe_allow_html=True)
        
        report_type = st.selectbox(
            "Select Report Type",
            ["Executive Summary", "Forecast Analysis", "Inventory Report", "Cost Analysis", "Full Report"]
        )
        
        date_range = st.date_input(
            "Report Period",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            format="MM/DD/YYYY"
        )
        
        include_charts = st.checkbox("Include Visualizations", value=True)
        include_recommendations = st.checkbox("Include AI Recommendations", value=True)
        
        if st.button("📄 Generate Report", use_container_width=True):
            if not REPORTLAB_AVAILABLE:
                st.error("⚠️ PDF generation requires the 'reportlab' library. Please install it using: pip install reportlab")
            else:
                with st.spinner("Generating your report..."):
                    try:
                        # Generate PDF
                        pdf_data = generate_pdf_report(
                            report_type=report_type,
                            date_range=date_range,
                            include_charts=include_charts,
                            include_recommendations=include_recommendations
                        )
                        
                        if pdf_data:
                            st.success("✅ Report generated successfully!")
                            
                            # Download button with proper PDF data
                            st.download_button(
                                label="📥 Download Report (PDF)",
                                data=pdf_data,
                                file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        else:
                            st.error("❌ Failed to generate PDF report. Please check if you have data loaded.")
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
                        st.info("💡 Tip: Make sure you have data loaded and forecasts/inventory recommendations available.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3 style="margin-bottom: 1.5rem;">💾 Data Exports</h3>
        """, unsafe_allow_html=True)
        
        export_format = st.selectbox(
            "Export Format",
            ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"]
        )
        
        data_to_export = st.multiselect(
            "Select Data to Export",
            ["Historical Sales", "Forecast Results", "Inventory Recommendations", "Cost Analysis"],
            default=["Historical Sales", "Forecast Results"]
        )
        
        if st.button("💾 Export Data", use_container_width=True):
            with st.spinner("Preparing your export..."):
                time.sleep(1)
                
                # Mock export
                if "CSV" in export_format:
                    if st.session_state.data is not None:
                        csv = st.session_state.data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Export",
                            data=csv,
                            file_name=f"inventory_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.info("Export functionality for this format coming soon!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # # Recent reports section
    # st.markdown("""
    # <div class="section-card">
    #     <h3 style="margin-bottom: 1.5rem;">📋 Recent Reports</h3>
    # """, unsafe_allow_html=True)
    
    # recent_reports = [
    #     {'name': 'Monthly Executive Summary', 'date': '2024-01-15', 'size': '2.3 MB', 'type': 'PDF'},
    #     {'name': 'Q4 Inventory Analysis', 'date': '2024-01-10', 'size': '1.8 MB', 'type': 'Excel'},
    #     {'name': 'Forecast Accuracy Report', 'date': '2024-01-05', 'size': '1.2 MB', 'type': 'PDF'},
    # ]
    
    # for report in recent_reports:
    #     st.markdown(f"""
    #     <div style="background: #f8fafc; padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center;">
    #         <div>
    #             <div style="font-weight: 600; color: #1a202c;">{report['name']}</div>
    #             <div style="font-size: 0.875rem; color: #718096; margin-top: 0.25rem;">
    #                 {report['date']} • {report['size']} • {report['type']}
    #             </div>
    #         </div>
    #         <button class="action-button" style="padding: 0.5rem 1rem; font-size: 0.875rem;">
    #             Download
    #         </button>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # st.markdown("</div>", unsafe_allow_html=True)

# Main application logic
def main():
    # Sidebar integration configuration
    configure_integration_settings()
    
    # Display header
    show_header()
    
    # Display navigation
    show_navigation()
    
    # Route to appropriate page
    if st.session_state.current_page == 'home':
        show_home_page()
    elif st.session_state.current_page == 'upload':
        show_upload_page()
    elif st.session_state.current_page == 'forecast':
        show_forecast_page()
    elif st.session_state.current_page == 'inventory':
        show_inventory_page()
    elif st.session_state.current_page == 'analytics':
        show_analytics_page()
    elif st.session_state.current_page == 'boardroom':
        show_boardroom_page()
    elif st.session_state.current_page == 'reports':
        show_reports_page()
    
    # Footer
    st.markdown("""
    <div style="margin-top: 4rem; padding: 2rem 0; border-top: 1px solid #e2e8f0; text-align: center; color: #718096;">
        <p style="margin: 0;">Powered by IntelliStock AI • © 2024 All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

# Include the actual forecasting and optimization classes from the previous code
class DemandForecaster:
    """Handles forecasting workflows across statistical and ML models."""
    
    def __init__(self, df: pd.DataFrame):
        if df is None or 'date' not in df.columns or 'sales' not in df.columns:
            raise ValueError("Dataframe must contain 'date' and 'sales' columns.")
        
        data = df[['date', 'sales']].dropna().copy()
        data['date'] = pd.to_datetime(data['date'])
        data['sales'] = pd.to_numeric(data['sales'], errors='coerce')
        data = data.dropna(subset=['sales'])
        data = data.sort_values('date').reset_index(drop=True)
        
        if len(data) < 20:
            raise ValueError("At least 20 records are required for forecasting.")
        
        self.data = data
        self.freq = pd.infer_freq(self.data['date'])
        if self.freq is None:
            self.freq = 'D'
        self.offset = self._get_offset(self.freq)
    
    @staticmethod
    def _get_offset(freq: str):
        try:
            return pd.tseries.frequencies.to_offset(freq)
        except Exception:
            return pd.tseries.frequencies.to_offset('D')
    
    def _seasonal_period(self):
        if self.freq.startswith('W'):
            return 52
        if self.freq.startswith('M'):
            return 12
        return 7
    
    @staticmethod
    def _safe_mape(actual, predicted):
        actual = np.array(actual, dtype=float)
        predicted = np.array(predicted, dtype=float)
        mask = actual != 0
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    
    @staticmethod
    def _safe_wape(actual, predicted):
        actual = np.array(actual, dtype=float)
        predicted = np.array(predicted, dtype=float)
        denom = np.abs(actual).sum()
        if denom == 0:
            return np.nan
        return float(np.abs(actual - predicted).sum() / denom * 100)
    
    @staticmethod
    def _to_float(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return value
    
    def _future_dates(self, start_date, periods):
        return pd.date_range(start=start_date + self.offset, periods=periods, freq=self.freq)
    
    def _forecast_arima(self, data, horizon, confidence_level, include_seasonality):
        series = data.set_index('date')['sales']
        try:
            if include_seasonality and len(series) > self._seasonal_period() * 2:
                model = SARIMAX(
                    series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, self._seasonal_period()),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted = model.fit(disp=False)
            else:
                fitted = ARIMA(series, order=(1, 1, 1)).fit()
        except Exception:
            fitted = ARIMA(series, order=(1, 1, 1)).fit()
        
        forecast_res = fitted.get_forecast(steps=horizon)
        conf_int = forecast_res.conf_int(alpha=1 - confidence_level)
        conf_int = conf_int.rename(columns=lambda c: c.replace('lower', 'lower_bound').replace('upper', 'upper_bound'))
        forecast_mean = forecast_res.predicted_mean
        future_dates = self._future_dates(series.index[-1], horizon)
        
        lower_col = conf_int.columns[0]
        upper_col = conf_int.columns[1]
        
        return pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_mean.values,
            'lower_bound': conf_int[lower_col].values,
            'upper_bound': conf_int[upper_col].values
        })
    
    def _forecast_prophet(self, data, horizon, confidence_level, include_seasonality, include_holidays):
        prophet_df = data.rename(columns={'date': 'ds', 'sales': 'y'})
        model = Prophet(
            interval_width=confidence_level,
            yearly_seasonality=include_seasonality,
            weekly_seasonality=include_seasonality,
            daily_seasonality=False
        )
        if include_holidays:
            try:
                model.add_country_holidays(country_name='US')
            except Exception:
                pass
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=horizon, freq=self.freq)
        forecast = model.predict(future)
        tail = forecast.tail(horizon)
        return pd.DataFrame({
            'date': pd.to_datetime(tail['ds']),
            'forecast': tail['yhat'].values,
            'lower_bound': tail['yhat_lower'].values,
            'upper_bound': tail['yhat_upper'].values
        })
    
    def _forecast_xgboost(self, data, horizon, confidence_level):
        df = data.copy()
        df['lag_1'] = df['sales'].shift(1)
        df['lag_7'] = df['sales'].shift(7)
        df['lag_14'] = df['sales'].shift(14)
        df['lag_30'] = df['sales'].shift(30)
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        feature_cols = ['lag_1', 'lag_7', 'lag_14', 'lag_30', 'dayofweek', 'month']
        df = df.dropna(subset=feature_cols + ['sales'])
        if len(df) < 30:
            raise ValueError("Not enough history for XGBoost forecasting.")
        
        X = df[feature_cols]
        y = df['sales']
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        model.fit(X, y)
        residuals = y - model.predict(X)
        residual_std = float(residuals.std()) if residuals.std() > 0 else float(y.std() * 0.1)
        z_score = 1.96 if confidence_level >= 0.95 else 1.65
        buffer = residual_std * z_score
        
        history = data[['date', 'sales']].copy().reset_index(drop=True)
        predictions = []
        
        for _ in range(horizon):
            next_date = history['date'].iloc[-1] + self.offset
            history_values = history['sales']
            features = {
                'lag_1': history_values.iloc[-1],
                'lag_7': history_values.iloc[-7] if len(history_values) >= 7 else history_values.iloc[-1],
                'lag_14': history_values.iloc[-14] if len(history_values) >= 14 else history_values.iloc[-1],
                'lag_30': history_values.iloc[-30] if len(history_values) >= 30 else history_values.iloc[-1],
                'dayofweek': next_date.dayofweek,
                'month': next_date.month
            }
            pred = float(model.predict(pd.DataFrame([features]))[0])
            predictions.append({
                'date': next_date,
                'forecast': pred,
                'lower_bound': max(pred - buffer, 0),
                'upper_bound': pred + buffer
            })
            history = pd.concat([history, pd.DataFrame({'date': [next_date], 'sales': [pred]})], ignore_index=True)
        
        return pd.DataFrame(predictions)
    
    def _forecast_ensemble(self, data, horizon, confidence_level, include_seasonality, include_holidays):
        forecasts = []
        for method in ('arima', 'prophet', 'xgboost'):
            try:
                fc = self._forecast_model(
                    data,
                    method,
                    horizon,
                    confidence_level,
                    include_seasonality,
                    include_holidays
                ).set_index('date')
                fc = fc.rename(columns={
                    'forecast': f'forecast_{method}',
                    'lower_bound': f'lower_bound_{method}',
                    'upper_bound': f'upper_bound_{method}'
                })
                forecasts.append(fc)
            except Exception:
                continue
        if not forecasts:
            raise ValueError("No base models produced forecasts for the ensemble.")
        
        combined = pd.concat(forecasts, axis=1)
        forecast_cols = [col for col in combined.columns if col.startswith('forecast_')]
        lower_cols = [col for col in combined.columns if col.startswith('lower_bound_')]
        upper_cols = [col for col in combined.columns if col.startswith('upper_bound_')]
        
        result = pd.DataFrame({
            'date': combined.index,
            'forecast': combined[forecast_cols].mean(axis=1),
            'lower_bound': combined[lower_cols].min(axis=1),
            'upper_bound': combined[upper_cols].max(axis=1)
        }).reset_index(drop=True)
        return result
    
    def _forecast_model(self, data, model_name, horizon, confidence_level, include_seasonality, include_holidays):
        model_name = model_name.lower()
        if model_name == 'arima':
            return self._forecast_arima(data, horizon, confidence_level, include_seasonality)
        if model_name == 'prophet':
            return self._forecast_prophet(data, horizon, confidence_level, include_seasonality, include_holidays)
        if model_name == 'xgboost':
            return self._forecast_xgboost(data, horizon, confidence_level)
        if model_name == 'ensemble':
            return self._forecast_ensemble(data, horizon, confidence_level, include_seasonality, include_holidays)
        raise ValueError(f"Unsupported model {model_name}")
    
    def _train_test_split(self, test_window=None):
        n = len(self.data)
        if n < 20:
            raise ValueError("Not enough records for backtesting.")
        if test_window is None:
            test_window = max(14, int(n * 0.2))
        test_window = min(max(test_window, 7), n - 7)
        if test_window <= 0:
            test_window = max(7, n // 3)
        train = self.data.iloc[:-test_window].copy()
        test = self.data.iloc[-test_window:].copy()
        return train, test
    
    def backtest(self, model_name='prophet', test_window=None, confidence_level=0.95, include_seasonality=True, include_holidays=False):
        train, test = self._train_test_split(test_window)
        horizon = len(test)
        forecast_df = self._forecast_model(train, model_name, horizon, confidence_level, include_seasonality, include_holidays)
        forecast_df = forecast_df.sort_values('date').reset_index(drop=True)
        
        baseline_forecast = np.repeat(train['sales'].iloc[-1], horizon)
        ai_forecast = forecast_df['forecast'].values
        actuals = test['sales'].values
        
        baseline_mape = self._safe_mape(actuals, baseline_forecast)
        ai_mape = self._safe_mape(actuals, ai_forecast)
        baseline_wape = self._safe_wape(actuals, baseline_forecast)
        ai_wape = self._safe_wape(actuals, ai_forecast)
        
        metrics = {
            'model': model_name.upper(),
            'test_days': horizon,
            'training_records': len(train),
            'baseline_mape': self._to_float(baseline_mape),
            'ai_mape': self._to_float(ai_mape),
            'baseline_wape': self._to_float(baseline_wape),
            'ai_wape': self._to_float(ai_wape),
            'mape_improvement': self._to_float(baseline_mape - ai_mape) if baseline_mape is not None and ai_mape is not None and not np.isnan(baseline_mape) and not np.isnan(ai_mape) else None,
            'wape_improvement': self._to_float(baseline_wape - ai_wape) if baseline_wape is not None and ai_wape is not None and not np.isnan(baseline_wape) and not np.isnan(ai_wape) else None,
            'ai_rmse': self._to_float(sqrt(mean_squared_error(actuals, ai_forecast))),
            'ai_mae': self._to_float(mean_absolute_error(actuals, ai_forecast)),
            'test_actuals': test.reset_index(drop=True),
            'test_forecast': ai_forecast.tolist(),
            'forecast_dates': forecast_df['date'].tolist()
        }
        return metrics
    
    def forecast(self, model_name='prophet', horizon=30, confidence_level=0.95, include_seasonality=True, include_holidays=False):
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive.")
        
        forecast_df = self._forecast_model(
            self.data,
            model_name,
            horizon,
            confidence_level,
            include_seasonality,
            include_holidays
        )
        forecast_df['method'] = model_name
        
        backtest_metrics = self.backtest(
            model_name=model_name,
            confidence_level=confidence_level,
            include_seasonality=include_seasonality,
            include_holidays=include_holidays
        )
        
        summary = {
            'avg_forecast': self._to_float(forecast_df['forecast'].mean()),
            'total_forecast': self._to_float(forecast_df['forecast'].sum()),
            'peak_date': forecast_df.loc[forecast_df['forecast'].idxmax(), 'date'].strftime('%Y-%m-%d') if len(forecast_df) else None,
            'peak_value': self._to_float(forecast_df['forecast'].max())
        }
        
        return {
            'forecast': forecast_df,
            'metrics': backtest_metrics,
            'summary': summary,
            'backtest': backtest_metrics
        }

class InventoryOptimizer:
    """Handles inventory optimization calculations"""
    # ... (include the full implementation from the previous code)
    pass

# Run the application
if __name__ == "__main__":
    main()
