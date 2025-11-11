"""
Data processing utilities for cleaning, validation, and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handles data cleaning, validation, and preprocessing"""
    
    def __init__(self):
        self.required_columns = ['date', 'product_id', 'sales']
        self.optional_columns = ['price', 'promotion', 'season', 'region']
        
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate uploaded data"""
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check data types
        try:
            df['date'] = pd.to_datetime(df['date'])
            df['sales'] = pd.to_numeric(df['sales'])
            df['product_id'] = df['product_id'].astype(str)
        except Exception as e:
            return False, f"Data type conversion error: {str(e)}"
        
        # Check for negative sales
        if (df['sales'] < 0).any():
            return False, "Negative sales values found"
        
        return True, "Data validation successful"
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Handle missing values
        df['sales'] = df['sales'].fillna(0)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'product_id'])
        
        # Add time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        # Add lag features
        for lag in [7, 14, 30]:
            df[f'sales_lag_{lag}'] = df.groupby('product_id')['sales'].shift(lag)
        
        # Add rolling features
        for window in [7, 30]:
            df[f'sales_ma_{window}'] = df.groupby('product_id')['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        return df
    
    def prepare_forecasting_data(self, df: pd.DataFrame, product_id: str) -> pd.DataFrame:
        """Prepare data for forecasting models"""
        # Filter by product
        product_df = df[df['product_id'] == product_id].copy()
        
        # Ensure continuous date range
        date_range = pd.date_range(
            start=product_df['date'].min(),
            end=product_df['date'].max(),
            freq='D'
        )
        
        product_df = product_df.set_index('date').reindex(date_range).reset_index()
        product_df['product_id'] = product_id
        
        # Fill missing sales with 0
        product_df['sales'] = product_df['sales'].fillna(0)
        
        return product_df
    
    def calculate_inventory_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate inventory-related metrics"""
        metrics = {}
        
        # Average daily sales
        metrics['avg_daily_sales'] = df.groupby('product_id')['sales'].mean().to_dict()
        
        # Sales variability (coefficient of variation)
        metrics['sales_cv'] = (
            df.groupby('product_id')['sales'].std() / 
            df.groupby('product_id')['sales'].mean()
        ).to_dict()
        
        # Stockout frequency (days with 0 sales)
        metrics['stockout_freq'] = (
            df.groupby('product_id')['sales'].apply(lambda x: (x == 0).sum() / len(x))
        ).to_dict()
        
        return metrics
