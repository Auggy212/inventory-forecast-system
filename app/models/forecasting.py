"""
Forecasting models implementation
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ForecastingEngine:
    """Main forecasting engine with multiple model support"""
    
    def __init__(self):
        self.models = {
            'prophet': self._forecast_prophet,
            'arima': self._forecast_arima,
            'xgboost': self._forecast_xgboost,
            'lstm': self._forecast_lstm
        }
        
    def forecast(self, data: pd.DataFrame, model_name: str, 
                 forecast_days: int = 30) -> Dict:
        """Generate forecast using specified model"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported")
        
        # Call appropriate model
        forecast_result = self.models[model_name](data, forecast_days)
        
        # Calculate confidence intervals
        forecast_result['confidence_interval'] = self._calculate_confidence_interval(
            forecast_result['forecast'],
            forecast_result.get('std', None)
        )
        
        return forecast_result
    
    def _forecast_prophet(self, data: pd.DataFrame, forecast_days: int) -> Dict:
        """Prophet model forecasting"""
        # Prepare data for Prophet
        prophet_df = data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
        
        # Initialize and fit Prophet
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Add custom seasonality if needed
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        model.fit(prophet_df)
        
        # Make predictions
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Extract results
        historical = forecast[forecast['ds'].isin(data['date'])]
        future_forecast = forecast[~forecast['ds'].isin(data['date'])]
        
        return {
            'forecast': future_forecast['yhat'].values,
            'dates': future_forecast['ds'].values,
            'lower': future_forecast['yhat_lower'].values,
            'upper': future_forecast['yhat_upper'].values,
            'model': 'prophet',
            'historical_fit': historical['yhat'].values
        }
    
    def _forecast_arima(self, data: pd.DataFrame, forecast_days: int) -> Dict:
        """ARIMA model forecasting"""
        # Prepare time series
        ts = data.set_index('date')['sales']
        
        # Auto-select ARIMA parameters (simplified)
        model = ARIMA(ts, order=(2, 1, 2))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=forecast_days)
        forecast_df = fitted_model.get_forecast(steps=forecast_days)
        
        # Get confidence intervals
        confidence_int = forecast_df.conf_int()
        
        # Generate future dates
        last_date = data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days
        )
        
        return {
            'forecast': forecast.values,
            'dates': future_dates,
            'lower': confidence_int.iloc[:, 0].values,
            'upper': confidence_int.iloc[:, 1].values,
            'model': 'arima',
            'historical_fit': fitted_model.fittedvalues.values
        }
    
    def _forecast_xgboost(self, data: pd.DataFrame, forecast_days: int) -> Dict:
        """XGBoost model forecasting"""
        # Feature engineering
        features = ['year', 'month', 'week', 'dayofweek', 'quarter']
        lag_features = [col for col in data.columns if 'lag' in col]
        ma_features = [col for col in data.columns if 'ma' in col]
        
        all_features = features + lag_features + ma_features
        available_features = [f for f in all_features if f in data.columns]
        
        # Prepare training data
        train_data = data.dropna()
        X_train = train_data[available_features]
        y_train = train_data['sales']
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Generate future features
        last_date = data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days
        )
        
        # Create future dataframe with features
        future_df = pd.DataFrame({'date': future_dates})
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['week'] = future_df['date'].dt.isocalendar().week
        future_df['dayofweek'] = future_df['date'].dt.dayofweek
        future_df['quarter'] = future_df['date'].dt.quarter
        
        # Use last known values for lag features
        for feature in lag_features + ma_features:
            if feature in available_features:
                future_df[feature] = train_data[feature].iloc[-1]
        
        # Make predictions
        X_future = future_df[available_features]
        forecast = model.predict(X_future)
        
        # Calculate prediction intervals (simplified)
        std = np.std(y_train) * 0.1  # Simplified uncertainty
        
        return {
            'forecast': forecast,
            'dates': future_dates,
            'lower': forecast - 2 * std,
            'upper': forecast + 2 * std,
            'model': 'xgboost',
            'historical_fit': model.predict(X_train)
        }
    
    def _forecast_lstm(self, data: pd.DataFrame, forecast_days: int) -> Dict:
        """LSTM model forecasting"""
        # Prepare sequence data
        sequence_length = 30
        sales_data = data['sales'].values
        
        # Normalize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(sales_data.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # Make predictions
        last_sequence = scaled_data[-sequence_length:]
        predictions = []
        
        for _ in range(forecast_days):
            next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
            predictions.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], next_pred)
        
        # Inverse transform
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Generate dates
        last_date = data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days
        )
        
        # Calculate uncertainty (simplified)
        std = np.std(sales_data) * 0.15
        
        return {
            'forecast': predictions.flatten(),
            'dates': future_dates,
            'lower': predictions.flatten() - 2 * std,
            'upper': predictions.flatten() + 2 * std,
            'model': 'lstm',
            'historical_fit': None  # LSTM doesn't provide simple historical fit
        }
    
    def _calculate_confidence_interval(self, forecast: np.ndarray, 
                                     std: Optional[np.ndarray] = None) -> Dict:
        """Calculate confidence intervals"""
        if std is None:
            std = np.std(forecast) * 0.1
        
        return {
            '95': {
                'lower': forecast - 1.96 * std,
                'upper': forecast + 1.96 * std
            },
            '80': {
                'lower': forecast - 1.28 * std,
                'upper': forecast + 1.28 * std
            }
        }
