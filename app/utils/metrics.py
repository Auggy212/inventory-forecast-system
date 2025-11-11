"""
Metrics calculation utilities
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict

class MetricsCalculator:
    """Calculate various performance metrics"""
    
    def calculate_forecast_metrics(self, actual: np.ndarray, 
                                 predicted: np.ndarray) -> Dict:
        """Calculate forecast accuracy metrics"""
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {}
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)
        
        # Bias
        bias = np.mean(predicted - actual)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'bias': float(bias),
            'accuracy': float(max(0, 100 - mape))
        }
    
    def calculate_inventory_metrics(self, inventory_levels: np.ndarray,
                                  demand: np.ndarray) -> Dict:
        """Calculate inventory performance metrics"""
        
        # Service level (% of demand fulfilled)
        fulfilled = np.minimum(inventory_levels, demand)
        service_level = np.sum(fulfilled) / np.sum(demand) * 100
        
        # Inventory turnover
        avg_inventory = np.mean(inventory_levels)
        if avg_inventory > 0:
            turnover = np.sum(demand) / avg_inventory
        else:
            turnover = 0
        
        # Fill rate
        stockouts = demand > inventory_levels
        fill_rate = (1 - np.sum(stockouts) / len(demand)) * 100
        
        return {
            'service_level': float(service_level),
            'inventory_turnover': float(turnover),
            'fill_rate': float(fill_rate),
            'avg_inventory': float(avg_inventory),
            'stockout_days': int(np.sum(stockouts))
        }
