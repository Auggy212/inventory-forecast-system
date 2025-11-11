"""
Inventory optimization module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

class InventoryOptimizer:
    """Handles inventory optimization calculations"""
    
    def __init__(self, holding_cost_rate: float = 0.2, 
                 stockout_cost_rate: float = 0.5,
                 ordering_cost: float = 100):
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_rate = stockout_cost_rate
        self.ordering_cost = ordering_cost
        
    def calculate_optimal_inventory(self, forecast: Dict, 
                                   lead_time_days: int = 7) -> Dict:
        """Calculate optimal inventory levels based on forecast"""
        
        # Extract forecast data
        predicted_demand = forecast['forecast']
        dates = forecast['dates']
        lower_bound = forecast.get('lower', predicted_demand * 0.8)
        upper_bound = forecast.get('upper', predicted_demand * 1.2)
        
        # Calculate demand during lead time
        lead_time_demand = np.sum(predicted_demand[:lead_time_days])
        lead_time_std = np.std(predicted_demand[:lead_time_days])
        
        # Safety stock calculation (95% service level)
        z_score = stats.norm.ppf(0.95)
        safety_stock = z_score * lead_time_std * np.sqrt(lead_time_days)
        
        # Reorder point
        reorder_point = lead_time_demand + safety_stock
        
        # Economic Order Quantity (EOQ)
        annual_demand = np.sum(predicted_demand) * (365 / len(predicted_demand))
        eoq = np.sqrt((2 * annual_demand * self.ordering_cost) / self.holding_cost_rate)
        
        # Calculate inventory levels over time
        inventory_levels = self._simulate_inventory(
            predicted_demand, reorder_point, eoq, safety_stock
        )
        
        # Risk assessment
        stockout_risk = self._calculate_stockout_risk(inventory_levels, predicted_demand)
        overstock_risk = self._calculate_overstock_risk(inventory_levels, predicted_demand)
        
        # Cost analysis
        costs = self._calculate_costs(
            inventory_levels, predicted_demand, stockout_risk, overstock_risk
        )
        
        return {
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'eoq': eoq,
            'inventory_levels': inventory_levels,
            'stockout_risk': stockout_risk,
            'overstock_risk': overstock_risk,
            'costs': costs,
            'recommendations': self._generate_recommendations(
                stockout_risk, overstock_risk, costs
            )
        }
    
    def _simulate_inventory(self, demand: np.ndarray, reorder_point: float,
                           eoq: float, safety_stock: float) -> np.ndarray:
        """Simulate inventory levels over time"""
        inventory_levels = np.zeros(len(demand))
        current_inventory = reorder_point
        pending_order = 0
        order_arrival = -1
        
        for i in range(len(demand)):
            # Receive order if due
            if i == order_arrival:
                current_inventory += pending_order
                pending_order = 0
                order_arrival = -1
            
            # Fulfill demand
            current_inventory -= demand[i]
            
            # Check if need to reorder
            if current_inventory <= reorder_point and pending_order == 0:
                pending_order = eoq
                order_arrival = i + 7  # Lead time
            
            inventory_levels[i] = max(0, current_inventory)
        
        return inventory_levels
    
    def _calculate_stockout_risk(self, inventory: np.ndarray, 
                                demand: np.ndarray) -> np.ndarray:
        """Calculate stockout risk for each period"""
        risk = np.zeros(len(inventory))
        
        for i in range(len(inventory)):
            if inventory[i] < demand[i] * 0.1:  # Less than 10% of demand
                risk[i] = 1.0
            elif inventory[i] < demand[i] * 0.2:  # Less than 20% of demand
                risk[i] = 0.5
            else:
                risk[i] = 0.0
        
        return risk
    
    def _calculate_overstock_risk(self, inventory: np.ndarray,
                                 demand: np.ndarray) -> np.ndarray:
        """Calculate overstock risk for each period"""
        risk = np.zeros(len(inventory))
        
        for i in range(len(inventory)):
            if inventory[i] > demand[i] * 3:  # More than 3x demand
                risk[i] = 1.0
            elif inventory[i] > demand[i] * 2:  # More than 2x demand
                risk[i] = 0.5
            else:
                risk[i] = 0.0
        
        return risk
    
    def _calculate_costs(self, inventory: np.ndarray, demand: np.ndarray,
                        stockout_risk: np.ndarray, overstock_risk: np.ndarray) -> Dict:
        """Calculate various inventory costs"""
        
        # Holding costs
        avg_inventory = np.mean(inventory)
        holding_cost = avg_inventory * self.holding_cost_rate
        
        # Stockout costs
        stockout_instances = np.sum(stockout_risk > 0.5)
        stockout_cost = stockout_instances * np.mean(demand) * self.stockout_cost_rate
        
        # Overstock costs (obsolescence, spoilage)
        overstock_instances = np.sum(overstock_risk > 0.5)
        overstock_cost = overstock_instances * np.mean(demand) * 0.1
        
        # Total cost
        total_cost = holding_cost + stockout_cost + overstock_cost
        
        return {
            'holding_cost': holding_cost,
            'stockout_cost': stockout_cost,
            'overstock_cost': overstock_cost,
            'total_cost': total_cost,
            'cost_per_unit': total_cost / np.sum(demand)
        }
    
    def _generate_recommendations(self, stockout_risk: np.ndarray,
                                 overstock_risk: np.ndarray,
                                 costs: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Stockout risk recommendation
        avg_stockout_risk = np.mean(stockout_risk)
        if avg_stockout_risk > 0.3:
            recommendations.append({
                'type': 'critical',
                'title': 'High Stockout Risk',
                'description': f'Average stockout risk is {avg_stockout_risk:.1%}. Consider increasing safety stock.',
                'action': 'Increase safety stock by 20%',
                'impact': f'Reduce stockout cost by ${costs["stockout_cost"]*0.5:.2f}'
            })
        
        # Overstock risk recommendation
        avg_overstock_risk = np.mean(overstock_risk)
        if avg_overstock_risk > 0.3:
            recommendations.append({
                'type': 'warning',
                'title': 'High Overstock Risk',
                'description': f'Average overstock risk is {avg_overstock_risk:.1%}. Consider reducing order quantities.',
                'action': 'Reduce EOQ by 15%',
                'impact': f'Reduce holding cost by ${costs["holding_cost"]*0.15:.2f}'
            })
        
        # Cost optimization recommendation
        if costs['total_cost'] > np.mean([costs['holding_cost'], costs['stockout_cost']]) * 3:
            recommendations.append({
                'type': 'info',
                'title': 'Cost Optimization Opportunity',
                'description': 'Total inventory costs are high. Review ordering policies.',
                'action': 'Implement dynamic reorder points based on demand variability',
                'impact': f'Potential savings of ${costs["total_cost"]*0.2:.2f}'
            })
        
        return recommendations
