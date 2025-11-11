"""
Pydantic schemas for API validation
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class ForecastRequest(BaseModel):
    session_id: str
    product_id: str
    model: str = Field(default="prophet", description="Forecasting model to use")
    forecast_days: int = Field(default=30, ge=1, le=365)

class InventoryRequest(BaseModel):
    forecast_data: Dict
    lead_time_days: int = Field(default=7, ge=1, le=30)
    holding_cost_rate: float = Field(default=0.2, ge=0, le=1)
    stockout_cost_rate: float = Field(default=0.5, ge=0, le=2)

class UploadResponse(BaseModel):
    session_id: str
    status: str
    message: str
    summary: Dict

class ScenarioConfig(BaseModel):
    name: str
    promotion_lift: Optional[float] = None
    seasonality_factor: Optional[float] = None
    external_event: Optional[str] = None
