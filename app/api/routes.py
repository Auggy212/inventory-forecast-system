"""
API routes for the application
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import json
from typing import Optional, List
from datetime import datetime, timedelta
import io
import base64

from app.utils.data_processor import DataProcessor
from app.models.forecasting import ForecastingEngine
from app.models.inventory import InventoryOptimizer
from app.utils.metrics import MetricsCalculator
from app.api.schemas import ForecastRequest, InventoryRequest, UploadResponse

router = APIRouter()

# Initialize components
data_processor = DataProcessor()
forecasting_engine = ForecastingEngine()
inventory_optimizer = InventoryOptimizer()
metrics_calculator = MetricsCalculator()

# In-memory data storage (replace with database in production)
uploaded_data = {}

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process sales data"""
    try:
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO((await file.read()).decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(await file.read()))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate data
        is_valid, message = data_processor.validate_data(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Clean and process data
        df = data_processor.clean_data(df)
        
        # Calculate initial metrics
        metrics = data_processor.calculate_inventory_metrics(df)
        
        # Store data
        session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        uploaded_data[session_id] = df
        
        # Get summary statistics
        products = df['product_id'].unique().tolist()
        date_range = {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        }
        
        return {
            'session_id': session_id,
            'status': 'success',
            'message': 'Data uploaded and processed successfully',
            'summary': {
                'total_records': len(df),
                'products': products,
                'date_range': date_range,
                'metrics': metrics
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast")
async def generate_forecast(request: ForecastRequest):
    """Generate demand forecast"""
    try:
        # Retrieve data
        if request.session_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = uploaded_data[request.session_id]
        
        # Prepare data for forecasting
        product_data = data_processor.prepare_forecasting_data(df, request.product_id)
        
        # Generate forecast
        forecast_result = forecasting_engine.forecast(
            product_data,
            request.model,
            request.forecast_days
        )
        
        # Calculate metrics
        if 'historical_fit' in forecast_result and forecast_result['historical_fit'] is not None:
            actual = product_data['sales'].values[-len(forecast_result['historical_fit']):]
            metrics = metrics_calculator.calculate_forecast_metrics(
                actual,
                forecast_result['historical_fit']
            )
            forecast_result['metrics'] = metrics
        
        # Convert numpy arrays to lists for JSON serialization
        for key in ['forecast', 'dates', 'lower', 'upper', 'historical_fit']:
            if key in forecast_result and forecast_result[key] is not None:
                if hasattr(forecast_result[key], 'tolist'):
                    forecast_result[key] = forecast_result[key].tolist()
                elif isinstance(forecast_result[key], np.ndarray):
                    forecast_result[key] = forecast_result[key].tolist()
        
        return JSONResponse(content=forecast_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-inventory")
async def optimize_inventory(request: InventoryRequest):
    """Optimize inventory levels"""
    try:
        # Get forecast data
        forecast_data = request.forecast_data
        
        # Optimize inventory
        optimization_result = inventory_optimizer.calculate_optimal_inventory(
            forecast_data,
            request.lead_time_days
        )
        
        # Convert numpy arrays to lists
        for key in ['inventory_levels', 'stockout_risk', 'overstock_risk']:
            if key in optimization_result and hasattr(optimization_result[key], 'tolist'):
                optimization_result[key] = optimization_result[key].tolist()
        
        return JSONResponse(content=optimization_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/products/{session_id}")
async def get_products(session_id: str):
    """Get list of products in the uploaded data"""
    try:
        if session_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = uploaded_data[session_id]
        products = df['product_id'].unique().tolist()
        
        # Get product statistics
        product_stats = []
        for product in products:
            product_df = df[df['product_id'] == product]
            stats = {
                'product_id': product,
                'avg_sales': float(product_df['sales'].mean()),
                'total_sales': float(product_df['sales'].sum()),
                'sales_volatility': float(product_df['sales'].std()),
                'days_active': len(product_df)
            }
            product_stats.append(stats)
        
        return {
            'products': products,
            'statistics': product_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kpis/{session_id}")
async def get_kpis(session_id: str, product_id: Optional[str] = None):
    """Get key performance indicators"""
    try:
        if session_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = uploaded_data[session_id]
        
        # Filter by product if specified
        if product_id:
            df = df[df['product_id'] == product_id]
        
        # Calculate KPIs
        kpis = {
            'total_products': df['product_id'].nunique(),
            'total_sales': float(df['sales'].sum()),
            'avg_daily_sales': float(df.groupby('date')['sales'].sum().mean()),
            'sales_trend': 'increasing' if df.groupby('date')['sales'].sum().tail(30).mean() > 
                          df.groupby('date')['sales'].sum().head(30).mean() else 'decreasing',
            'forecast_accuracy': 92.5,  # Placeholder - would be calculated from actual forecasts
            'inventory_turnover': 12.3,  # Placeholder
            'stockout_incidents': int((df['sales'] == 0).sum()),
            'service_level': 95.2  # Placeholder
        }
        
        return kpis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-models")
async def compare_models(request: dict):
    """Compare multiple forecasting models"""
    try:
        session_id = request['session_id']
        product_id = request['product_id']
        models = request.get('models', ['prophet', 'arima', 'xgboost'])
        forecast_days = request.get('forecast_days', 30)
        
        if session_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = uploaded_data[session_id]
        product_data = data_processor.prepare_forecasting_data(df, product_id)
        
        # Run forecasts for each model
        results = {}
        for model in models:
            try:
                forecast = forecasting_engine.forecast(product_data, model, forecast_days)
                
                # Calculate metrics if historical fit available
                if forecast.get('historical_fit') is not None:
                    actual = product_data['sales'].values[-len(forecast['historical_fit']):]
                    metrics = metrics_calculator.calculate_forecast_metrics(
                        actual, forecast['historical_fit']
                    )
                    forecast['metrics'] = metrics
                
                results[model] = forecast
            except Exception as model_error:
                results[model] = {'error': str(model_error)}
        
        # Convert numpy arrays to lists
        for model_results in results.values():
            if isinstance(model_results, dict) and 'error' not in model_results:
                for key in ['forecast', 'dates', 'lower', 'upper', 'historical_fit']:
                    if key in model_results and model_results[key] is not None:
                        if hasattr(model_results[key], 'tolist'):
                            model_results[key] = model_results[key].tolist()
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scenario-analysis")
async def scenario_analysis(request: dict):
    """Run scenario analysis"""
    try:
        session_id = request['session_id']
        product_id = request['product_id']
        scenarios = request['scenarios']
        
        if session_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = uploaded_data[session_id]
        results = {}
        
        for scenario in scenarios:
            # Apply scenario modifications
            scenario_df = df.copy()
            
            if 'promotion_lift' in scenario:
                scenario_df['sales'] = scenario_df['sales'] * (1 + scenario['promotion_lift'])
            
            if 'seasonality_factor' in scenario:
                # Apply seasonal adjustment
                scenario_df['sales'] = scenario_df['sales'] * scenario['seasonality_factor']
            
            # Prepare data and forecast
            product_data = data_processor.prepare_forecasting_data(scenario_df, product_id)
            forecast = forecasting_engine.forecast(product_data, 'prophet', 30)
            
            # Optimize inventory for scenario
            inventory = inventory_optimizer.calculate_optimal_inventory(forecast)
            
            results[scenario['name']] = {
                'forecast': forecast,
                'inventory': inventory,
                'impact': {
                    'sales_change': float((scenario_df['sales'].mean() - df['sales'].mean()) / df['sales'].mean()),
                    'inventory_change': float(inventory['reorder_point'] - 1000)  # Placeholder baseline
                }
            }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-report/{session_id}")
async def download_report(session_id: str, format: str = 'csv'):
    """Download analysis report"""
    try:
        if session_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = uploaded_data[session_id]
        
        if format == 'csv':
            # Create CSV report
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return FileResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type='text/csv',
                filename=f'inventory_report_{session_id}.csv'
            )
        
        elif format == 'pdf':
            # Create PDF report (simplified version)
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
            
            output = io.BytesIO()
            doc = SimpleDocTemplate(output, pagesize=letter)
            elements = []
            
            styles = getSampleStyleSheet()
            
            # Title
            elements.append(Paragraph("Inventory & Demand Forecasting Report", styles['Title']))
            elements.append(Paragraph(f"Session: {session_id}", styles['Normal']))
            elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
            
            # Summary statistics
            summary_data = [
                ['Metric', 'Value'],
                ['Total Products', str(df['product_id'].nunique())],
                ['Date Range', f"{df['date'].min()} to {df['date'].max()}"],
                ['Total Sales', f"{df['sales'].sum():,.0f}"],
                ['Average Daily Sales', f"{df.groupby('date')['sales'].sum().mean():,.0f}"]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(summary_table)
            
            # Build PDF
            doc.build(elements)
            output.seek(0)
            
            return FileResponse(
                output,
                media_type='application/pdf',
                filename=f'inventory_report_{session_id}.pdf'
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
