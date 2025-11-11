// Global variables
let sessionId = null;
let currentProduct = null;
let forecastData = null;

// File upload
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showStatus('Please select a file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showStatus('Uploading and processing data...', 'info');
        
        const response = await axios.post('/api/v1/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        
        sessionId = response.data.session_id;
        showStatus('Data uploaded successfully!', 'success');
        
        // Update UI
        document.getElementById('kpiSection').classList.remove('hidden');
        document.getElementById('controlPanel').classList.remove('hidden');
        document.getElementById('tabsSection').classList.remove('hidden');
        document.getElementById('downloadSection').classList.remove('hidden');
        
        // Load products
        await loadProducts();
        
        // Load initial KPIs
        await loadKPIs();
        
    } catch (error) {
        showStatus(`Error: ${error.response?.data?.detail || error.message}`, 'error');
    }
}

// Load products
async function loadProducts() {
    try {
        const response = await axios.get(`/api/v1/products/${sessionId}`);
        const select = document.getElementById('productSelect');
        
        select.innerHTML = '<option value="">Select Product</option>';
        response.data.products.forEach(product => {
            const option = document.createElement('option');
            option.value = product;
            option.textContent = product;
            select.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading products:', error);
    }
}

// Load KPIs
async function loadKPIs() {
    try {
        const response = await axios.get(`/api/v1/kpis/${sessionId}`);
        const kpis = response.data;
        
        document.getElementById('totalSales').textContent = formatNumber(kpis.total_sales);
        document.getElementById('forecastAccuracy').textContent = `${kpis.forecast_accuracy}%`;
        document.getElementById('serviceLevel').textContent = `${kpis.service_level}%`;
        document.getElementById('stockoutRisk').textContent = kpis.stockout_incidents;
        
    } catch (error) {
        console.error('Error loading KPIs:', error);
    }
}

// Run forecast
async function runForecast() {
    const product = document.getElementById('productSelect').value;
    const model = document.getElementById('modelSelect').value;
    const forecastDays = parseInt(document.getElementById('forecastDays').value);
    
    if (!product) {
        showStatus('Please select a product', 'error');
        return;
    }
    
    currentProduct = product;
    
    try {
        showStatus('Generating forecast...', 'info');
        
        const response = await axios.post('/api/v1/forecast', {
            session_id: sessionId,
            product_id: product,
            model: model,
            forecast_days: forecastDays
        });
        
        forecastData = response.data;
        
        // Plot forecast
        plotForecast(forecastData);
        
        // Display metrics
        if (forecastData.metrics) {
            displayForecastMetrics(forecastData.metrics);
        }
        
        // Run inventory optimization
        await optimizeInventory();
        
        showStatus('Forecast generated successfully!', 'success');
        
    } catch (error) {
        showStatus(`Error: ${error.response?.data?.detail || error.message}`, 'error');
    }
}

// Plot forecast chart
function plotForecast(data) {
    const historicalTrace = {
        x: data.historical_dates || [],
        y: data.historical_values || [],
        type: 'scatter',
        mode: 'lines',
        name: 'Historical',
        line: { color: 'rgb(31, 119, 180)' }
    };
    
    const forecastTrace = {
        x: data.dates,
        y: data.forecast,
        type: 'scatter',
        mode: 'lines',
        name: 'Forecast',
        line: { color: 'rgb(255, 127, 14)', dash: 'dot' }
    };
    
    const lowerBound = {
        x: data.dates,
        y: data.lower,
        type: 'scatter',
        mode: 'lines',
        name: 'Lower Bound',
        line: { width: 0 },
        showlegend: false
    };
    
    const upperBound = {
        x: data.dates,
        y: data.upper,
        type: 'scatter',
        mode: 'lines',
        name: 'Upper Bound',
        fill: 'tonexty',
        fillcolor: 'rgba(255, 127, 14, 0.2)',
        line: { width: 0 },
        showlegend: false
    };
    
    const layout = {
        title: `Demand Forecast - $${currentProduct} ($${data.model.toUpperCase()})`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Sales' },
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('forecastChart', [historicalTrace, lowerBound, upperBound, forecastTrace], layout);
}

// Display forecast metrics
function displayForecastMetrics(metrics) {
    const metricsHtml = `
        <div class="bg-gray-50 p-4 rounded-md">
            <p class="text-sm text-gray-600">RMSE</p>
            <p class="text-xl font-bold">${metrics.rmse.toFixed(2)}</p>
        </div>
        <div class="bg-gray-50 p-4 rounded-md">
            <p class="text-sm text-gray-600">MAE</p>
            <p class="text-xl font-bold">${metrics.mae.toFixed(2)}</p>
        </div>
        <div class="bg-gray-50 p-4 rounded-md">
            <p class="text-sm text-gray-600">MAPE</p>
            <p class="text-xl font-bold">${metrics.mape.toFixed(1)}%</p>
        </div>
        <div class="bg-gray-50 p-4 rounded-md">
            <p class="text-sm text-gray-600">R²</p>
            <p class="text-xl font-bold">${metrics.r2.toFixed(3)}</p>
        </div>
    `;
    
    document.getElementById('forecastMetrics').innerHTML = metricsHtml;
}

// Optimize inventory
async function optimizeInventory() {
    if (!forecastData) return;
    
    try {
        const response = await axios.post('/api/v1/optimize-inventory', {
            forecast_data: forecastData,
            lead_time_days: 7,
            holding_cost_rate: 0.2,
            stockout_cost_rate: 0.5
        });
        
        const inventoryData = response.data;
        
        // Update inventory metrics
        document.getElementById('safetyStock').textContent = Math.round(inventoryData.safety_stock);
        document.getElementById('reorderPoint').textContent = Math.round(inventoryData.reorder_point);
        document.getElementById('eoq').textContent = Math.round(inventoryData.eoq);
        
        // Plot inventory levels
        plotInventoryLevels(inventoryData);
        
        // Display recommendations
        displayRecommendations(inventoryData.recommendations);
        
    } catch (error) {
        console.error('Error optimizing inventory:', error);
    }
}

// Plot inventory levels
function plotInventoryLevels(data) {
    const inventoryTrace = {
        x: forecastData.dates,
        y: data.inventory_levels,
        type: 'scatter',
        mode: 'lines',
        name: 'Inventory Level',
        line: { color: 'rgb(44, 160, 44)' }
    };
    
    const demandTrace = {
        x: forecastData.dates,
        y: forecastData.forecast,
        type: 'scatter',
        mode: 'lines',
        name: 'Forecasted Demand',
        line: { color: 'rgb(31, 119, 180)' }
    };
    
    const reorderLine = {
        x: forecastData.dates,
        y: Array(forecastData.dates.length).fill(data.reorder_point),
        type: 'scatter',
        mode: 'lines',
        name: 'Reorder Point',
        line: { color: 'rgb(255, 127, 14)', dash: 'dash' }
    };
    
    const layout = {
        title: 'Inventory Optimization',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Units' },
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('inventoryChart', [inventoryTrace, demandTrace, reorderLine], layout);
}

// Display recommendations
function displayRecommendations(recommendations) {
    let html = '<div class="space-y-4">';
    
    recommendations.forEach(rec => {
        const colorClass = rec.type === 'critical' ? 'red' : 
                          rec.type === 'warning' ? 'yellow' : 'blue';
        
        html += `
            <div class="border-l-4 border-${colorClass}-500 bg-${colorClass}-50 p-4 rounded-md">
                <h4 class="font-semibold text-${colorClass}-800">${rec.title}</h4>
                <p class="text-${colorClass}-700 mt-1">${rec.description}</p>
                <div class="mt-2">
                    <span class="text-sm font-medium text-${colorClass}-800">Action: </span>
                    <span class="text-sm text-${colorClass}-700">${rec.action}</span>
                </div>
                <div class="mt-1">
                    <span class="text-sm font-medium text-${colorClass}-800">Impact: </span>
                    <span class="text-sm text-${colorClass}-700">${rec.impact}</span>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    document.getElementById('recommendations').innerHTML = html;
}

// Compare models
async function compareModels() {
    if (!currentProduct) {
        showStatus('Please select a product first', 'error');
        return;
    }
    
    try {
        showStatus('Comparing models...', 'info');
        
        const response = await axios.post('/api/v1/compare-models', {
            session_id: sessionId,
            product_id: currentProduct,
            models: ['prophet', 'arima', 'xgboost'],
            forecast_days: 30
        });
        
        plotModelComparison(response.data);
        displayModelMetrics(response.data);
        
        showStatus('Model comparison complete!', 'success');
        
    } catch (error) {
        showStatus(`Error: ${error.response?.data?.detail || error.message}', 'error');
    }
}

// Plot model comparison
function plotModelComparison(models) {
    const traces = [];
    const colors = {
        prophet: 'rgb(31, 119, 180)',
        arima: 'rgb(255, 127, 14)',
        xgboost: 'rgb(44, 160, 44)',
        lstm: 'rgb(214, 39, 40)'
    };
    
    Object.entries(models).forEach(([modelName, data]) => {
        if (!data.error && data.forecast) {
            traces.push({
                x: data.dates,
                y: data.forecast,
                type: 'scatter',
                mode: 'lines',
                name: modelName.toUpperCase(),
                line: { color: colors[modelName] }
            });
        }
    });
    
    const layout = {
        title: 'Model Comparison',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Forecasted Sales' },
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('comparisonChart', traces, layout);
}

// Display model metrics comparison
function displayModelMetrics(models) {
    let html = '<div class="overflow-x-auto"><table class="min-w-full divide-y divide-gray-200">';
    html += '<thead class="bg-gray-50"><tr>';
    html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>';
    html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">RMSE</th>';
    html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">MAE</th>';
    html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">MAPE (%)</th>';
    html += '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R²</th>';
    html += '</tr></thead><tbody class="bg-white divide-y divide-gray-200">';
    
    Object.entries(models).forEach(([modelName, data]) => {
        if (!data.error && data.metrics) {
            html += '<tr>';
            html += `<td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${modelName.toUpperCase()}</td>`;
            html += `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${data.metrics.rmse.toFixed(2)}</td>`;
            html += `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${data.metrics.mae.toFixed(2)}</td>`;
            html += `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${data.metrics.mape.toFixed(1)}</td>`;
            html += `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${data.metrics.r2.toFixed(3)}</td>`;
            html += '</tr>';
        }
    });
    
    html += '</tbody></table></div>';
    document.getElementById('modelMetrics').innerHTML = html;
}

// Run scenario analysis
async function runScenarios() {
    if (!currentProduct) {
        showStatus('Please select a product first', 'error');
        return;
    }
    
    const promotionLift = parseFloat(document.getElementById('promotionLift').value) / 100;
    const seasonalityFactor = parseFloat(document.getElementById('seasonalityFactor').value);
    
    try {
        showStatus('Running scenario analysis...', 'info');
        
        const response = await axios.post('/api/v1/scenario-analysis', {
            session_id: sessionId,
            product_id: currentProduct,
            scenarios: [
                { name: 'Baseline', promotion_lift: 0, seasonality_factor: 1 },
                { name: 'Promotion', promotion_lift: promotionLift, seasonality_factor: 1 },
                { name: 'Seasonal Peak', promotion_lift: 0, seasonality_factor: seasonalityFactor },
                { name: 'Combined', promotion_lift: promotionLift, seasonality_factor: seasonalityFactor }
            ]
        });
        
        plotScenarioResults(response.data);
        showStatus('Scenario analysis complete!', 'success');
        
    } catch (error) {
        showStatus(`Error: ${error.response?.data?.detail || error.message}', 'error');
    }
}

// Plot scenario results
function plotScenarioResults(scenarios) {
    const traces = [];
    const colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)'];
    let colorIndex = 0;
    
    Object.entries(scenarios).forEach(([scenarioName, data]) => {
        traces.push({
            x: data.forecast.dates,
            y: data.forecast.forecast,
            type: 'scatter',
            mode: 'lines',
            name: scenarioName,
            line: { color: colors[colorIndex % colors.length] }
        });
        colorIndex++;
    });
    
    const layout = {
        title: 'Scenario Analysis',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Forecasted Sales' },
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('scenarioResults', traces, layout);
}

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.add('hidden');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active', 'text-blue-600', 'border-blue-600');
        button.classList.add('text-gray-500', 'border-transparent');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}Tab`).classList.remove('hidden');
    
    // Add active class to selected button
    event.target.classList.add('active', 'text-blue-600', 'border-blue-600');
    event.target.classList.remove('text-gray-500', 'border-transparent');
    
    // Run specific actions based on tab
    if (tabName === 'comparison' && currentProduct) {
        compareModels();
    }
}

// Download report
async function downloadReport(format) {
    if (!sessionId) {
        showStatus('No data available for download', 'error');
        return;
    }
    
    try {
        window.location.href = `/api/v1/download-report/${sessionId}?format=${format}`;
    } catch (error) {
        showStatus(`Error downloading report: ${error.message}', 'error');
    }
}

// Utility functions
function showStatus(message, type) {
    const statusDiv = document.getElementById('uploadStatus');
    const colorClass = type === 'error' ? 'text-red-600' : 
                      type === 'success' ? 'text-green-600' : 'text-blue-600';
    
    statusDiv.innerHTML = `<p class="${colorClass} font-medium">${message}</p>`;
    
    if (type !== 'info') {
        setTimeout(() => {
            statusDiv.innerHTML = '';
        }, 5000);
    }
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(Math.round(num));
}

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners
    document.getElementById('fileInput').addEventListener('change', function(e) {
        const fileName = e.target.files[0]?.name;
        if (fileName) {
            showStatus(`Selected: ${fileName}`, 'info');
        }
    });
});
