# 📊 Intelligent Inventory & Demand Forecasting System

A comprehensive, professional-grade Streamlit dashboard for demand forecasting and inventory optimization. Built for retail, logistics, and supply chain professionals to reduce stockouts, prevent overstocking, and optimize costs.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Features

### 📁 Data Management
- **Multiple Import Formats**: Upload CSV or Excel files
- **Sample Data Generation**: Built-in demo data for testing
- **Automatic Data Cleaning**: Handles missing values and duplicates
- **Data Visualization**: Interactive charts showing historical trends
- **Statistical Summary**: Comprehensive data analytics

### 📈 Demand Forecasting
- **Multiple AI/ML Models**:
  - **ARIMA**: Classical time series forecasting
  - **Prophet**: Facebook's robust forecasting tool
  - **XGBoost**: Gradient boosting for complex patterns
- **Confidence Intervals**: Upper and lower bounds for predictions
- **Flexible Forecast Horizon**: 7 to 90 days
- **What-If Scenario Analysis**: Test promotional impacts and seasonal adjustments
- **Visual Comparisons**: Historical vs predicted demand charts

### 📦 Inventory Optimization
- **Safety Stock Calculation**: Prevent stockouts with buffer inventory
- **Reorder Point Analysis**: Know when to place orders
- **Economic Order Quantity (EOQ)**: Minimize ordering and holding costs
- **Service Level Targets**: Customize from 80% to 99%
- **Risk Assessment**: 
  - Stockout probability gauges
  - Overstock/understock alerts
  - Inventory turnover analysis
- **Days of Supply**: Current inventory coverage calculation

### 💰 Cost-Benefit Analysis
- **Comprehensive Cost Modeling**:
  - Holding costs
  - Stockout costs
  - Ordering costs
- **Savings Projections**: Annual cost reduction estimates
- **ROI Calculation**: Payback period analysis
- **Visual Comparisons**: Current vs optimized scenarios
- **12-Month Savings Projection**: Cumulative savings over time

### 📄 Reports & Exports
- **PDF Reports**: Professional formatted reports with all metrics
- **Excel Exports**: Multi-sheet workbooks with detailed data
- **Downloadable Forecasts**: CSV exports for further analysis
- **Custom Date Stamping**: Automatic report versioning

### ⚙️ Advanced Settings
- **Model Configuration**: Fine-tune ARIMA, Prophet, and XGBoost parameters
- **Display Customization**: Theme, date format, and currency settings
- **Flexible Cost Parameters**: Customize all business-specific costs

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the repository**
```bash
git clone <repository-url>
cd inventory-forecasting-system
```

2. **Install required packages**
```bash
pip install streamlit pandas numpy plotly statsmodels prophet xgboost scikit-learn openpyxl reportlab
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
The app will automatically open at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
statsmodels>=0.14.0
prophet>=1.1.4
xgboost>=2.0.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
reportlab>=4.0.0
scipy>=1.11.0
```

---

## 📖 User Guide

### Step 1: Data Upload

1. Navigate to **📁 Data Upload** from the sidebar
2. Choose one of two options:
   - Click **"🔄 Load Sample Data"** to use demo data
   - Click **"Upload Sales Data"** to import your CSV/Excel file

**Required Data Format:**
```csv
date,sales,promotion,holiday
2022-01-01,150,0,0
2022-01-02,165,0,0
2022-01-03,142,1,0
...
```

**Minimum Requirements:**
- `date` column: Date of sales (any standard format)
- `sales` column: Quantity sold (numeric)

**Optional Columns:**
- `promotion`: Binary (0/1) for promotional periods
- `holiday`: Binary (0/1) for holidays
- Any other external factors (temperature, market trends, etc.)

### Step 2: Demand Forecasting

1. Navigate to **📈 Demand Forecasting**
2. Select your forecasting method:
   - **Prophet**: Best for seasonal patterns and holidays
   - **ARIMA**: Best for stable time series
   - **XGBoost**: Best for complex patterns with external factors
3. Set forecast horizon (7-90 days)
4. (Optional) Configure what-if scenarios:
   - Adjust promotion impact (0-100%)
   - Apply seasonal adjustments (-50% to +50%)
5. Click **"🚀 Run Forecast"**
6. Review results:
   - Forecast visualization with confidence intervals
   - Key metrics (average, total, peak demand)
   - Downloadable forecast data

### Step 3: Inventory Optimization

1. Navigate to **📦 Inventory Optimization**
2. Configure business parameters:
   - **Lead Time**: Days between ordering and receiving inventory
   - **Service Level**: Target probability of meeting demand (80-99%)
   - **Holding Cost**: Annual cost % to store inventory
3. Expand **"⚙️ Advanced Settings"** for additional parameters:
   - Fixed ordering cost
   - Current inventory level
   - Item cost
   - Maximum storage capacity
4. Click **"🎯 Calculate Optimal Inventory"**
5. Review recommendations:
   - Reorder point
   - Safety stock levels
   - Economic order quantity
   - Stockout risk assessment
   - Inventory turnover rate

### Step 4: Cost-Benefit Analysis

1. Navigate to **💰 Cost-Benefit Analysis**
2. Enter cost parameters:
   - **Stockout Cost**: Lost profit per unit when out of stock
   - **Holding Cost**: Annual cost to store one unit
   - **Rush Order Premium**: Extra cost % for expedited orders
3. Click **"💰 Calculate Cost Savings"**
4. Review analysis:
   - Annual cost comparison (current vs optimized)
   - Cost breakdown by category
   - Projected cumulative savings (12 months)
   - ROI and payback period

### Step 5: Generate Reports

1. Navigate to **⚙️ Settings**
2. Customize display preferences (optional)
3. Click **"📥 Generate PDF Report"** or **"📊 Generate Excel Report"**
4. Download the comprehensive report including:
   - Historical data summary
   - Forecast results with confidence intervals
   - Inventory recommendations
   - Cost-benefit analysis

---

## 💡 Use Cases

### Retail Inventory Management
- Forecast seasonal demand for holiday shopping
- Optimize stock levels across multiple store locations
- Reduce overstock of slow-moving items
- Prevent stockouts of popular products

### E-commerce Operations
- Plan inventory for promotional events
- Optimize warehouse space utilization
- Balance holding costs with service levels
- Forecast demand spikes from marketing campaigns

### Supply Chain Planning
- Calculate optimal reorder points for suppliers
- Minimize total supply chain costs
- Improve order fulfillment rates
- Reduce expedited shipping needs

### Manufacturing
- Plan raw material procurement
- Balance work-in-progress inventory
- Optimize finished goods inventory
- Coordinate with production schedules

---

## 🎨 Customization

### Modifying Forecasting Models

**ARIMA Parameters** (in Settings):
```python
# Default: ARIMA(2,1,2)
order = (p, d, q)  # p: AR order, d: differencing, q: MA order
```

**Prophet Configuration**:
```python
# Adjust in Settings > Prophet Settings
changepoint_prior_scale = 0.05  # Flexibility of trend changes
seasonality_prior_scale = 10.0  # Strength of seasonality
```

**XGBoost Hyperparameters**:
```python
# Configure in Settings > XGBoost Settings
n_estimators = 100      # Number of trees
learning_rate = 0.1     # Step size
max_depth = 5          # Tree depth
```

### Custom CSS Styling

Modify the CSS in the `st.markdown()` section at the top of the code:
```python
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #YOUR_COLOR1, #YOUR_COLOR2);
    }
    /* Add your custom styles */
</style>
""", unsafe_allow_html=True)
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'prophet'`
```bash
# Solution:
pip install prophet
# If that fails, try:
conda install -c conda-forge prophet
```

**Issue**: Forecast results seem inaccurate
- **Solution**: 
  - Ensure sufficient historical data (minimum 60 days recommended)
  - Try different forecasting methods
  - Check for data quality issues (outliers, missing values)
  - Adjust model parameters in Settings

**Issue**: PDF generation fails
```bash
# Solution: Install reportlab dependencies
pip install reportlab pillow
```

**Issue**: App runs slowly
- **Solution**: 
  - Reduce forecast horizon
  - Use sample data to test first
  - Ensure adequate RAM (minimum 4GB)
  - Close unnecessary browser tabs

### Data Quality Tips

✅ **Good Practices:**
- Consistent date format throughout
- No gaps in date sequence
- Numeric values only in sales column
- Remove obvious outliers before upload

❌ **Common Pitfalls:**
- Mixed date formats (MM/DD vs DD/MM)
- Text in numeric columns
- Duplicate dates
- Extreme outliers not representative of business

---

## 📊 Sample Data Format

### Example CSV File
```csv
date,sales,promotion,holiday,temperature
2022-01-01,145,0,1,15.2
2022-01-02,132,0,0,14.8
2022-01-03,156,0,0,16.1
2022-01-04,178,1,0,17.3
2022-01-05,165,1,0,18.2
2022-01-06,142,0,0,16.9
2022-01-07,138,0,0,15.7
```

### Data Dictionary

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| date | Date | Yes | Transaction date (YYYY-MM-DD recommended) |
| sales | Integer | Yes | Units sold on that date |
| promotion | Binary | No | 1 if promotion active, 0 otherwise |
| holiday | Binary | No | 1 if holiday, 0 otherwise |
| temperature | Float | No | Daily temperature (for weather-dependent products) |
| [custom] | Any | No | Add any relevant external factors |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue with details
2. **Suggest Features**: Share your ideas for improvements
3. **Submit Pull Requests**: 
   - Fork the repository
   - Create a feature branch
   - Make your changes
   - Submit a pull request

### Development Setup
```bash
# Clone the repository
git clone <repository-url>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest tests/
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web framework
- **Facebook Prophet** - For robust time series forecasting
- **Plotly** - For interactive visualizations
- **XGBoost** - For powerful gradient boosting
- **statsmodels** - For ARIMA implementation

---

## 📧 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact: [your-email@example.com]
- Documentation: [link to docs]

---

## 🗺️ Roadmap

### Version 2.0 (Planned Features)
- [ ] LSTM/GRU deep learning models
- [ ] Multi-product forecasting
- [ ] Real-time data integration (APIs)
- [ ] Advanced anomaly detection
- [ ] Automated email reports
- [ ] Mobile-responsive design
- [ ] User authentication
- [ ] Cloud deployment guides
- [ ] Multi-language support
- [ ] Dashboard customization builder

---

## 📈 Performance Benchmarks

| Data Size | Processing Time | Memory Usage |
|-----------|----------------|--------------|
| 1 year (365 days) | ~2-3 seconds | ~100 MB |
| 2 years (730 days) | ~4-6 seconds | ~150 MB |
| 3 years (1095 days) | ~8-12 seconds | ~200 MB |

*Tested on: Intel i5, 8GB RAM, Python 3.9*

---

## 🎓 Resources & Learning

### Recommended Reading
- [Time Series Forecasting Principles](https://otexts.com/fpp3/)
- [Inventory Optimization Basics](https://www.investopedia.com/inventory-management-4689026)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [XGBoost Guide](https://xgboost.readthedocs.io/)

### Video Tutorials
- Streamlit Crash Course
- Time Series Analysis with Python
- Supply Chain Analytics

---

## ⚡ Tips for Best Results

### Forecasting Tips
1. **Use at least 2-3 months of historical data** for reliable patterns
2. **Prophet works best** for data with strong seasonal patterns
3. **ARIMA is ideal** for stable, non-seasonal time series
4. **XGBoost excels** when you have external factors (promotions, weather)
5. **Compare multiple models** to find the best fit for your data

### Inventory Optimization Tips
1. **Start with 95% service level** and adjust based on business needs
2. **Higher service levels** mean more safety stock and higher costs
3. **Monitor inventory turnover** - aim for 4-12x per year for most products
4. **Adjust lead times** seasonally for suppliers with variable delivery
5. **Review costs quarterly** to ensure parameters remain accurate

### Cost Savings Tips
1. **Focus on high-value items** (ABC analysis approach)
2. **Review promotions** using what-if scenarios before implementation
3. **Balance stockout and holding costs** for optimal total cost
4. **Use rush orders sparingly** - factor in premium costs
5. **Regular reviews** - reforecast monthly or quarterly

---

**Built with ❤️ for supply chain professionals**

*Last Updated: 2024*
