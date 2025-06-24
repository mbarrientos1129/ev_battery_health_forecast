#EV Battery Health Forecasting

**Overview:** Predict the battery State of Health from telematics and weather.

## Structure 
- `data/raw/` - original CSVs
- `data/processed/` - cleaned data
- `notebooks/` - Jupyter analyses
- `src/` - data prep and modeling modules
- `models/` - saved PyTorch models
- `reports/` - charts and write-ups

##Setup
```bash 
git clone https://github.com/mbarrientos1129/ev_battery_health_forecast
cd ev_battery_health_forecast
pip install -r requirements.txt