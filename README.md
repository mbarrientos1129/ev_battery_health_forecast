#EV Battery Health Forecasting

**Overview:** Predict the battery State of Health from telematics and weather.

## Structure 
- `data/raw/` - original CSVs
- `data/processed/` - cleaned data
- `notebooks/` - Jupyter analyses
- `src/` - data prep and modeling modules
- `models/` - saved PyTorch models
- `reports/` - charts and write-ups

## Setup
```bash 
git clone https://github.com/mbarrientos1129/ev_battery_health_forecast
cd ev_battery_health_forecast
pip install -r requirements.txt
```
## Step 2: Data Ingestion & Cleaning
We take our raw EV metrics file (`data/raw/ev_metrics.csv`) and turn it into an analysis-ready table:

1. **Load the CSV**  
   ```python
   df = pd.read_csv('data/raw/ev_metrics.csv')
   ```
2. **Coerce numeric fields**
    ```python
    df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
    df['ambient_temperature'] = pd.to_numeric(df['ambient_temperature'], errors='coerce')
    ```
3. **Drop invalid rows**
    ```python
    df = df.dropna(subset=['Capacity'])
    ```
4. **Compute State-of-Health (SoH)**
    ```python
    max_cap = df['Capacity'].max()
    df['SOH'] = df['Capacity'] / max_cap * 100
    ```
5. **Parse timestamps**
    ```python
    def parse_start_time(s): …
    df['timestamp'] = df['start_time'].apply(parse_start_time)
    ```
6. **Filter to discharge cycles**
    ```python
    df = df[df['type']=='discharge']
    ```
7. **Result**
    ```python
    We end up with a DataFrame containing:['timestamp','Capacity','SOH','ambient_temperature','battery_id','test_id'].
    ```


