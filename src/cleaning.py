import pandas as pd
from datetime import datetime

def parse_start_time(ts_str: str) -> datetime:
    parts = ts_str.strip('[]').split()
    nums  = [int(float(p)) for p in parts[:6]]
    return datetime(*nums)

def load_and_clean(raw_ev_path: str) -> pd.DataFrame:
    # 1) Load CSV
    df = pd.read_csv(raw_ev_path)

    # 2) Coerce numeric columns
    df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
    df['ambient_temperature'] = pd.to_numeric(df['ambient_temperature'], errors='coerce')

    # 3) Drop rows where Capacity is missing (you could also drop ambient_temperature if you need it)
    df = df.dropna(subset=['Capacity'])

    # 4) Compute SOH
    max_cap = df['Capacity'].max()
    df['SOH'] = df['Capacity'] / max_cap * 100

    # 5) Parse timestamp
    df['timestamp'] = df['start_time'].apply(parse_start_time)

    # 6) Filter to discharge cycles
    df = df[df['type'] == 'discharge']

    # 7) Select and return
    return df[['timestamp', 'Capacity', 'SOH', 'ambient_temperature', 'battery_id', 'test_id']]
