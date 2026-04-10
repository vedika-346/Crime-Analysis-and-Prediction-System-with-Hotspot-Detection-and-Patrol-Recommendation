import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_file  = os.path.join(BASE_DIR, "data", "crime.csv")
output_file = os.path.join(BASE_DIR, "data", "cleaned_crime.csv")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading dataset...")
data = pd.read_csv(input_file)
print(f"  Raw rows: {len(data):,}")

# Print all unique crime types so you can see exactly what labels exist
print("\n  All unique Primary Types in raw data:")
print(data['Primary Type'].value_counts().to_string())

# Reproducible sample — increase to 100k if RAM allows
data = data.sample(50_000, random_state=42)

# ── Column selection & rename ─────────────────────────────────────────────────
columns_needed = [
    'Date', 'Primary Type', 'Description',
    'Community Area', 'Block', 'Ward',
    'Latitude', 'Longitude',
    'Arrest', 'Domestic'
]
data = data[[c for c in columns_needed if c in data.columns]]

data.rename(columns={
    'Primary Type':     'Crime_Type',
    'Description':      'Crime_Description',
    'Community Area':   'Community_Area',
}, inplace=True)

# ── Clean ─────────────────────────────────────────────────────────────────────
data.dropna(subset=['Date', 'Crime_Type', 'Latitude', 'Longitude', 'Community_Area'],
            inplace=True)

# Lat/lon sanity bounds (Chicago bounding box)
data = data[
    data['Latitude'].between(41.6, 42.1) &
    data['Longitude'].between(-87.95, -87.5)
]

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.dropna(subset=['Date'], inplace=True)

# ── Standardise crime type labels ─────────────────────────────────────────────
# Strip whitespace and uppercase so labels are consistent
data['Crime_Type'] = data['Crime_Type'].str.strip().str.upper()

# ── Feature engineering ───────────────────────────────────────────────────────
data['Year']      = data['Date'].dt.year
data['Month']     = data['Date'].dt.month
data['Day']       = data['Date'].dt.day
data['Hour']      = data['Date'].dt.hour
data['DayOfWeek'] = data['Date'].dt.dayofweek   # 0 = Mon
data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)

def time_bucket(h):
    if   0 <= h < 6:   return 'Night'
    elif 6 <= h < 12:  return 'Morning'
    elif 12 <= h < 18: return 'Afternoon'
    else:              return 'Evening'

data['Time_Bucket'] = data['Hour'].apply(time_bucket)

data['Hour_Sin']  = np.sin(2 * np.pi * data['Hour']  / 24)
data['Hour_Cos']  = np.cos(2 * np.pi * data['Hour']  / 24)
data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)

data['Community_Area'] = data['Community_Area'].astype(int)

for col in ['Arrest', 'Domestic']:
    if col in data.columns:
        data[col] = data[col].astype(bool).astype(int)

# ── NO grouping of rare types here — model.py handles all remapping ───────────
# Only drop truly rare types (< 100 samples) that no model can learn
top_crimes = data['Crime_Type'].value_counts()
keep = top_crimes[top_crimes >= 100].index
data['Crime_Type'] = data['Crime_Type'].where(data['Crime_Type'].isin(keep), 'OTHER')

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
data.to_csv(output_file, index=False)

print("\nEDA DONE")
print(f"  Saved {len(data):,} rows → {output_file}")
print(f"  Crime types after EDA: {data['Crime_Type'].nunique()}")
print(data['Crime_Type'].value_counts().to_string())
print(data.head(3).to_string())