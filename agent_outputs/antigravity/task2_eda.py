import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'antigravity'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'flights.csv'

print('Loading', DATA_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)

# Filter cancellations and diversions if columns exist
if 'CANCELLED' in df.columns:
    df = df[df['CANCELLED'] == 0]
if 'DIVERTED' in df.columns:
    df = df[df['DIVERTED'] == 0]

# Calculate target if not present
if 'DELAY_15' not in df.columns and 'ARRIVAL_DELAY' in df.columns:
    df['DELAY_15'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

# Summary
summary = df.describe(include='all')
summary.to_csv(OUTPUT_DIR / 'task2_eda_summary.csv', index=True)

# Airline delay analysis
if 'AIRLINE' in df.columns and 'DELAY_15' in df.columns:
    delay_by_airline = df.groupby('AIRLINE')['DELAY_15'].mean().sort_values(ascending=False).reset_index()
    delay_by_airline.to_csv(OUTPUT_DIR / 'task2_delay_by_airline.csv', index=False)

print('Task2 EDA complete')
