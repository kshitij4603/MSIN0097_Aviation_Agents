import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'flights.csv'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'antigravity'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data from {DATA_PATH}")
flights = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Initial rows: {len(flights):,}")

if len(flights) > 500000:
    flights = flights.sample(n=500000, random_state=42)
    print(f"Sampled to {len(flights):,} rows for performance")

if 'CANCELLED' in flights.columns:
    flights = flights[flights['CANCELLED'] == 0]
if 'DIVERTED' in flights.columns:
    flights = flights[flights['DIVERTED'] == 0]

flights = flights.dropna(subset=['ARRIVAL_DELAY'])
flights['DELAYED_15'] = (flights['ARRIVAL_DELAY'] > 15).astype(int)
print(f"After cleanup rows: {len(flights):,}")

arrival = flights['ARRIVAL_DELAY'].astype(float)
delay_desc = {
    'count': arrival.count(),
    'mean': arrival.mean(),
    'std': arrival.std(),
    'min': arrival.min(),
    '25%': arrival.quantile(0.25),
    '50%': arrival.quantile(0.50),
    '75%': arrival.quantile(0.75),
    '90%': arrival.quantile(0.90),
    '95%': arrival.quantile(0.95),
    '99%': arrival.quantile(0.99),
    'max': arrival.max(),
    'skewness': arrival.skew(),
    'kurtosis': arrival.kurtosis(),
}
delay_summary = pd.DataFrame([delay_desc])
delay_summary.to_csv(OUTPUT_DIR / 'eda_delay_distribution.csv', index=False)
print('Saved eda_delay_distribution.csv')

numeric_cols = flights.select_dtypes(include=[np.number]).columns.tolist()
if 'ARRIVAL_DELAY' in numeric_cols:
    numeric_cols.remove('ARRIVAL_DELAY')

corr_matrix = flights[numeric_cols + ['ARRIVAL_DELAY']].corr(method='pearson')
arr_delay_corr = corr_matrix['ARRIVAL_DELAY'].drop('ARRIVAL_DELAY')
arr_delay_corr_abs = arr_delay_corr.abs().sort_values(ascending=False)
top5 = arr_delay_corr_abs.head(5).index.tolist()
corr_summary = arr_delay_corr.loc[top5].reset_index()
corr_summary.columns = ['feature', 'correlation_with_arrival_delay']
corr_summary.to_csv(OUTPUT_DIR / 'eda_correlations.csv', index=False)
print('Saved eda_correlations.csv')

if all(col in flights.columns for col in ['AIRLINE', 'DAY_OF_WEEK']):
    airline_day = flights.pivot_table(index='AIRLINE', columns='DAY_OF_WEEK', values='DELAYED_15', aggfunc='mean')
    airline_day.columns = [f'DAY_{int(c)}' for c in airline_day.columns]
    airline_day = airline_day.fillna(0).round(4)
    if 'DAY_1' in airline_day.columns:
        airline_day = airline_day.sort_values(by='DAY_1', ascending=False)
    airline_day.to_csv(OUTPUT_DIR / 'eda_airline_day_pivot.csv')
    print('Saved eda_airline_day_pivot.csv')
else:
    print('AIRLINE or DAY_OF_WEEK missing; skipping airline-day pivot')

airport_agg = flights.groupby('ORIGIN_AIRPORT').agg(mean_arrival_delay=('ARRIVAL_DELAY', 'mean'), delay_rate=('DELAYED_15', 'mean')).dropna()
airport_agg['severity'] = airport_agg['delay_rate'] * airport_agg['mean_arrival_delay']

airline_agg = flights.groupby('AIRLINE').agg(mean_arrival_delay=('ARRIVAL_DELAY', 'mean'), delay_rate=('DELAYED_15', 'mean')).dropna()
airline_agg['severity'] = airline_agg['delay_rate'] * airline_agg['mean_arrival_delay']

airport_top = airport_agg.nlargest(10, 'severity').reset_index().assign(type='origin_airport')
airline_top = airline_agg.nlargest(10, 'severity').reset_index().assign(type='airline')

worst_offenders = pd.concat([
    airline_top[['AIRLINE', 'severity', 'type']].rename(columns={'AIRLINE': 'entity'}),
    airport_top[['ORIGIN_AIRPORT', 'severity', 'type']].rename(columns={'ORIGIN_AIRPORT': 'entity'})
], ignore_index=True)
worst_offenders = worst_offenders.sort_values('severity', ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 10))
colors = worst_offenders['type'].map({'airline': '#d62728', 'origin_airport': '#1f77b4'})
plt.barh(worst_offenders['entity'], worst_offenders['severity'], color=colors)
plt.gca().invert_yaxis()
plt.xlabel('Severity (delay rate * mean arrival delay)')
plt.title('Worst Offenders: Airlines and Origin Airports by Delay Severity')

if not worst_offenders.empty:
    top_value = worst_offenders['severity'].max()
    for idx, row in worst_offenders.iterrows():
        plt.text(row['severity'] + 0.02 * top_value, idx, f"{row['severity']:.2f}", va='center', fontsize=8)

from matplotlib.patches import Patch
patches = [Patch(color='#d62728', label='Airline'), Patch(color='#1f77b4', label='Origin Airport')]
plt.legend(handles=patches, loc='lower right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'eda_worst_offenders.png', dpi=150)
plt.close()
print('Saved eda_worst_offenders.png')

rec = ('Strategic recommendation: Focus corporate booking away from carriers and origin airports identified as top severity offenders, as they carry high systemic delay risk. ' 
       'Use the airline vs. day-of-week pivot to avoid the specific weekday windows where these operators spike. ' 
       'Adopt a proactive route-level selection policy based on the delay distribution and correlation analysis to reduce exposure to extreme arrival delays.')
print(rec)

with open(OUTPUT_DIR / 'task2_eda_recommendation.txt', 'w', encoding='utf-8') as f:
    f.write(rec + '\n')
print('Saved task2_eda_recommendation.txt')
