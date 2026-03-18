import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Relative path requirement
DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'flights.csv'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'antigravity'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load values
print(f"Loading data from {DATA_PATH}")
flights = pd.read_csv(DATA_PATH, low_memory=False, nrows=300000)
print(f"Loaded {len(flights):,} rows (nrows=300000 sample)")

# If the dataset is still large, optionally downsample further
if len(flights) > 150_000:
    flights = flights.sample(n=150_000, random_state=42)
    print("Downsampled to 150000 rows for fast execution")

# Canonical cleaning
if 'CANCELLED' in flights.columns:
    flights = flights[flights['CANCELLED'] == 0]
if 'DIVERTED' in flights.columns:
    flights = flights[flights['DIVERTED'] == 0]

flights = flights.dropna(subset=['ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'DISTANCE'])
print(f"After cleaning: {len(flights):,} rows")

# Strong definition of delay: arrival delay > 15
flights['DELAYED_15'] = (flights['ARRIVAL_DELAY'] > 15).astype(int)

# Airline aggregate
airline_stats = (
    flights.groupby('AIRLINE')[['ARRIVAL_DELAY', 'DELAYED_15']]
    .mean()
    .rename(columns={'ARRIVAL_DELAY': 'mean_arrival_delay', 'DELAYED_15': 'delay_rate'})
    .reset_index()
)
airline_stats['severity'] = airline_stats['delay_rate'] * airline_stats['mean_arrival_delay']
airline_stats = airline_stats.sort_values('severity', ascending=False)

# Origin airport aggregate
airport_stats = (
    flights.groupby('ORIGIN_AIRPORT')[['ARRIVAL_DELAY', 'DELAYED_15']]
    .mean()
    .rename(columns={'ARRIVAL_DELAY': 'mean_arrival_delay', 'DELAYED_15': 'delay_rate'})
    .reset_index()
)
airport_stats['severity'] = airport_stats['delay_rate'] * airport_stats['mean_arrival_delay']
airport_stats = airport_stats.sort_values('severity', ascending=False)

# Select top offenders
top_airlines = airline_stats.head(10).assign(entity_type='airline').rename(columns={'AIRLINE':'entity'})
top_airports = airport_stats.head(10).assign(entity_type='origin_airport').rename(columns={'ORIGIN_AIRPORT':'entity'})

worst = pd.concat([top_airlines, top_airports], ignore_index=True)
worst = worst.sort_values('severity', ascending=False).reset_index(drop=True)

# Visualization
plt.figure(figsize=(14, 10))
colors = ['#d62728' if t == 'airline' else '#1f77b4' for t in worst['entity_type']]

plt.barh(worst['entity'], worst['severity'], color=colors, alpha=0.85)
plt.gca().invert_yaxis()
plt.title('Top 20 Worst Offenders of Arrival Delay Severity (Airline and Origin Airport)', fontsize=16)
plt.xlabel('Severity Score = Delay Rate * Mean Arrival Delay (m)')
plt.ylabel('Entity (Airline or Origin Airport)')
for i, (score, entity) in enumerate(zip(worst['severity'], worst['entity'])):
    plt.text(score + 0.03 * worst['severity'].max(), i, f"{score:.2f}", va='center', fontsize=9)

# legend
import matplotlib.patches as mpatches
airline_patch = mpatches.Patch(color='#d62728', label='Airline')
airport_patch = mpatches.Patch(color='#1f77b4', label='Origin Airport')
plt.legend(handles=[airline_patch, airport_patch], loc='lower right')
plt.tight_layout()
plot_path = OUTPUT_DIR / 'eda_worst_offenders.png'
plt.savefig(plot_path, dpi=150)
plt.close()

# Save top offenders table
worst.to_csv(OUTPUT_DIR / 'eda_worst_offenders_table.csv', index=False)

# Print recommendation
recommendation = (
    'Based on evidence, the worst offenders are the top airlines and origin airports with high combined delay rate and mean arrival delay. '
    'The agency should avoid routes where these entities dominate, as they contribute disproportionate client inconvenience and cost. '
    'Focus bookings on carriers/airports with lower severity scores and emphasize proactive contingency planning for the flagged offenders.'
)
print(recommendation)

# Persist recommendation text
with open(OUTPUT_DIR / 'task2_eda_recommendation.txt', 'w') as f:
    f.write(recommendation + '\n')

print(f"Saved plot to {plot_path}. Working directory: {OUTPUT_DIR}")
