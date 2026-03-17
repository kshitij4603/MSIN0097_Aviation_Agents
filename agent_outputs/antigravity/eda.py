import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    base_dir = r"c:\Users\Media\Desktop\Predictive_group_coursework_data"
    flights_path = os.path.join(base_dir, "flights.csv")
    airlines_path = os.path.join(base_dir, "airlines.csv")
    output_dir = os.path.join(base_dir, "agent_outputs", "antigravity")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading datasets for EDA...")
    flights = pd.read_csv(flights_path, low_memory=False)
    airlines = pd.read_csv(airlines_path)
    
    # Drop cancelled flights as they are not valid for predicting delayed arrivals
    if 'CANCELLED' in flights.columns:
        flights = flights[flights['CANCELLED'] == 0]
        
    # Sample 500,000 rows
    if len(flights) > 500000:
        flights = flights.sample(n=500000, random_state=42)
        
    print(f"Dataset shape after filtering and sampling: {flights.shape}")
    
    # Merge with airlines to get descriptive names
    if 'AIRLINE' in flights.columns and 'IATA_CODE' in airlines.columns:
        flights = flights.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')
        if 'AIRLINE_y' in flights.columns:
            flights['AIRLINE_NAME'] = flights['AIRLINE_y']
        elif 'AIRLINE_NAME' not in flights.columns and 'AIRLINE' in airlines.columns:
            flights['AIRLINE_NAME'] = flights['AIRLINE']
        else:
            flights['AIRLINE_NAME'] = flights['AIRLINE_x']
    else:
        flights['AIRLINE_NAME'] = flights['AIRLINE']

    # Drop missing delays (remaining missing are usually diverted flights)
    flights = flights.dropna(subset=['ARRIVAL_DELAY'])
    
    # Create the binary target
    flights['DELAYED_15'] = (flights['ARRIVAL_DELAY'] > 15).astype(int)
    
    # Feature Engineering (strictly 24-hour advance information)
    flights['SCHED_DEP_HOUR'] = flights['SCHEDULED_DEPARTURE'] // 100
    
    # Set plotting aesthetics
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Delay Rate by Airline
    plt.figure(figsize=(12, 6))
    airline_delay = flights.groupby('AIRLINE_NAME')['DELAYED_15'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=airline_delay, x='DELAYED_15', y='AIRLINE_NAME', palette='viridis')
    plt.title('Proportion of Flights Delayed > 15 Mins by Airline')
    plt.xlabel('Delay Rate')
    plt.ylabel('Airline')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_plot1_airline_delays.png'))
    plt.close()
    
    # Plot 2: Delay Rate by Scheduled Departure Hour
    plt.figure(figsize=(12, 6))
    hour_delay = flights.groupby('SCHED_DEP_HOUR')['DELAYED_15'].mean().reset_index()
    sns.lineplot(data=hour_delay, x='SCHED_DEP_HOUR', y='DELAYED_15', marker='o', color='crimson', linewidth=2)
    plt.title('Proportion of Flights Delayed > 15 Mins by Scheduled Departure Hour')
    plt.xlabel('Scheduled Departure Hour (0-23)')
    plt.ylabel('Delay Rate')
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_plot2_hour_delays.png'))
    plt.close()
    
    # Plot 3: Delay Rate by Day of Week vs Month (Heatmap)
    plt.figure(figsize=(10, 8))
    pivot = flights.pivot_table(index='DAY_OF_WEEK', columns='MONTH', values='DELAYED_15', aggfunc='mean')
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.3f', linewidths=.5)
    plt.title('Delay Rate Heatmap: Day of Week vs Month')
    plt.xlabel('Month')
    plt.ylabel('Day of Week (1=Monday)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_plot3_heatmap.png'))
    plt.close()
    
    print("EDA plots generated successfully.")

if __name__ == "__main__":
    main()

''' 
BUSINESS INSIGHT:
1. Compounding Temporal Vulnerability: Delay risks rise monotonically as the scheduled departure hour increases. Early morning flights enjoy the lowest delay likelihoods because disruptions (e.g., late aircraft chaining, crew swaps, airspace congestion) stack as the day progresses.
2. Carrier Execution Disparity: Distinct airlines operate at structurally different baseline delay rates. Controlling for airline identity provides significant signal of expected reliability.
3. Seasonality Calendar Effects: Interactions between Month and Day of Week highlight peak periods of travel stress (e.g., specific holiday months vs off-peak mid-week days).

Actionable Conclusion: Deploying models on variables strictly known 24-hours prior—Airline, Scheduled Departure Hour, and Calendar variables—provides robust, non-leaky signals that directly exploit these operational patterns.
'''
