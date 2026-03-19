import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pathlib import Path
import os

def load_and_preprocess_data():
    # File paths (dynamic relative pathing)
    root_dir = Path(__file__).resolve().parents[2]  # MSIN0097_Aviation_Agents
    flights_path = root_dir / 'data' / 'flights.csv'  # ../../data/flights.csv from agent_outputs/antigravity
    airlines_path = root_dir.parent / 'airlines.csv'  # workspace root
    airports_path = root_dir.parent / 'airports.csv'  # workspace root

    try:
        # Load datasets
        print("Loading datasets...")
        flights = pd.read_csv(flights_path)
        airlines = pd.read_csv(airlines_path)
        airports = pd.read_csv(airports_path)

        # Merge Airlines
        print("Merging airline data...")
        flights = flights.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left', suffixes=('', '_y'))
        if 'IATA_CODE' in flights.columns:
            flights.drop(columns=['IATA_CODE'], inplace=True)
            
        # Merge Origin Airports
        print("Merging origin airport data...")
        flights = flights.merge(airports, left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left', suffixes=('', '_origin'))
        if 'IATA_CODE' in flights.columns:
            flights.drop(columns=['IATA_CODE'], inplace=True)
            
        # Merge Destination Airports
        print("Merging destination airport data...")
        flights = flights.merge(airports, left_on='DESTINATION_AIRPORT', right_on='IATA_CODE', how='left', suffixes=('', '_dest'))
        if 'IATA_CODE' in flights.columns:
            flights.drop(columns=['IATA_CODE'], inplace=True)

        # Identify missing values in ARRIVAL_DELAY and impute
        # Using median imputation as a statistically sound strategy for delays which are often skewed
        print("Imputing missing ARRIVAL_DELAY values...")
        imputer = SimpleImputer(strategy='median')
        flights['ARRIVAL_DELAY'] = imputer.fit_transform(flights[['ARRIVAL_DELAY']])

        # Cast all string/object columns to category types to save memory
        print("Optimizing memory usage...")
        object_cols = flights.select_dtypes(include=['object']).columns
        for col in object_cols:
            flights[col] = flights[col].astype('category')

        # Print DataFrame info to confirm memory optimization
        print("\n--- DataFrame Memory Optimization Info ---")
        flights.info(memory_usage='deep')
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Ensure 'flights.csv', 'airlines.csv', and 'airports.csv' are in the working directory.")

if __name__ == "__main__":
    load_and_preprocess_data()
