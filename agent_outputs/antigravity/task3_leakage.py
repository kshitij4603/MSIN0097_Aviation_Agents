import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from prepare_data import load_and_preprocess_data

"""
Reasoning to address data leakage:
If the model must predict flight delays at least 24 hours *before* scheduled departure, we cannot use 
any features that describe events occurring on the day of the flight. 
Features such as DEPARTURE_TIME, DEPARTURE_DELAY, TAXI_OUT, WHEELS_OFF, ELAPSED_TIME, AIR_TIME, 
WHEELS_ON, TAXI_IN, ARRIVAL_TIME, DIVERTED, CANCELLED, and all specific delay variables 
(AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY, WEATHER_DELAY) represent 
"future" information that wouldn't be known a day prior (data leakage). 
TAIL_NUMBER is also removed as precise aircraft assignment can dynamically change close to departure.
"""

def main():
    print("Loading data...")
    df = load_and_preprocess_data()
    
    # Binary target
    df['TARGET'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    
    # Identify leaky columns violating the 24-hour temporal constraint
    leaky_columns = [
        'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 
        'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN', 'ARRIVAL_TIME', 
        'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 
        'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY',
        'TAIL_NUMBER' 
    ]
    
    leaky_columns = [col for col in leaky_columns if col in df.columns]
    
    X = df.drop(columns=['ARRIVAL_DELAY', 'TARGET'] + leaky_columns)
    y = df['TARGET']
    
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'string':
            X[col] = X[col].astype('category')
            
    print("Splitting data (No Leakage)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost (No Leakage)...")
    model = XGBClassifier(enable_categorical=True, tree_method='hist', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"Non-Leaky ROC-AUC: {roc_auc:.4f}")
    print(f"Non-Leaky PR-AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    main()
