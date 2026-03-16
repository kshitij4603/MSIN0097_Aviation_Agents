import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from prepare_data import load_and_preprocess_data

def main():
    print("Loading data...")
    df = load_and_preprocess_data()
    
    df['TARGET'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    
    # Dropping the leaky variables
    leaky_columns = [
        'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 
        'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN', 'ARRIVAL_TIME', 
        'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 
        'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY',
        'TAIL_NUMBER'
    ]
    leaky_columns = [col for col in leaky_columns if col in df.columns]
    
    print("Engineering strict 24-hour advance temporal & routing features...")
    # Feature 1: Route identifier (concatenating Origin and Destination into a single string)
    df['ROUTE'] = df['ORIGIN_AIRPORT'].astype(str) + "_" + df['DESTINATION_AIRPORT'].astype(str)
    
    # Feature 2: Scheduled Departure Hour (extracting hour from HHMM temporal integer representation)
    df['SCHED_DEP_HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
    
    # Feature 3: Is Weekend Profile Formatted (Binary variable tracking heavy travel routing days)
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([6, 7]).astype(int)
    
    X = df.drop(columns=['ARRIVAL_DELAY', 'TARGET'] + leaky_columns)
    y = df['TARGET']
    
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'string':
            X[col] = X[col].astype('category')
            
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost (Enhanced Features)...")
    model = XGBClassifier(enable_categorical=True, tree_method='hist', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    print(f"Enhanced Model ROC-AUC: {roc_auc:.4f}")
    print(f"Enhanced Model PR-AUC: {pr_auc:.4f}")
    
    print("\nTop Features including engineered temporal variables:")
    print(importances.head(10))

if __name__ == "__main__":
    main()
