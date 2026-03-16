import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from prepare_data import load_and_preprocess_data

def main():
    print("Loading data...")
    df = load_and_preprocess_data()
    
    print("Preparing baseline model...")
    # Binary target
    df['TARGET'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    
    # Drop original target 
    X = df.drop(columns=['ARRIVAL_DELAY', 'TARGET'])
    y = df['TARGET']
    
    # Convert all string/objects to categorical for XGBoost
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'string':
            X[col] = X[col].astype('category')
            
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost baseline...")
    model = XGBClassifier(enable_categorical=True, tree_method='hist', random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"Baseline ROC-AUC: {roc_auc:.4f}")
    print(f"Baseline PR-AUC: {pr_auc:.4f}")
    
    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(importances.head(10))
    
    plt.figure(figsize=(10, 6))
    importances.head(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances (Baseline)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('baseline_feature_importances.png')
    print("Saved feature importances plot to 'baseline_feature_importances.png'.")

if __name__ == "__main__":
    main()
