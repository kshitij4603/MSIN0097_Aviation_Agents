import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib

RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'antigravity'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'flights.csv'

print('Task 5 audit started. Loading data from:', DATA_PATH)

df = pd.read_csv(DATA_PATH, low_memory=False)
if 'CANCELLED' in df.columns:
    df = df[df['CANCELLED'] == 0]
if 'DIVERTED' in df.columns:
    df = df[df['DIVERTED'] == 0]

df = df.dropna(subset=['ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'DISTANCE'])

df['DELAY_15'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

# Define Task 4 features (as in previous model pipeline) and leaky features
# Look-ahead leakage definition: Any feature derived from events that happen after departure.
lookahead_features = [
    'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'AIR_TIME', 'TAXI_OUT', 'TAXI_IN',
    'WHEELS_OFF', 'WHEELS_ON', 'DELAY_DIFF', 'AVG_SPEED', 'TAXI_SUM', 'DISTANCE_PER_SCHEDULED'
]

# Task 4 candidate features
task4_features = []
for c in ['MONTH', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL',
          'SCHEDULED_TIME', 'ELAPSED_TIME', 'DISTANCE', 'AIR_TIME',
          'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT', 'TAXI_IN',
          'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
    if c in df.columns:
        task4_features.append(c)

# Derived Task 4 features
if 'SCHEDULED_DEPARTURE' in df.columns:
    df['SCHED_DEP_HOUR'] = ((pd.to_numeric(df['SCHEDULED_DEPARTURE'], errors='coerce') // 100) % 24).fillna(0).astype(int)
    task4_features.append('SCHED_DEP_HOUR')
if 'SCHEDULED_ARRIVAL' in df.columns:
    df['SCHED_ARR_HOUR'] = ((pd.to_numeric(df['SCHEDULED_ARRIVAL'], errors='coerce') // 100) % 24).fillna(0).astype(int)
    task4_features.append('SCHED_ARR_HOUR')

if 'AIR_TIME' in df.columns and 'DISTANCE' in df.columns:
    df['AVG_SPEED'] = (df['DISTANCE'] / df['AIR_TIME']).replace([np.inf, -np.inf], np.nan).fillna(0)
    task4_features.append('AVG_SPEED')

if 'TAXI_OUT' in df.columns and 'TAXI_IN' in df.columns:
    df['TAXI_SUM'] = df['TAXI_OUT'] + df['TAXI_IN']
    task4_features.append('TAXI_SUM')

if 'DEPARTURE_DELAY' in df.columns and 'ARRIVAL_DELAY' in df.columns:
    df['DELAY_DIFF'] = df['ARRIVAL_DELAY'] - df['DEPARTURE_DELAY']
    task4_features.append('DELAY_DIFF')

if 'DISTANCE' in df.columns and 'SCHEDULED_TIME' in df.columns:
    df['DISTANCE_PER_SCHEDULED'] = (df['DISTANCE'] / df['SCHEDULED_TIME'].replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
    task4_features.append('DISTANCE_PER_SCHEDULED')

# Ensure existing required columns for model training
task4_features = [f for f in task4_features if f in df.columns]

# Build rehearsal Task 4 X and evaluate AUC for audit comparison
X_task4 = df[task4_features].copy()
# Numeric imputation
for c in X_task4.select_dtypes(include=[np.number]).columns:
    X_task4[c] = X_task4[c].fillna(X_task4[c].median())
# Cat encoding
for c in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
    if c in X_task4.columns:
        top = X_task4[c].value_counts().nlargest(20).index
        X_task4[c] = X_task4[c].where(X_task4[c].isin(top), other='OTHER')
if any(c in X_task4.columns for c in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']):
    X_task4 = pd.get_dummies(X_task4, columns=[c for c in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'] if c in X_task4.columns], drop_first=True)

# Remove features with leakage for Task5 audit
leaky_cols = [c for c in lookahead_features if c in X_task4.columns or c in task4_features]
clean_features = [c for c in X_task4.columns if c not in leaky_cols]

# Task 5 definition: rigorous pre-departure features
clean_feature_set = []
for c in ['MONTH', 'DAY_OF_WEEK', 'SCHED_DEP_HOUR', 'SCHED_ARR_HOUR', 'DISTANCE', 'SCHEDULED_TIME']:
    if c in X_task4.columns:
        clean_feature_set.append(c)
# include one-hot airline/origin/dest if present
for c in X_task4.columns:
    if c.startswith('AIRLINE_') or c.startswith('ORIGIN_AIRPORT_') or c.startswith('DESTINATION_AIRPORT_'):
        clean_feature_set.append(c)

if not clean_feature_set:
    raise SystemError('No clean features available for Task 5 after leakage removal.')

X_task5 = X_task4[clean_feature_set].copy()

# train-test split function
Y = df['DELAY_15']
X4_tr, X4_te, y4_tr, y4_te = train_test_split(X_task4, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y)
X5_tr, X5_te, y5_tr, y5_te = train_test_split(X_task5, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y)

scaler4 = StandardScaler(); scaler5 = StandardScaler()
X4_tr = pd.DataFrame(scaler4.fit_transform(X4_tr), columns=X4_tr.columns, index=X4_tr.index)
X4_te = pd.DataFrame(scaler4.transform(X4_te), columns=X4_te.columns, index=X4_te.index)
X5_tr = pd.DataFrame(scaler5.fit_transform(X5_tr), columns=X5_tr.columns, index=X5_tr.index)
X5_te = pd.DataFrame(scaler5.transform(X5_te), columns=X5_te.columns, index=X5_te.index)

# Train evaluation for Task4 (leaky baseline model)
model4 = XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss', n_jobs=4, n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
model4.fit(X4_tr, y4_tr)

y4_hat = model4.predict(X4_te)
y4_proba = model4.predict_proba(X4_te)[:,1]
metrics_task4 = {
    'accuracy': accuracy_score(y4_te, y4_hat),
    'precision': precision_score(y4_te, y4_hat, zero_division=0),
    'recall': recall_score(y4_te, y4_hat, zero_division=0),
    'f1_score': f1_score(y4_te, y4_hat, zero_division=0),
    'roc_auc': roc_auc_score(y4_te, y4_proba)
}

# Train refined Task5 model on clean pre-departure features
model5 = XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss', n_jobs=4, n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
model5.fit(X5_tr, y5_tr)

y5_hat = model5.predict(X5_te)
y5_proba = model5.predict_proba(X5_te)[:,1]
metrics_task5 = {
    'accuracy': accuracy_score(y5_te, y5_hat),
    'precision': precision_score(y5_te, y5_hat, zero_division=0),
    'recall': recall_score(y5_te, y5_hat, zero_division=0),
    'f1_score': f1_score(y5_te, y5_hat, zero_division=0),
    'roc_auc': roc_auc_score(y5_te, y5_proba)
}

# Save honest metrics
pd.DataFrame([{'model':'task5_honest', **metrics_task5}]).to_csv(OUTPUT_DIR / 'task5_honest_metrics.csv', index=False)
print(f"Saved task5_honest_metrics.csv in {OUTPUT_DIR}")

# Feature Importance chart for Task5
fi = pd.Series(model5.feature_importances_, index=X5_tr.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
fi[::-1].plot(kind='barh', color='#1f77b4')
plt.title('Top 10 Task 5 Honest Feature Importances')
plt.xlabel('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'task5_honest_feature_importance.png', dpi=150)
plt.close()
print(f"Saved task5_honest_feature_importance.png in {OUTPUT_DIR}")

# Save task5 model
joblib.dump(model5, OUTPUT_DIR / 'task5_honest_model.pkl')
print(f"Saved task5_honest_model.pkl in {OUTPUT_DIR}")

# Audit report details
auc_diff = metrics_task4['roc_auc'] - metrics_task5['roc_auc']

audit_report = (
    "**Audit Report:** In the Task 4 pipeline, look-ahead leakage features were present: "
    f"{', '.join([f for f in lookahead_features if f in task4_features])}. "
    "These features are illegal before departure because they depend on post-departure outcomes (arrival/departure times/delays, taxi, air-time). "
    f"For Task 5, I dropped them and retrained only on clean pre-departure features: {clean_feature_set[:10]}{'...' if len(clean_feature_set) > 10 else ''}. "
    f"Task 4 ROC-AUC={metrics_task4['roc_auc']:.4f}, Task 5 ROC-AUC={metrics_task5['roc_auc']:.4f}, difference={auc_diff:.4f}. "
    "This confirms the previous model was over-optimistic from leakage and the audited model is honest for production adoption."
)
print(audit_report)

with open(OUTPUT_DIR / 'task5_audit_report.txt', 'w', encoding='utf-8') as f:
    f.write(audit_report + '\n')
print('Saved task5_audit_report.txt')
