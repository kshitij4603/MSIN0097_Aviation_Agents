import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

RANDOM_SEED = 42

DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'flights.csv'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'antigravity'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('Loading', DATA_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)

# target
df = df[df['CANCELLED'] == 0] if 'CANCELLED' in df.columns else df
df = df[df['DIVERTED'] == 0] if 'DIVERTED' in df.columns else df

df = df.dropna(subset=['ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'DISTANCE'])

df['DELAY_15'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

# reduce sample size for speed and reproducibility
if len(df) > 500000:
    df = df.sample(n=500000, random_state=RANDOM_SEED)

# select features
features = []
for col in ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'TAXI_OUT', 'TAXI_IN', 'WHEELS_OFF', 'WHEELS_ON', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'AIRLINE']:
    if col in df.columns:
        features.append(col)

# reduce to shorter subset for speed and reliability
selected_features = [f for f in features if f not in ['DAY', 'DEPARTURE_TIME','WHEELS_OFF','WHEELS_ON']]

# handle missing through median
df = df[selected_features + ['DELAY_15']].copy()
for c in df.columns:
    if df[c].dtype.kind in 'biufc':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('NA')

# encode categorical
if 'AIRLINE' in df.columns:
    dummies = pd.get_dummies(df['AIRLINE'], prefix='airline', drop_first=True)
    df = pd.concat([df.drop(columns=['AIRLINE']), dummies], axis=1)

X = df.drop(columns=['DELAY_15'])
y = df['DELAY_15']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

# scaling numeric features
digit_cols = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[digit_cols] = scaler.fit_transform(X_train[digit_cols])
X_test_scaled[digit_cols] = scaler.transform(X_test[digit_cols])

# model
model = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100, n_jobs=2)
model.fit(X_train_scaled, y_train)

# predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_prob)
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(OUTPUT_DIR / 'task3_metrics.csv', index=False)
print('Saved task3_metrics.csv')

# feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
imp_df = importances.reset_index()
imp_df.columns = ['feature', 'importance']

plt.figure(figsize=(12, 6))
plt.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color='#2c7fb8')
plt.title('Top 10 Feature Importances - Baseline RandomForest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'task3_feature_importance.png', dpi=150)
plt.close()
print('Saved task3_feature_importance.png')

imp_df.to_csv(OUTPUT_DIR / 'task3_feature_importance.csv', index=False)

# save model
joblib.dump(model, OUTPUT_DIR / 'task3_baseline.pkl')
print('Saved task3_baseline.pkl')

summary = ("Baseline RandomForest achieved {:.4f} accuracy, {:.4f} precision, {:.4f} recall, "
           "{:.4f} F1, and {:.4f} ROC-AUC. Top predictive features include: {}. "
           "We can proceed with tree-based ensembles for Task 3. ").format(
    metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['roc_auc'], ', '.join(importances.index.tolist())
)
print(summary)

with open(OUTPUT_DIR / 'task3_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary + '\n')
