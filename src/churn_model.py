import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import xgboost as xgb

df = pd.read_csv('data/customer_features.csv')
print(df.shape)
print(df['repeat_purchase'].value_counts())
# Drop columns that leak the target or aren't useful as features
drop_cols = ['customer_unique_id', 'first_purchase', 'last_purchase',
             'frequency', 'monetary', 'clv_target', 'repeat_purchase', 'recency_days']

feature_cols = [c for c in df.columns if c not in drop_cols]
print("Features used:", feature_cols)

X = df[feature_cols]
y = df['repeat_purchase']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Positive rate:", y_train.mean())
print("Test shape:", X_test.shape, "Positive rate:", y_test.mean())
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print("scale_pos_weight:", scale_pos_weight)
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    random_state=42
)

model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc_roc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print("AUC-ROC:", auc_roc)
print("PR-AUC:", pr_auc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
results = X_test.copy()
results['actual'] = y_test.values
results['predicted_proba'] = y_pred_proba
results = results.sort_values('predicted_proba', ascending=False)

for k in [0.05, 0.10, 0.20]:
    top_k = results.head(int(len(results) * k))
    precision_at_k = top_k['actual'].mean()
    print(f"Precision@Top{int(k*100)}%: {precision_at_k:.3f} (baseline: {y_test.mean():.3f})")
import joblib
joblib.dump(model, 'src/churn_model.pkl')
print("Model saved to src/churn_model.pkl")