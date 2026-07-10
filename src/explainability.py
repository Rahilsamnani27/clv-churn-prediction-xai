import pandas as pd
import numpy as np
import joblib
import shap

df = pd.read_csv('data/customer_features.csv')

drop_cols = ['customer_unique_id', 'first_purchase', 'last_purchase',
             'frequency', 'monetary', 'clv_target', 'repeat_purchase',
             'recency_days']

feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]

model = joblib.load('src/churn_model.pkl')
print("Model loaded. Features:", feature_cols)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("SHAP values shape:", shap_values.shape)
mean_abs_shap = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nGlobal Feature Importance:\n", mean_abs_shap)
import matplotlib
matplotlib.use('Agg')  # avoids display issues when running from terminal
import matplotlib.pyplot as plt

shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig('screenshots/shap_summary.png', dpi=150)
print("Saved SHAP summary plot to screenshots/shap_summary.png")

def explain_customer(idx):
    """Returns top contributing features (with direction) for a single customer's churn-risk prediction."""
    shap_row = shap_values[idx]
    contributions = pd.DataFrame({
        'feature': feature_cols,
        'value': X.iloc[idx].values,
        'shap_value': shap_row
    }).sort_values('shap_value', key=abs, ascending=False)
    return contributions

# Test on one at-risk customer
sample_idx = 0
print(f"\nCustomer {sample_idx} explanation:\n", explain_customer(sample_idx))
print("\nPredicted repeat-purchase probability:", model.predict_proba(X.iloc[[sample_idx]])[:, 1])

customer_ids = df['customer_unique_id'].values
all_explanations = []

for idx in range(len(X)):
    top_features = explain_customer(idx).head(3)
    reason = "; ".join([
        f"{row['feature']}={row['value']:.1f} ({'+' if row['shap_value']>0 else ''}{row['shap_value']:.3f})"
        for _, row in top_features.iterrows()
    ])
    all_explanations.append({
        'customer_unique_id': customer_ids[idx],
        'churn_reason': reason
    })

explanations_df = pd.DataFrame(all_explanations)
explanations_df.to_csv('data/shap_explanations.csv', index=False)
print("Saved SHAP explanations to data/shap_explanations.csv")