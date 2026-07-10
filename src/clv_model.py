import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

df = pd.read_csv('data/customer_features.csv')
print(df.shape)
print(df['clv_target'].describe())
# clv_target = monetary, so exclude monetary, frequency (leakage), recency_days (leakage, same reason as churn model)
drop_cols = ['customer_unique_id', 'first_purchase', 'last_purchase',
             'frequency', 'monetary', 'clv_target', 'repeat_purchase',
             'recency_days', 'avg_order_value']

feature_cols = [c for c in df.columns if c not in drop_cols]
print("Features used:", feature_cols)

X = df[feature_cols]
y = df['clv_target']
y_log = np.log1p(y)  # log(1 + y), handles zero values safely
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# Keep original scale test values for interpretable metrics
y_test_actual = np.expm1(y_test_log)
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train_log)
y_pred_log = model.predict(X_test)
y_pred_actual = np.expm1(y_pred_log)
y_pred_actual = np.clip(y_pred_actual, 0, None)  # no negative predictions

mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Mean actual CLV:", y_test_actual.mean())
comparison = pd.DataFrame({
    'actual_clv': y_test_actual.values[:10],
    'predicted_clv': y_pred_actual[:10]
})
print(comparison)
joblib.dump(model, 'src/clv_model.pkl')
print("Model saved to src/clv_model.pkl")