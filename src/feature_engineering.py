import pandas as pd
import numpy as np

# 2.1 Load the cleaned master data
master = pd.read_csv('data/master_clean.csv')
master['order_purchase_timestamp'] = pd.to_datetime(master['order_purchase_timestamp'])

print(master.shape)
print(master.columns.tolist())

# 2.2 Set a reference snapshot date
snapshot_date = master['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
print("Snapshot date:", snapshot_date)

# 2.3 Aggregate to customer level — RFM
customer_features = master.groupby('customer_unique_id').agg(
    recency_days=('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('payment_value', 'sum'),
    avg_order_value=('payment_value', 'mean'),
    avg_review_score=('review_score', 'mean'),
    first_purchase=('order_purchase_timestamp', 'min'),
    last_purchase=('order_purchase_timestamp', 'max'),
).reset_index()

print(customer_features.shape)
print(customer_features.head())

# 2.4 Customer tenure
customer_features['tenure_days'] = (snapshot_date - customer_features['first_purchase']).dt.days

# 2.5 Additional behavior features
items_per_order = master.groupby('customer_unique_id')['n_items'].mean().reset_index()
items_per_order.columns = ['customer_unique_id', 'avg_items_per_order']
customer_features = customer_features.merge(items_per_order, on='customer_unique_id', how='left')

freight = master.groupby('customer_unique_id')['total_freight'].mean().reset_index()
freight.columns = ['customer_unique_id', 'avg_freight']
customer_features = customer_features.merge(freight, on='customer_unique_id', how='left')

# 2.6 Define target: will this customer place a repeat (2nd+) order?
customer_features['repeat_purchase'] = (customer_features['frequency'] > 1).astype(int)

print(customer_features['repeat_purchase'].value_counts())
print(customer_features['repeat_purchase'].value_counts(normalize=True))

# 2.7 Define CLV target
customer_features['clv_target'] = customer_features['monetary']

# Fill remaining nulls
customer_features['avg_order_value'] = customer_features['avg_order_value'].fillna(customer_features['monetary'])
customer_features['avg_review_score'] = customer_features['avg_review_score'].fillna(customer_features['avg_review_score'].median())

# 2.8 Final checks and save
print(customer_features.describe())
print(customer_features.isnull().sum())

customer_features.to_csv('data/customer_features.csv', index=False)
print("Saved customer_features.csv")