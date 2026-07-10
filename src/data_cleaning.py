import pandas as pd

# 1.2 Load all tables
customers = pd.read_csv('data/olist_customers_dataset.csv')
orders = pd.read_csv('data/olist_orders_dataset.csv')
order_items = pd.read_csv('data/olist_order_items_dataset.csv')
payments = pd.read_csv('data/olist_order_payments_dataset.csv')
reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')
products = pd.read_csv('data/olist_products_dataset.csv')

# --- EXPLORATION FIRST ---
for name, df in [('customers', customers), ('orders', orders), ('order_items', order_items),
                  ('payments', payments), ('reviews', reviews), ('products', products)]:
    print(f"\n===== {name.upper()} =====")
    print(df.info())
    print(df.describe(include='all'))
    print("Duplicate rows:", df.duplicated().sum())
    print("Null counts:\n", df.isnull().sum()[df.isnull().sum() > 0])
    
# 1.3 Parse dates
date_cols = ['order_purchase_timestamp', 'order_approved_at',
             'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date']
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors='coerce')

# 1.4 Handle missing values
orders = orders.dropna(subset=['order_purchase_timestamp'])
print(orders['order_status'].value_counts())

reviews['review_score'] = reviews['review_score'].fillna(reviews['review_score'].median())
products['product_category_name'] = products['product_category_name'].fillna('unknown')

# 1.5 Remove duplicates
for name, df in [('orders', orders), ('order_items', order_items), ('payments', payments)]:
    dupes = df.duplicated().sum()
    print(name, 'duplicates:', dupes)
    df.drop_duplicates(inplace=True)

# 1.6 Filter to delivered orders only
orders_clean = orders[orders['order_status'] == 'delivered'].copy()

# 1.7 Merge into one master customer-order table
order_value = order_items.groupby('order_id').agg(
    total_item_value=('price', 'sum'),
    total_freight=('freight_value', 'sum'),
    n_items=('order_item_id', 'count')
).reset_index()

master = orders_clean.merge(customers, on='customer_id', how='left')
master = master.merge(order_value, on='order_id', how='left')
master = master.merge(payments.groupby('order_id')['payment_value'].sum().reset_index(),
                       on='order_id', how='left')
master = master.merge(reviews[['order_id', 'review_score']], on='order_id', how='left')

print(master.shape)
print(master.head())

# 1.8 Save cleaned master table
master.to_csv('data/master_clean.csv', index=False)
print("Saved cleaned data to data/master_clean.csv")