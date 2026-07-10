import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
df = pd.read_csv('data/customer_features.csv')
shap_reasons = pd.read_csv('data/shap_explanations.csv')

model = joblib.load('src/churn_model.pkl')

drop_cols = ['customer_unique_id', 'first_purchase', 'last_purchase',
             'frequency', 'monetary', 'clv_target', 'repeat_purchase',
             'recency_days']
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]

df['repeat_purchase_proba'] = model.predict_proba(X)[:, 1]
df = df.merge(shap_reasons, on='customer_unique_id', how='left')
clv_threshold = df['clv_target'].quantile(0.75)  # top 25% by spend
proba_threshold = df['repeat_purchase_proba'].quantile(0.30)  # bottom 30% likelihood

at_risk_high_value = df[
    (df['clv_target'] >= clv_threshold) &
    (df['repeat_purchase_proba'] <= proba_threshold)
].sort_values('clv_target', ascending=False)

print(f"Found {len(at_risk_high_value)} high-value at-risk customers")
print(at_risk_high_value[['customer_unique_id', 'clv_target', 'repeat_purchase_proba', 'churn_reason']].head())
def generate_retention_email(customer_row):
    # Parse the dominant (first/strongest) factor from churn_reason
    top_reason = customer_row['churn_reason'].split(';')[0].strip()

    prompt = f"""You are a customer retention specialist for an e-commerce company.

Customer profile:
- Total historical spend: R${customer_row['clv_target']:.2f}
- Average order value: R${customer_row['avg_order_value']:.2f}
- Average review score given: {customer_row['avg_review_score']:.1f}/5
- Customer tenure: {customer_row['tenure_days']:.0f} days
- Predicted likelihood of repeat purchase: {customer_row['repeat_purchase_proba']*100:.1f}%
- The SINGLE strongest factor driving low repeat-purchase likelihood is: {top_reason}

Instructions:
- If the strongest factor relates to review_score being low, the email MUST acknowledge a past
  poor experience and apologize/offer to make it right — do NOT default to a generic "we miss you" tone.
- If the strongest factor relates to avg_freight being high, address shipping cost directly and
  lead the incentive with a free/discounted shipping offer.
- If the strongest factor relates to tenure or avg_order_value, use a loyalty/appreciation angle
  instead of an apology.
- Do not mention internal metrics, scores, or SHAP values directly to the customer.
- Under 120 words, warm and specific, not generic.

Write the retention email now, tailored to the strongest factor above."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content
sample_customers = at_risk_high_value.head(3)

for idx, row in sample_customers.iterrows():
    print(f"\n{'='*60}")
    print(f"Customer: {row['customer_unique_id']}")
    print(f"CLV: R${row['clv_target']:.2f} | Repeat-purchase probability: {row['repeat_purchase_proba']*100:.1f}%")
    print(f"Churn reason: {row['churn_reason']}")
    print(f"\n--- Generated Email ---")
    email = generate_retention_email(row)
    print(email)

results = []
for idx, row in at_risk_high_value.head(20).iterrows():
    email = generate_retention_email(row)
    results.append({
        'customer_unique_id': row['customer_unique_id'],
        'clv_target': row['clv_target'],
        'repeat_purchase_proba': row['repeat_purchase_proba'],
        'churn_reason': row['churn_reason'],
        'retention_email': email
    })

results_df = pd.DataFrame(results)
results_df.to_csv('data/retention_emails.csv', index=False)
print(f"\nSaved {len(results_df)} retention emails to data/retention_emails.csv")
