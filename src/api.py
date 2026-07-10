from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="CLV & Churn Prediction API")

# Load everything once at startup
df = pd.read_csv('data/customer_features.csv')
shap_reasons = pd.read_csv('data/shap_explanations.csv')
churn_model = joblib.load('src/churn_model.pkl')
clv_model = joblib.load('src/clv_model.pkl')

churn_drop_cols = ['customer_unique_id', 'first_purchase', 'last_purchase',
                    'frequency', 'monetary', 'clv_target', 'repeat_purchase', 'recency_days']
churn_features = [c for c in df.columns if c not in churn_drop_cols]

clv_drop_cols = churn_drop_cols + ['avg_order_value']
clv_features = [c for c in df.columns if c not in clv_drop_cols]

df['repeat_purchase_proba'] = churn_model.predict_proba(df[churn_features])[:, 1]
df = df.merge(shap_reasons, on='customer_unique_id', how='left')


def generate_retention_email(customer_row):
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
- If the strongest factor relates to review_score being low, acknowledge a past poor experience and apologize.
- If the strongest factor relates to avg_freight being high, address shipping cost directly with a free/discounted shipping offer.
- If the strongest factor relates to tenure or avg_order_value, use a loyalty/appreciation angle.
- Do not mention internal metrics or scores directly.
- Under 120 words, warm and specific.

Write the retention email now."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content


@app.get("/")
def root():
    return {"message": "CLV & Churn Prediction API is running"}


@app.get("/at-risk-customers")
def get_at_risk_customers(top_n: int = 20):
    clv_threshold = df['clv_target'].quantile(0.75)
    proba_threshold = df['repeat_purchase_proba'].quantile(0.30)

    at_risk = df[
        (df['clv_target'] >= clv_threshold) &
        (df['repeat_purchase_proba'] <= proba_threshold)
    ].sort_values('clv_target', ascending=False).head(top_n)

    return at_risk[['customer_unique_id', 'clv_target', 'repeat_purchase_proba',
                     'avg_review_score', 'tenure_days', 'churn_reason']].to_dict(orient='records')


@app.get("/customer/{customer_id}")
def get_customer(customer_id: str):
    row = df[df['customer_unique_id'] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    row = row.iloc[0]
    return {
        "customer_unique_id": customer_id,
        "clv_prediction": float(row['clv_target']),
        "repeat_purchase_probability": float(row['repeat_purchase_proba']),
        "avg_review_score": float(row['avg_review_score']),
        "tenure_days": float(row['tenure_days']),
        "churn_reason": row['churn_reason']
    }


@app.get("/customer/{customer_id}/retention-email")
def get_retention_email(customer_id: str):
    row = df[df['customer_unique_id'] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    row = row.iloc[0]
    email = generate_retention_email(row)
    return {"customer_unique_id": customer_id, "retention_email": email}