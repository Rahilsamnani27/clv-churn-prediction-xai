import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="CLV & Churn Dashboard", layout="wide")
st.title("Customer CLV & Repeat-Purchase Risk Dashboard")
st.caption("Identifies high-value customers unlikely to repeat-purchase, with AI-generated retention emails")

# Load at-risk customers
top_n = st.slider("Number of at-risk customers to show", 5, 50, 20)

with st.spinner("Loading at-risk customers..."):
    response = requests.get(f"{API_BASE}/at-risk-customers", params={"top_n": top_n})
    customers = response.json()

df = pd.DataFrame(customers)
df['clv_target'] = df['clv_target'].round(2)
df['repeat_purchase_proba'] = (df['repeat_purchase_proba'] * 100).round(1)

st.subheader(f"Top {top_n} High-Value At-Risk Customers")
st.dataframe(
    df[['customer_unique_id', 'clv_target', 'repeat_purchase_proba', 'avg_review_score', 'tenure_days']]
    .rename(columns={
        'clv_target': 'CLV (R$)',
        'repeat_purchase_proba': 'Repeat-Purchase Probability (%)',
        'avg_review_score': 'Avg Review Score',
        'tenure_days': 'Tenure (days)'
    }),
    use_container_width=True
)

st.divider()

# Customer detail + email generation
st.subheader("Generate Retention Email")
selected_id = st.selectbox("Select a customer", df['customer_unique_id'].tolist())

selected_row = df[df['customer_unique_id'] == selected_id].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("CLV", f"R${selected_row['clv_target']:.2f}")
col2.metric("Repeat-Purchase Probability", f"{selected_row['repeat_purchase_proba']:.1f}%")
col3.metric("Avg Review Score", f"{selected_row['avg_review_score']}/5")

st.write("**Churn reason:**", selected_row['churn_reason'])

if st.button("Generate Retention Email"):
    with st.spinner("Generating personalized email..."):
        email_response = requests.get(f"{API_BASE}/customer/{selected_id}/retention-email")
        email = email_response.json()['retention_email']
    st.text_area("Generated Email", email, height=200)