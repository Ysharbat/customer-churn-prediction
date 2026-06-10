import streamlit as st
import pandas as pd
import numpy as np
import pickle



# Preprocessing function
def preprocess(df, columns, scaler, numeric_cols):
    df = df.copy()
    # Scale numeric features — must match training
    cols_to_scale = [c for c in numeric_cols if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=columns, fill_value=0)
    return df

# pkl Loads


def load_pkl(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"{path} not found. Please rerun the modeling notebook.")
        st.stop()

model               = load_pkl("../app/model.pkl")
columns             = load_pkl("../app/columns.pkl")
contract_medians    = load_pkl("../app/contract_medians.pkl")
priority_thresholds = load_pkl("../app/priority_thresholds.pkl")
scaler              = load_pkl("../app/scaler.pkl")
numeric_cols        = load_pkl("../app/numeric_cols.pkl")



CONTRACT_WEIGHTS = {
    'Month-to-month': 1.5,
    'One year':       1.0,
    'Two year':       0.7
}

# p^1.5 amplifies high-confidence predictions so borderline cases rank lower
def compute_priority_score(probability, clv, contract):
    weight = CONTRACT_WEIGHTS.get(contract, 1.0)
    return (probability ** 1.5) * clv * weight


data = pd.read_csv("app_data.csv")


st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("Customer Churn & Value Prediction")
st.write("Predict churn, estimate customer value, and prioritize retention efforts.")



threshold = st.slider(
    "🎯 Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05
)

st.caption("Lower threshold = catch more churners (higher cost). Higher threshold = fewer false alarms.")

if threshold < 0.3:
    st.info("Aggressive Strategy: Catch more churners (higher cost)")
elif threshold < 0.6:
    st.info("Balanced Strategy")
else:
    st.info("Conservative Strategy: Fewer false positives")
    




mode = st.radio(
    "Choose Mode",
    ["Predict New Customer", "Select Existing Customer","Dashboard Settings"]
)
st.caption("Predict New Customer: explore hypothetical scenarios | Select Existing Customer: analyze real customers | Dashboard: retention targeting")

#==============================
# Mode 1- Manual Input
#==============================

if mode == "Predict New Customer":

    st.header("Predict New Customer")
    
               
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0,
        help="Typical range: $18 – $119 based on dataset"
    )
    

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    online_security = st.selectbox("Online Security", ["Yes", "No"])
    tech_support    = st.selectbox("Tech Support", ["Yes", "No"])
    payment_method  = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                      "Bank transfer (automatic)",
                                                      "Creditcard (automatic)"])


    expected_lifespan = contract_medians[contract]
    remaining_lifetime = float(
        np.clip(
            (expected_lifespan - tenure) + tenure * 0.3,
            6, 72
        )
    )
    
    clv_input = monthly_charges * remaining_lifetime

    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": tenure * monthly_charges,
        "CLV": clv_input,
        "Contract": contract,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "PaymentMethod": payment_method
    }


    input_df = pd.DataFrame([input_dict])

    

    input_df = preprocess(input_df, columns, scaler, numeric_cols)


    

    st.header("Prediction")

    if st.button("Predict"):

        probability = model.predict_proba(input_df)[0][1]
        prediction = int(probability >= threshold)
        
        st.subheader("Prediction Result")

        
        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")

        st.write(f"**Churn Probability:** {probability:.2f}")
        st.write(f"**Decision Threshold:** {threshold}")
        st.caption("Probability based on patterns of similar customers.")
    
    
    
        # Risk segmentation
    
        if probability < 0.35:
            risk = "Low Risk"
            st.success("🟢 Low Risk Customer")
        elif probability < 0.65:
            risk = "Medium Risk"
            st.warning("🟡 Medium Risk Customer")
        else:
            risk = "High Risk"
            st.error("🔴 High Risk Customer")

    
        # CLV + Priority score
    
        clv = clv_input
        priority_score = compute_priority_score(probability, clv, contract)
        

        st.subheader("Business Insights")
        st.write(f"**Customer Lifetime Value (CLV):** ${clv:,.2f}")
        st.write(f"**Priority Score:** {priority_score:,.2f}")

        if priority_score > priority_thresholds['high']:
            st.error("🔥 HIGH PRIORITY: Immediate retention action recommended")
        elif priority_score > priority_thresholds['medium']:
            st.warning("⚡ MEDIUM PRIORITY: Monitor and engage proactively")
        else:
            st.success("✅ LOW PRIORITY: No immediate action needed")
            
            
            
#==================================
# Mode 2 - Real Customer Data
#==================================


if mode == "Select Existing Customer":
    
    st.header("Select Existing Customer")

    customer_id = st.selectbox(
        "Choose Customer",
        data["customerID"].sort_values()
    )

    customer_data = data[data["customerID"] == customer_id]

    st.subheader("Customer Data")
    st.dataframe(customer_data)

    input_df = customer_data.drop(['Churn', 'customerID'], axis=1, errors='ignore')
    
    #CLV
    input_df = input_df.copy()
    input_df['expected_lifespan'] = input_df['Contract'].map(contract_medians)
    input_df['remaining_lifetime'] = (
        (input_df['expected_lifespan'] - input_df['tenure']).clip(lower=0) +
        input_df['tenure'] * 0.3
    ).clip(lower=6, upper=72)
    
    input_df['CLV'] = input_df['MonthlyCharges'] * input_df['remaining_lifetime']

    input_df = input_df.drop(['expected_lifespan', 'remaining_lifetime'], axis=1)
    
    input_encoded = preprocess(input_df, columns, scaler, numeric_cols)

    if st.button("Predict Selected Customer"):

        probability = model.predict_proba(input_encoded)[0][1]
        prediction = int(probability >= threshold)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")

        st.write(f"**Churn Probability:** {probability:.2f}")
        st.write(f"**Decision Threshold:** {threshold}")
        
        
        # Risk
        if probability < 0.35:
            risk = "Low Risk"
            st.success("🟢 Low Risk Customer")
        elif probability < 0.65:
            risk = "Medium Risk"
            st.warning("🟡 Medium Risk Customer")
        else:
            risk = "High Risk"
            st.error("🔴 High Risk Customer")
            
            
        # CLV + Priority
        clv = input_df['CLV'].values[0]
        priority_score = compute_priority_score(
            probability, clv, customer_data['Contract'].values[0]
        )

        st.subheader("💼 Business Insights")
        st.write(f"**Customer Lifetime Value (CLV):** ${clv:,.2f}")
        st.write(f"**Churn Probability:** {probability:.2f}")
        st.write(f"**Priority Score:** {priority_score:,.2f}")
        
         
        if priority_score > priority_thresholds['high']:
            st.error("🔥 HIGH PRIORITY: Immediate retention action recommended")
        elif priority_score > priority_thresholds['medium']:
            st.warning("⚡ MEDIUM PRIORITY: Monitor and engage proactively")
        else:
            st.success("✅ LOW PRIORITY: No immediate action needed")
            
        
            
        
#================================        
# Top Customers Dashboard
#================================

# leading underscores tell Streamlit not to hash these args (unhashable types)

@st.cache_data
def get_full_predictions(_model, _scaler, _columns, _numeric_cols, _contract_medians):
    full_df = data.copy()
    full_df['expected_lifespan'] = full_df['Contract'].map(_contract_medians)
    full_df['remaining_lifetime'] = (
        (full_df['expected_lifespan'] - full_df['tenure']).clip(lower=0)
        + full_df['tenure'] * 0.3
    ).clip(lower=6, upper=72)
    full_df['CLV'] = full_df['MonthlyCharges'] * full_df['remaining_lifetime']

    X_full = preprocess(
        full_df.drop(['Churn', 'customerID', 'expected_lifespan', 'remaining_lifetime'], axis=1),
        _columns, _scaler, _numeric_cols
    )
    full_df['churn_probability'] = _model.predict_proba(X_full)[:, 1]
    return full_df

    
    
if mode == "Dashboard Settings":
    
    st.header("Customers to Target for Retention")

    full_df = get_full_predictions(model, scaler, columns, numeric_cols, contract_medians)
    full_df['Predicted_churn'] = (full_df['churn_probability'] >= threshold).astype(int)
    
    

    # Distribution
    st.subheader("Prediction Summary")

    counts = full_df['Predicted_churn'].value_counts()
    total = len(full_df)

    st.write(f"🔴 High-Risk: {counts.get(1,0)} ({counts.get(1,0)/total:.1%})")
    st.write(f"🟢 Low-Risk Customers: {counts.get(0, 0)} ({counts.get(0,0)/total:.1%})")

    

    st.subheader("Dashboard Settings")
    
    # Filter high-risk customers (dynamic threshold)
    retention_df = full_df[full_df['churn_probability'] >= threshold].copy()

    

    retention_df['priority_score'] = retention_df.apply(
        lambda r: compute_priority_score(r['churn_probability'], r['CLV'], r['Contract']),
        axis=1
    )

    # Expected loss
    retention_df['expected_loss'] = retention_df['churn_probability'] * retention_df['CLV']
    
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers",   f"{len(full_df):,}")
    col2.metric("High-Risk",         f"{counts.get(1,0):,}")
    col3.metric("Revenue at Risk",   f"${retention_df['expected_loss'].sum():,.0f}")
    col4.metric("Avg CLV (at-risk)", f"${retention_df['CLV'].mean():,.0f}")
    
    st.caption("Adjust the number of customers to explore different risk segments.")

    #top_n = st.slider("Select number of customers", min_value=5, max_value=100, value=10)

    top_n_input = st.number_input("Number of customers", min_value=1, 
                                  max_value=500, value=10)
    

    # Safe top_n
    top_n = min(top_n_input, len(retention_df))

    top_customers = retention_df.sort_values(by='priority_score',
                                             ascending=False).head(top_n)
               


    st.subheader(f"Top {top_n} High-Risk Customers to Target")
    st.dataframe(top_customers)

    # High confidence (very risky)
    HIGH_CONF_CUTOFF = max(threshold, 0.7)
    high_confidence = full_df[full_df['churn_probability'] > HIGH_CONF_CUTOFF]
    st.caption(f"Showing customers with churn probability > {HIGH_CONF_CUTOFF:.2f}")

    top_n_conf = min(top_n_input, len(high_confidence))

    st.subheader(f"Top {top_n_conf} High Confidence Churners")
    st.dataframe(high_confidence.head(top_n_conf))

    st.subheader("Expected Revenue at Risk")
    st.write(f"Total Expected Loss: ${retention_df['expected_loss'].sum():,.0f}")

    # Download button
    csv = top_customers.to_csv(index=False)
    st.download_button(
        label="⬇️ Download High-Risk Customers as CSV",
        data=csv,
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        