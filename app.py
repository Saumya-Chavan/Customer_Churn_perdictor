import streamlit as st
import pandas as pd
import pickle
import os
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    layout="centered", # Centered layout as requested
    initial_sidebar_state="collapsed"
)

def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        ext = image_path.split('.')[-1].lower()
        mime_type = "image/png" if ext == "png" else "image/jpeg"
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:{mime_type};base64,{encoded_string});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            /* Make main background transparent so image shows through */
            [data-testid="stAppViewContainer"] {{
                background-color: transparent !important;
            }}
            /* Transparent header */
            [data-testid="stHeader"] {{
                background-color: transparent !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Call the background function (save your image as background.jpg or background.png in the same folder)
if os.path.exists("background.jpg"):
    set_background("background.jpg")
elif os.path.exists("background.png"):
    set_background("background.png")

# Modern Glassmorphism CSS styling (White/Light Frost Edition)
st.markdown("""
<style>
    /* Light frosted glass for bordered containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.3s ease, border-color 0.3s ease, background 0.3s ease;
        margin-top: 1.2rem !important;
        margin-bottom: 1.2rem !important;
        padding: 0.5rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.6) !important;
        transform: translateY(-2px);
    }
    
    /* Clean headers with professional coloring */
    h1 { color: #ffffff !important; font-weight: 800; text-align: center; margin-bottom: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.5);}
    .subtitle { text-align: center; color: #e2e8f0; font-size: 1.1rem; margin-bottom: 2rem; text-shadow: 0 1px 5px rgba(0,0,0,0.5);}

    /* Prediction Result Cards */
    .result-card {
        padding: 2rem;
        margin-top: 2rem;
        border-radius: 12px;
        text-align: center;
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-top-width: 5px;
        border-top-style: solid;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .result-churn { border-top-color: #ef4444; }
    .result-safe { border-top-color: #10b981; }
    
    .result-title { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    .result-churn .result-title { color: #fca5a5; }
    .result-safe .result-title { color: #6ee7b7; }
    
    .result-metric { font-size: 3.5rem; font-weight: 900; margin-bottom: 0.5rem; text-shadow: 0 2px 8px rgba(0,0,0,0.5); }
    .result-churn .result-metric { color: #ef4444; }
    .result-safe .result-metric { color: #10b981; }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource(show_spinner=False)
def load_assets():
    if not os.path.exists("customer_churn_model.pkl") or not os.path.exists("encoders.pkl"):
        return None, None
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model_data, encoders

model_data, encoders = load_assets()

# --- MAIN DASHBOARD HEADER ---
st.markdown("<h1>Telecom Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Identify at-risk accounts using predictive behavioral mapping</div>", unsafe_allow_html=True)

if model_data is None or encoders is None:
    st.error("System Error: Predictive model or encoders not found.")
    st.stop()

loaded_model = model_data["model"]

# --- CORE PARAMETERS ---
st.markdown("### 📋 Primary Indicators")

# Added a middle spacer column to create distinct separation between the two blocks
col1, space, col2 = st.columns([10, 1, 10])

with col1:
    with st.container(border=True):
        st.markdown("**Financial & Contract**")
        contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
        monthly_charges = st.number_input("Monthly Spend ($)", min_value=0.0, value=65.0, step=5.0)

with col2:
    with st.container(border=True):
        st.markdown("**Core Services**")
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        tech_support = st.selectbox("Tech Support Included?", ["No", "Yes", "No internet service"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# --- SECONDARY PARAMETERS (EXPANDER) ---
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("⚙️ Advanced Profile Configurations (Optional)", expanded=False):
    st.markdown("These demographic and service factors have less predictive weight. Defaults are applied.")
    
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    
    with ex_col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
    with ex_col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        
    with ex_col3:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
    total_charges = st.number_input("Total Lifetime Charges ($)", min_value=0.0, value=monthly_charges*max(tenure, 1))

# --- EXECUTION BUTTON ---
st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
with btn_col2:
    execute_inference = st.button("Generate Model Inference", type="primary", use_container_width=True)

st.divider()

# --- PREDICTION LOGIC & RESULTS ---
if execute_inference:
    with st.spinner("Processing analytical pipelines..."):
        # Map inputs
        input_data = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner,
            'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
            'MultipleLines': multiple_lines, 'InternetService': internet_service,
            'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Apply encoding
        for column, encoder in encoders.items():
            if column in input_df.columns:
                try:
                    input_df[column] = encoder.transform(input_df[column])
                except ValueError:
                    pass
                    
        # Prediction
        prediction = loaded_model.predict(input_df)
        pred_prob = loaded_model.predict_proba(input_df)
        
        # Centered Results Display
        if prediction[0] == 1:
            risk_score = pred_prob[0][1] * 100
            st.markdown(f"""
                <div class="result-card result-churn">
                    <div class="result-title">High Risk Detected</div>
                    <div class="result-metric">{risk_score:.1f}%</div>
                    <p style="color: #cbd5e1; font-weight: 500; font-size: 1.1rem; max-width: 600px; margin: 0 auto; line-height: 1.5;">
                        <span style="color: #ef4444;">Action Required:</span> The analytical model indicates this account requires immediate retention measures to prevent service termination.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            risk_score = pred_prob[0][0] * 100
            st.markdown(f"""
                <div class="result-card result-safe">
                    <div class="result-title">Account Stable</div>
                    <div class="result-metric">{risk_score:.1f}%</div>
                    <p style="color: #cbd5e1; font-weight: 500; font-size: 1.1rem; max-width: 600px; margin: 0 auto; line-height: 1.5;">
                        <span style="color: #10b981;">Optimal Standing:</span> This customer is highly likely to remain with the service. Standard relationship management procedures apply.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
