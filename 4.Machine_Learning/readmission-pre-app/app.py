import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import json
import os

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 5px;
    }
    .threshold-gauge {
        position: relative;
        height: 50px;
        background-color: #f0f0f0;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏥 Hospital Readmission Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered 30-day readmission risk assessment</p>', unsafe_allow_html=True)

# Configuration - Get from Databricks App secrets
ENDPOINT_URL = "https://dbc-0a95f537-e7bd.cloud.databricks.com/serving-endpoints/readmission/invocations"

# Get token from environment (Databricks Apps automatically injects this)
DATABRICKS_TOKEN = st.secrets['DATABRICKS_TOKEN']

# Model configuration
OPTIMAL_THRESHOLD = 0.30  # Update this with your actual optimal threshold from training
MODEL_VERSION = "champion"

def predict_readmission(input_df, threshold=OPTIMAL_THRESHOLD):
    """
    Call the serving endpoint for prediction
    
    Args:
        input_df: DataFrame with patient features
        threshold: Decision threshold (default: optimal threshold from training)
    
    Returns:
        dict with prediction results or None if error
    """
    if not DATABRICKS_TOKEN:
        st.error("⚠️ DATABRICKS_TOKEN not found in environment")
        return None
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Convert DataFrame to dataframe_split format
    payload = {
        "dataframe_split": input_df.to_dict(orient="split")
    }
    
    try:
        response = requests.post(
            ENDPOINT_URL,
            headers=headers,
            auth=('token', DATABRICKS_TOKEN),  # ✅ Changed to Basic Auth like curl
            data=json.dumps(payload),
            timeout=30
        )
        
        # Check for errors
        if response.status_code != 200:
            st.error(f"Endpoint error: {response.status_code}")
            st.error(response.text)
            return None
        
        result = response.json()
        
        # Extract predictions - handle different response formats
        predictions = result.get('predictions', [])
        
        if not predictions:
            st.error("No predictions returned from endpoint")
            return None
        
        # Get the prediction (first element if batch)
        pred_value = predictions[0]
        
        # Check if it's probability array [prob_class_0, prob_class_1]
        if isinstance(pred_value, list) and len(pred_value) == 2:
            prob_readmit = pred_value[1]  # Probability of readmission
            prediction = 1 if prob_readmit >= threshold else 0
        else:
            # Single probability value
            prob_readmit = float(pred_value) if pred_value <= 1.0 else pred_value
            prediction = 1 if prob_readmit >= threshold else 0
        
        return {
            'probability': prob_readmit,
            'prediction': prediction,
            'threshold_used': threshold,
            'raw_response': result
        }
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.exception(e)
        return None


def predict_readmissionx(input_df, threshold=OPTIMAL_THRESHOLD):
    """
    Call the serving endpoint for prediction
    
    Args:
        input_df: DataFrame with patient features
        threshold: Decision threshold (default: optimal threshold from training)
    
    Returns:
        dict with prediction results or None if error
    """
    if not DATABRICKS_TOKEN:
        st.error("⚠️ DATABRICKS_TOKEN not found in environment")
        return None
    
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Convert DataFrame to dataframe_split format
    payload = {
        "dataframe_split": input_df.to_dict(orient="split")
    }
    
    try:
        response = requests.post(
            ENDPOINT_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        
        # Check for errors
        if response.status_code != 200:
            st.error(f"Endpoint error: {response.status_code}")
            st.error(response.text)
            return None
        
        result = response.json()
        
        # Extract predictions - handle different response formats
        predictions = result.get('predictions', [])
        
        if not predictions:
            st.error("No predictions returned from endpoint")
            return None
        
        # Get the prediction (first element if batch)
        pred_value = predictions[0]
        
        # Check if it's probability array [prob_class_0, prob_class_1]
        if isinstance(pred_value, list) and len(pred_value) == 2:
            prob_readmit = pred_value[1]  # Probability of readmission
            prediction = 1 if prob_readmit >= threshold else 0
        else:
            # Single probability value
            prob_readmit = float(pred_value) if pred_value <= 1.0 else pred_value
            prediction = 1 if prob_readmit >= threshold else 0
        
        return {
            'probability': prob_readmit,
            'prediction': prediction,
            'threshold_used': threshold,
            'raw_response': result
        }
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.exception(e)
        return None

# Test endpoint connection on sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This tool predicts the likelihood of a patient being readmitted 
    to the hospital within 30 days based on clinical and demographic factors.
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Model Configuration")
    st.text(f"Endpoint: readmission")
    st.text(f"Model Version: {MODEL_VERSION}")
    
    # Connection status
    if DATABRICKS_TOKEN:
        st.success("✅ Connected to endpoint")
    else:
        st.error("❌ Token not configured")
    
    st.markdown("---")
    st.markdown("### ⚙️ Threshold Settings")
    
    use_optimal = st.checkbox("Use Optimal Threshold", value=True, 
                             help="Use F1-optimized threshold for best performance")
    
    if use_optimal:
        threshold = st.slider(
            "Decision Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=OPTIMAL_THRESHOLD, 
            step=0.01,
            help=f"Optimal threshold: {OPTIMAL_THRESHOLD:.3f}"
        )
    else:
        threshold = st.slider(
            "Decision Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01,
            help="Default threshold: 0.5"
        )
    
    st.info(f"Current threshold: **{threshold:.3f}**")
    
    with st.expander("📊 About Thresholds"):
        st.markdown(f"""
        **Why adjust threshold?**
        
        Hospital readmission is imbalanced (~10% readmit).
        
        - **Default (0.5)**: Misses many readmissions
        - **Optimal ({OPTIMAL_THRESHOLD:.2f})**: Balanced F1 score
        - **Lower (<0.3)**: Catch more readmissions
        - **Higher (>0.4)**: Reduce false alarms
        
        **Current setting:** {threshold:.3f}
        """)

# Main form
st.markdown("## Patient Information")

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Demographics")
    age = st.number_input("Age", min_value=0, max_value=120, value=65, help="Patient's age in years")
    
    sex = st.selectbox(
        "Sex",
        options=[1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female",
        help="1 = Male, 2 = Female"
    )
    
    race = st.selectbox(
        "Race",
        options=[1, 2, 3, 5],
        format_func=lambda x: {1: "White", 2: "Black", 3: "Other", 5: "Hispanic"}[x],
        help="Patient's race code"
    )

with col2:
    st.subheader("🏥 Utilization History")
    
    length_of_stay = st.number_input(
        "Length of Stay (days)",
        min_value=0,
        max_value=100,
        value=3,
        help="Number of days in current admission"
    )
    
    admissions_past_6mo = st.number_input(
        "Admissions (Past 6 Months)",
        min_value=0,
        max_value=50,
        value=0,
        help="Number of hospital admissions in the past 6 months"
    )
    
    cost_past_6mo = st.number_input(
        "Healthcare Cost (Past 6 Months)",
        min_value=0.0,
        max_value=500000.0,
        value=5000.0,
        step=100.0,
        format="%.2f",
        help="Total healthcare costs in past 6 months"
    )
    
    rx_fills_30days_after = st.number_input(
        "Rx Fills (30 Days After)",
        min_value=0,
        max_value=100,
        value=5,
        help="Number of prescription fills in 30 days post-discharge"
    )
    
    had_er_visit_7days_prior = st.selectbox(
        "ER Visit (7 Days Prior)",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Had emergency room visit in past 7 days"
    )

with col3:
    st.subheader("🩺 Chronic Conditions")
    
    sp_alzhdmta = st.checkbox("Alzheimer's Disease", help="SP_ALZHDMTA")
    sp_chf = st.checkbox("Congestive Heart Failure", help="SP_CHF")
    sp_chrnkidn = st.checkbox("Chronic Kidney Disease", help="SP_CHRNKIDN")
    sp_cncr = st.checkbox("Cancer", help="SP_CNCR")
    sp_copd = st.checkbox("COPD", help="SP_COPD")
    sp_depressn = st.checkbox("Depression", help="SP_DEPRESSN")
    sp_diabetes = st.checkbox("Diabetes", help="SP_DIABETES")
    sp_ischmcht = st.checkbox("Ischemic Heart Disease", help="SP_ISCHMCHT")

st.markdown("---")

# Prediction button
if st.button("🔮 Predict Readmission Risk", type="primary"):
    
    if not DATABRICKS_TOKEN:
        st.error("⚠️ DATABRICKS_TOKEN not configured. Check your app secrets.")
        st.stop()
    
    with st.spinner("Analyzing patient data..."):
        try:
            # Prepare the input data as a pandas DataFrame
            # Match the exact column order from your training data
            input_data = pd.DataFrame({
                "AGE": [float(age)],
                "BENE_SEX_IDENT_CD": [int(sex)],
                "BENE_RACE_CD": [int(race)],
                "SP_ALZHDMTA": [int(sp_alzhdmta)],
                "SP_CHF": [int(sp_chf)],
                "SP_CHRNKIDN": [int(sp_chrnkidn)],
                "SP_CNCR": [int(sp_cncr)],
                "SP_COPD": [int(sp_copd)],
                "SP_DEPRESSN": [int(sp_depressn)],
                "SP_DIABETES": [int(sp_diabetes)],
                "SP_ISCHMCHT": [int(sp_ischmcht)],
                "length_of_stay": [int(length_of_stay)],
                "admissions_past_6mo": [int(admissions_past_6mo)],
                "cost_past_6mo": [float(cost_past_6mo)],
                "rx_fills_30days_after": [int(rx_fills_30days_after)],
                "had_er_visit_7days_prior": [int(had_er_visit_7days_prior)]
            })
            
            # Make prediction using endpoint
            result = predict_readmission(input_data, threshold=threshold)
            
            if result is None:
                st.error("❌ Failed to get prediction. Please check the logs above.")
                st.stop()
            
            # Extract results
            prob_readmit = result['probability']
            prediction = result['prediction']
            threshold_used = result['threshold_used']
            
            # Display results
            st.markdown("## 📊 Prediction Results")
            
            # Create columns for metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Readmission Probability", 
                    f"{prob_readmit*100:.1f}%"
                )
            
            with metric_col2:
                prediction_label = "WILL READMIT" if prediction == 1 else "NO READMISSION"
                prediction_icon = "🔴" if prediction == 1 else "🟢"
                st.metric(
                    "Prediction",
                    f"{prediction_icon} {prediction_label}"
                )
            
            with metric_col3:
                # Risk level based on probability
                if prob_readmit > 0.7:
                    risk_level = "🔴 HIGH"
                elif prob_readmit > 0.4:
                    risk_level = "🟡 MEDIUM"
                else:
                    risk_level = "🟢 LOW"
                st.metric("Risk Level", risk_level)
            
            with metric_col4:
                risk_factors = sum([
                    sp_alzhdmta, sp_chf, sp_chrnkidn, sp_cncr,
                    sp_copd, sp_depressn, sp_diabetes, sp_ischmcht
                ])
                st.metric("Chronic Conditions", risk_factors)
            
            # Visual risk gauge with threshold indicator
            st.markdown("### 📈 Risk Assessment")
            
            # Determine color based on risk level
            if prob_readmit > 0.7:
                bar_color = "#c62828"
            elif prob_readmit > 0.4:
                bar_color = "#f57c00"
            else:
                bar_color = "#2e7d32"
            
            # Create visual gauge
            st.markdown(
                f"""
                <div style="position: relative; height: 50px; background-color: #f0f0f0; border-radius: 5px;">
                    <div style="
                        background-color: {bar_color};
                        width: {prob_readmit*100}%;
                        height: 100%;
                        border-radius: 5px;
                        text-align: center;
                        line-height: 50px;
                        color: white;
                        font-weight: bold;
                        font-size: 18px;
                    ">
                        {prob_readmit*100:.1f}%
                    </div>
                    <div style="
                        position: absolute;
                        left: {threshold_used*100}%;
                        top: 0;
                        width: 3px;
                        height: 100%;
                        background-color: black;
                    "></div>
                    <div style="
                        position: absolute;
                        left: {max(0, min(95, threshold_used*100-2))}%;
                        top: -25px;
                        font-size: 12px;
                        color: black;
                        font-weight: bold;
                    ">Threshold: {threshold_used:.3f}</div>
                </div>
                <div style="text-align: center; margin-top: 10px; color: #666;">
                    <small>Decision: {prob_readmit:.3f} {'≥' if prediction == 1 else '<'} {threshold_used:.3f} = 
                    {'Readmission Predicted' if prediction == 1 else 'No Readmission'}</small>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Risk assessment based on prediction (not just probability)
            if prediction == 1:
                if prob_readmit > 0.7:
                    st.markdown(
                        '<div class="prediction-box high-risk">🔴 HIGH RISK - READMISSION LIKELY</div>',
                        unsafe_allow_html=True
                    )
                    st.error("This patient has a HIGH likelihood of being readmitted within 30 days.")
                else:
                    st.markdown(
                        '<div class="prediction-box high-risk">⚠️ READMISSION PREDICTED</div>',
                        unsafe_allow_html=True
                    )
                    st.warning("This patient is predicted to be readmitted within 30 days.")
            else:
                st.markdown(
                    '<div class="prediction-box low-risk">✅ LOW RISK - NO READMISSION EXPECTED</div>',
                    unsafe_allow_html=True
                )
                st.success("This patient has a low likelihood of being readmitted within 30 days.")
            
            # Patient summary
            st.markdown("### 📋 Patient Summary")
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown(f"""
                **Demographics:**
                - Age: {age} years
                - Sex: {"Male" if sex == 1 else "Female"}
                - Race: {["", "White", "Black", "Other", "", "Hispanic"][race]}
                
                **Utilization:**
                - Length of Stay: {length_of_stay} days
                - Prior Admissions (6mo): {admissions_past_6mo}
                - Healthcare Costs (6mo): ${cost_past_6mo:,.2f}
                """)
            
            with summary_col2:
                st.markdown(f"""
                **Clinical Factors:**
                - Rx Fills (30d after): {rx_fills_30days_after}
                - ER Visit (7d prior): {"Yes" if had_er_visit_7days_prior else "No"}
                
                **Chronic Conditions:** {risk_factors} condition(s)
                """)
                conditions = []
                if sp_alzhdmta: conditions.append("Alzheimer's")
                if sp_chf: conditions.append("CHF")
                if sp_chrnkidn: conditions.append("Kidney Disease")
                if sp_cncr: conditions.append("Cancer")
                if sp_copd: conditions.append("COPD")
                if sp_depressn: conditions.append("Depression")
                if sp_diabetes: conditions.append("Diabetes")
                if sp_ischmcht: conditions.append("Ischemic Heart")
                
                if conditions:
                    st.markdown("- " + "\n- ".join(conditions))
            
            # Recommendations based on prediction
            st.markdown("### 💡 Recommended Actions")
            
            if prediction == 1:  # Predicted readmission
                if prob_readmit > 0.7:
                    st.markdown("""
                    **🔴 HIGH RISK - INTENSIVE INTERVENTION REQUIRED:**
                    - **URGENT:** Schedule follow-up within **3-5 days** of discharge
                    - Assign dedicated care coordinator for close monitoring
                    - Arrange home health services or transitional care
                    - Complete comprehensive medication reconciliation
                    - Provide detailed discharge education with teach-back
                    - Set up daily check-in calls for first week
                    - Consider skilled nursing facility or rehab if appropriate
                    - Ensure transportation arranged for follow-up visits
                    """)
                else:
                    st.markdown("""
                    **⚠️ ELEVATED RISK - ENHANCED FOLLOW-UP:**
                    - Schedule follow-up within **7-10 days** of discharge
                    - Review discharge instructions thoroughly with patient/family
                    - Arrange telemedicine check-in within 48-72 hours
                    - Ensure patient understands warning signs to watch for
                    - Verify medication adherence plan and fill prescriptions
                    - Consider care coordination program enrollment
                    - Provide 24/7 contact number for questions
                    """)
            else:
                st.markdown("""
                **✅ STANDARD CARE PROTOCOL:**
                - Schedule routine follow-up within **2-4 weeks**
                - Provide standard discharge instructions
                - Patient education on self-care and medication regimen
                - Give contact information for questions or concerns
                - Ensure understanding of when to seek emergency care
                """)
            
            # Show input data in expander
            with st.expander("🔍 View Input Data"):
                st.dataframe(input_data, use_container_width=True)
            
            # Model info
            with st.expander("🤖 Model Information"):
                st.markdown(f"""
                **Model Details:**
                - Model Version: {MODEL_VERSION}
                - Endpoint: readmission
                - Threshold Used: {threshold_used:.3f}
                - Probability Returned: {prob_readmit:.4f}
                - Prediction: {prediction}
                
                **Response Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                """)
                
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>⚕️ For clinical decision support only. Not a substitute for professional medical judgment.</small><br>
    <small>Model: {MODEL_VERSION} | Threshold: {threshold:.3f} | Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small>
</div>
""", unsafe_allow_html=True)