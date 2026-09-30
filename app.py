"""
app.py - Streamlit Application for Industrial Machine Failure Prediction
Learn Depth Academy LLP - Track 1 Capstone Problem 21
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Industrial Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'machine_failure_model.pkl')
PIPELINE_PATH = os.path.join(BASE_DIR, 'model', 'preprocessing_pipeline.pkl')

@st.cache_resource
def load_artifacts():
    """Loads the trained model and preprocessing pipeline."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline

model, pipeline = load_artifacts()

# Sidebar: Project and Model Metadata
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/engine.png", width=80)
    st.title("Project Information")
    st.markdown("""
    **Track 1 – Final Capstone – Problem 21**  
    **Institution:** Learn Depth Academy LLP  
    **Domain:** Industrial Systems  
    **Task:** Binary Classification  
    **Dataset:** UCI AI4I 2020 Predictive Maintenance  
    **DOI:** `10.24432/C5HS5C`  
    """)
    st.divider()
    st.subheader("Selected Model")
    st.info("""
    **Decision Tree Classifier**  
    - Selected for balanced recall (83.82%)
    - Lowest False Negatives (FN=11 / 68 failures)
    - Transparent, rule-based physical decision boundaries
    """)
    st.divider()
    st.warning("""
    ⚠️ **Academic Disclaimer:**  
    This application is an educational prototype developed for academic evaluation and viva demonstration. It is not intended for real-time industrial production deployment.
    """)

# Check for query parameters (for demo testing and screenshot captures)
qp = st.query_params
demo_mode = qp.get("demo", None)
auto_submit = False

default_type = 'L'
default_air = 300.0
default_proc = 310.0
default_rpm = 1500
default_torque = 40.0
default_wear = 50

if demo_mode == "failure":
    default_type = 'L'
    default_air = 302.5
    default_proc = 311.5
    default_rpm = 1300
    default_torque = 68.0
    default_wear = 215
    auto_submit = True
elif demo_mode == "normal":
    default_type = 'M'
    default_air = 300.0
    default_proc = 310.0
    default_rpm = 1550
    default_torque = 38.0
    default_wear = 45
    auto_submit = True

# Header Section
st.title("⚙️ Industrial Machine Failure Prediction")
st.markdown("#### *Machine Failure Risk Prediction using Machine Learning*")
st.markdown("""
Predict whether a milling machine is likely to experience operational failure within an upcoming operating window based on sensor measurements and operating conditions.
""")

st.divider()

if model is None or pipeline is None:
    st.error("Error: Model or preprocessing artifacts not found. Please run `python src/evaluate_model.py` first to generate model artifacts.")
    st.stop()

# Two-column layout for inputs and results
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.subheader("📋 Machine Operational Parameters")
    st.markdown("Provide operational telemetry from machine sensors:")

    # Form with input controls and validation
    with st.form(key="prediction_form"):
        # Type
        type_options = ['L', 'M', 'H']
        type_index = type_options.index(default_type) if default_type in type_options else 0
        m_type = st.selectbox(
            "Product Quality Variant (Type)",
            options=type_options,
            index=type_index,
            help="Quality grade: L (Low, 60% of variants), M (Medium, 30%), H (High, 10%)."
        )
        
        c1, c2 = st.columns(2)
        with c1:
            air_temp = st.number_input(
                "Air Temperature [K]",
                min_value=290.0,
                max_value=310.0,
                value=float(default_air),
                step=0.1,
                help="Ambient operational temperature in Kelvin (Typical dataset range: 295.3 - 304.5 K)."
            )
            rot_speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=1000,
                max_value=3000,
                value=int(default_rpm),
                step=10,
                help="Spindle rotational speed in RPM (Typical dataset range: 1168 - 2886 rpm)."
            )
            tool_wear = st.number_input(
                "Tool Wear [min]",
                min_value=0,
                max_value=300,
                value=int(default_wear),
                step=1,
                help="Cumulative machining duration with current cutting tool in minutes (Max in dataset: 253 min)."
            )

        with c2:
            proc_temp = st.number_input(
                "Process Temperature [K]",
                min_value=300.0,
                max_value=320.0,
                value=float(default_proc),
                step=0.1,
                help="Internal cutting process temperature in Kelvin (Typical dataset range: 305.7 - 313.8 K)."
            )
            torque = st.number_input(
                "Torque [Nm]",
                min_value=0.0,
                max_value=100.0,
                value=float(default_torque),
                step=0.5,
                help="Spindle torque in Newton-meters (Typical dataset range: 3.8 - 76.6 Nm)."
            )

        submit_btn = st.form_submit_button("🔍 Predict Machine Failure Status", use_container_width=True)
        if auto_submit:
            submit_btn = True

# Validation logic and prediction
with col_result:
    st.subheader("📊 Diagnostic Output")

    if submit_btn:
        # Check domain constraint: Process Temperature must typically be greater than Air Temperature
        if proc_temp < air_temp:
            st.warning("⚠️ Warning: Process temperature is normally higher than ambient air temperature during active machining.")

        # Prepare input dataframe
        input_data = pd.DataFrame([{
            'Type': m_type,
            'Air temperature [K]': float(air_temp),
            'Process temperature [K]': float(proc_temp),
            'Rotational speed [rpm]': int(rot_speed),
            'Torque [Nm]': float(torque),
            'Tool wear [min]': int(tool_wear)
        }])

        try:
            # Transform using the trained ColumnTransformer
            X_input_proc = pipeline.transform(input_data)
            
            # Generate prediction and probabilities
            pred = model.predict(X_input_proc)[0]
            probs = model.predict_proba(X_input_proc)[0]
            fail_prob = probs[1] * 100.0
            no_fail_prob = probs[0] * 100.0

            st.write("---")
            if pred == 1:
                st.error("### 🚨 Prediction: Machine Failure")
                st.metric(
                    label="Failure Probability",
                    value=f"{fail_prob:.2f}%",
                    delta="Critical Risk Level",
                    delta_color="inverse"
                )
                st.progress(min(fail_prob / 100.0, 1.0))
                st.markdown("""
                **Risk Assessment: HIGH RISK / FAILURE LIKELY**  
                The combination of operating variables (e.g. excessive torque, severe tool wear, or abnormal thermal delta) indicates that the machine is at elevated risk of immediate breakdown.
                
                **Recommended Action:**
                - Halt spindle operation immediately.
                - Inspect tool wear condition and replace cutting insert if wear threshold exceeded.
                - Verify lubrication and coolant flow to reduce thermal dissipation stress.
                """)
            else:
                st.success("### ✅ Prediction: No Machine Failure")
                st.metric(
                    label="Failure Probability",
                    value=f"{fail_prob:.2f}%",
                    delta="Normal Operating Condition"
                )
                st.progress(min(fail_prob / 100.0, 1.0))
                st.markdown("""
                **Risk Assessment: LOW RISK / NOMINAL STATE**  
                The machine is currently operating within standard parameter envelopes. No immediate operational failure is indicated.
                
                **Recommended Action:**
                - Continue standard production monitoring.
                - Adhere to regular maintenance schedules.
                """)

            st.write("---")
            st.markdown("#### Input Telemetry Summary")
            st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction pipeline execution: {str(e)}")
    else:
        st.info("👈 Enter the operating parameters in the panel on the left and click **Predict Machine Failure Status** to run model inference.")

# Bottom Section: Project Information & Sensor Mapping
st.divider()
exp1, exp2, exp3 = st.columns(3)

with exp1:
    with st.expander("📖 Sensor vs Capstone Input Mapping", expanded=True):
        st.markdown("""
        **Generic Capstone Inputs vs Actual Dataset Features:**
        - **Temperature:** Captured via `Air temperature [K]` and `Process temperature [K]`.
        - **Vibration & Pressure:** Directly manifested as mechanical reaction forces captured via `Torque [Nm]` and `Rotational speed [rpm]`.
        - **Runtime & Maintenance Gap:** Monitored cumulatively via continuous `Tool wear [min]`.
        *Note: Features adhere strictly to UCI AI4I 2020 without artificial fabrication.*
        """)

with exp2:
    with st.expander("🔬 Model Performance Verification", expanded=True):
        st.markdown("""
        **Test Set Evaluation (2,000 observations):**
        - **Accuracy:** 93.65%
        - **Recall (Sensitivity):** 83.82% (57 of 68 failures detected)
        - **Precision:** 32.95%
        - **F1-Score:** 0.4730
        - **ROC-AUC:** 0.8654
        - **False Negatives:** 11 / 68
        """)

with exp3:
    with st.expander("🛡️ Leakage Prevention Measures", expanded=True):
        st.markdown("""
        **Excluded Features to Guarantee Zero Leakage:**
        - `UDI` & `Product ID`: Arbitrary identifiers.
        - `TWF`, `HDF`, `PWF`, `OSF`, `RNF`: Post-failure failure mode labels.
        - Preprocessing pipeline was fitted exclusively on the 80% training split.
        """)
