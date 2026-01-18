import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Custom CSS for dark theme with visible text
st.markdown("""
<style>
    /* Main app background - deep dark gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Main content area */
    .main .block-container {
        background: transparent;
        padding: 2rem 3rem;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        color: #e94560 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 0 0 10px rgba(233, 69, 96, 0.3);
    }
    
    /* Main title specific styling */
    h1 {
        background: linear-gradient(90deg, #e94560, #ff6b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Paragraph and general text */
    p, .stMarkdown, span, label, .stText {
        color: #eaeaea !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 2px solid #e94560;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #ff6b9d !important;
        text-align: center;
        padding: 0.5rem;
        border-bottom: 1px solid #e94560;
        margin-bottom: 1rem;
    }
    
    [data-testid="stSidebar"] label {
        color: #a0d2eb !important;
        font-weight: 500;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #e94560, #ff6b9d) !important;
    }
    
    .stSlider [data-testid="stThumbValue"] {
        color: #ffffff !important;
        font-weight: bold;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #1f1f35 !important;
        border: 1px solid #e94560 !important;
        color: #ffffff !important;
    }
    
    .stSelectbox label {
        color: #a0d2eb !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #e94560 0%, #ff6b9d 100%);
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 25px;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.6);
        background: linear-gradient(135deg, #ff6b9d 0%, #e94560 100%);
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background-color: #1f1f35 !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stDataFrame [data-testid="stDataFrame"] {
        background-color: #1f1f35 !important;
    }
    
    [data-testid="stDataFrame"] div {
        color: #eaeaea !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, #1d5c2e 0%, #2a7a3f 100%);
        color: #90ee90 !important;
        border: 1px solid #32cd32;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stSuccess p {
        color: #90ee90 !important;
        font-weight: 600;
    }
    
    /* Error message styling */
    .stError, .stAlert[data-baseweb="notification"][kind="negative"] {
        background: linear-gradient(135deg, #5c1d1d 0%, #7a2a2a 100%);
        color: #ff9999 !important;
        border: 1px solid #ff6b6b;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stError p {
        color: #ff9999 !important;
        font-weight: 600;
    }
    
    /* Card-like containers */
    .stSubheader, h2 {
        background: linear-gradient(90deg, rgba(233, 69, 96, 0.2), transparent);
        padding: 0.5rem 1rem;
        border-left: 4px solid #e94560;
        border-radius: 0 10px 10px 0;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(31, 31, 53, 0.8);
        border: 1px solid #e94560;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetric"] label {
        color: #a0d2eb !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e94560 !important;
        font-weight: bold;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #e94560;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #ff6b9d;
    }
    
    /* Input widgets text visibility */
    .stNumberInput input, .stTextInput input {
        color: #ffffff !important;
        background-color: #1f1f35 !important;
        border: 1px solid #e94560 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(31, 31, 53, 0.8) !important;
        color: #a0d2eb !important;
        border: 1px solid #e94560;
        border-radius: 10px;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(31, 31, 53, 0.6) !important;
        border: 1px solid #e94560;
        border-top: none;
        color: #eaeaea !important;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #1d3d5c 0%, #2a547a 100%);
        color: #a0d2eb !important;
        border: 1px solid #4da6ff;
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        border-color: #e94560 !important;
        opacity: 0.5;
    }
    
    /* Table in dataframes */
    table {
        background-color: #1f1f35 !important;
    }
    
    th {
        background-color: #2a2a45 !important;
        color: #e94560 !important;
    }
    
    td {
        color: #eaeaea !important;
        background-color: #1f1f35 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title with emoji heart
st.markdown("<h1>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)

# Animated description section
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(31, 31, 53, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
    border: 1px solid #e94560;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0 2rem 0;
    box-shadow: 0 8px 32px rgba(233, 69, 96, 0.2);
">
    <p style="color: #eaeaea; font-size: 1.1rem; line-height: 1.8; margin: 0; text-align: center;">
        🔬 This application predicts the likelihood of heart disease using a 
        <span style="color: #e94560; font-weight: bold;">Support Vector Machine (SVM)</span> 
        model optimized for high recall.<br>
        📊 Adjust the values in the sidebar to check your prediction.
    </p>
</div>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        if os.path.exists("artifacts/best_model.pkl"):
            with open("artifacts/best_model.pkl", "rb") as f:
                artifacts["model"] = pickle.load(f)
        else:
            st.error("Model file 'artifacts/best_model.pkl' not found.")
            return None

        if os.path.exists("artifacts/scaler.pkl"):
            with open("artifacts/scaler.pkl", "rb") as f:
                artifacts["scaler"] = pickle.load(f)
        
        if os.path.exists("artifacts/model_columns.pkl"):
            with open("artifacts/model_columns.pkl", "rb") as f:
                artifacts["columns"] = pickle.load(f)
        else:
            st.error("Model columns file 'artifacts/model_columns.pkl' not found. Cannot ensure feature consistency.")
            return None
            
        return artifacts
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None

artifacts = load_artifacts()

if artifacts is not None:
    model = artifacts["model"]
    model_columns = artifacts["columns"]
    scaler = artifacts.get("scaler")

    # Sidebar header with styling
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #ff6b9d !important; margin: 0;">🩺 Input Features</h2>
        <p style="color: #a0d2eb; font-size: 0.9rem; margin-top: 0.5rem;">Adjust the parameters below</p>
    </div>
    """, unsafe_allow_html=True)
    
    def user_input_features():
        st.sidebar.markdown("### 📈 Vital Signs")
        
        # Numerical features
        age = st.sidebar.slider("Age", 29, 77, 50)
        trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 94, 200, 120)
        chol = st.sidebar.slider("Cholesterol (chol)", 126, 564, 200)
        thalach = st.sidebar.slider("Max Heart Rate (thalach)", 71, 202, 150)
        oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
        
        st.sidebar.markdown("### 🔍 Medical History")
        
        # Categorical features
        sex = st.sidebar.selectbox("Sex", (0, 1), format_func=lambda x: "👩 Female" if x == 0 else "👨 Male")
        cp = st.sidebar.selectbox("Chest Pain Type (cp)", (0, 1, 2, 3), format_func=lambda x: f"Type {x}")
        fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", (0, 1), format_func=lambda x: "✓ True" if x == 1 else "✗ False")
        restecg = st.sidebar.selectbox("Resting ECG (restecg)", (0, 1, 2))
        exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", (0, 1), format_func=lambda x: "✓ Yes" if x == 1 else "✗ No")
        slope = st.sidebar.selectbox("Slope of Peak Exercise ST (slope)", (0, 1, 2))
        ca = st.sidebar.selectbox("Major Vessels Colored by Flourosopy (ca)", (0, 1, 2, 3, 4))
        thal = st.sidebar.selectbox("Thalassemia (thal)", (0, 1, 2, 3))
        
        data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Display input parameters in a styled container
    st.markdown("### 📋 Your Input Parameters")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Age", f"{input_df['age'].values[0]} years")
        st.metric("Resting BP", f"{input_df['trestbps'].values[0]} mmHg")
        st.metric("Cholesterol", f"{input_df['chol'].values[0]} mg/dl")
        
    with col2:
        st.metric("Max Heart Rate", f"{input_df['thalach'].values[0]} bpm")
        st.metric("ST Depression", f"{input_df['oldpeak'].values[0]}")
        st.metric("Sex", "Female" if input_df['sex'].values[0] == 0 else "Male")
        
    with col3:
        st.metric("Chest Pain Type", f"Type {input_df['cp'].values[0]}")
        st.metric("Fasting Blood Sugar", "High" if input_df['fbs'].values[0] == 1 else "Normal")
        st.metric("Exercise Angina", "Yes" if input_df['exang'].values[0] == 1 else "No")

    st.markdown("---")
    
    # Centered predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Heart Disease Risk", use_container_width=True)

    if predict_button:
        # Preprocessing to match training data
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        input_processed = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # Align with training columns
        input_processed = input_processed.reindex(columns=model_columns, fill_value=0)
        
        try:
            prediction = model.predict(input_processed)
        except Exception as e:
            if scaler is not None:
                try:
                    input_scaled = scaler.transform(input_processed)
                    prediction = model.predict(input_scaled)
                except Exception as inner_e:
                    st.error(f"Prediction failed: {e}. Scaler retry failed: {inner_e}")
                    prediction = None
            else:
                st.error(f"Prediction failed: {e}")
                prediction = None

        if prediction is not None:
            st.markdown("### 🎯 Prediction Result")
            
            if prediction[0] == 1:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(139, 0, 0, 0.3) 0%, rgba(178, 34, 34, 0.3) 100%);
                    border: 2px solid #ff4444;
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    margin: 1rem 0;
                    box-shadow: 0 8px 32px rgba(255, 68, 68, 0.3);
                ">
                    <h2 style="color: #ff6b6b !important; margin: 0;">⚠️ High Risk Detected</h2>
                    <p style="color: #ffaaaa; font-size: 1.2rem; margin-top: 1rem;">
                        The model indicates a <strong>high risk</strong> of heart disease.<br>
                        Please consult a healthcare professional for proper evaluation.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(0, 100, 0, 0.3) 0%, rgba(34, 139, 34, 0.3) 100%);
                    border: 2px solid #44ff44;
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    margin: 1rem 0;
                    box-shadow: 0 8px 32px rgba(68, 255, 68, 0.2);
                ">
                    <h2 style="color: #90ee90 !important; margin: 0;">✅ Low Risk</h2>
                    <p style="color: #aaffaa; font-size: 1.2rem; margin-top: 1rem;">
                        The model indicates <strong>no significant risk</strong> of heart disease.<br>
                        Continue maintaining a healthy lifestyle!
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability display
            if hasattr(model, "predict_proba"):
                prediction_proba = model.predict_proba(input_processed)
                
                st.markdown("### 📊 Prediction Confidence")
                
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(0, 100, 0, 0.3) 0%, rgba(34, 139, 34, 0.3) 100%);
                        border: 1px solid #44ff44;
                        border-radius: 10px;
                        padding: 1.5rem;
                        text-align: center;
                    ">
                        <p style="color: #90ee90; font-size: 0.9rem; margin: 0;">Probability of No Disease</p>
                        <h2 style="color: #44ff44 !important; margin: 0.5rem 0 0 0;">{prediction_proba[0][0]:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with prob_col2:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(139, 0, 0, 0.3) 0%, rgba(178, 34, 34, 0.3) 100%);
                        border: 1px solid #ff4444;
                        border-radius: 10px;
                        padding: 1.5rem;
                        text-align: center;
                    ">
                        <p style="color: #ffaaaa; font-size: 0.9rem; margin: 0;">Probability of Disease</p>
                        <h2 style="color: #ff6b6b !important; margin: 0.5rem 0 0 0;">{prediction_proba[0][1]:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p style="color: #888;">
        ⚕️ <em>Disclaimer: This tool is for educational purposes only and should not replace professional medical advice.</em>
    </p>
</div>
""", unsafe_allow_html=True)
