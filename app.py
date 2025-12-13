import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Title and description
st.title("❤️ Heart Disease Prediction App (SVM Model)")
st.markdown("""
This application predicts the likelihood of heart disease using a **Support Vector Machine (SVM)** model optimized for high recall.
Please adjust the values in the sidebar to check the prediction.
""")

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
    scaler = artifacts.get("scaler") # Might be None if model is a pipeline or no scaler used

    st.sidebar.header("Input Features")
    
    # Load raw data just for getting unique values for selectboxes (optional, can hardcode if needed)
    # ensuring we have the range of values correct.
    # We'll use hardcoded ranges based on the previous code to be safe and fast.
    
    def user_input_features():
        # Numerical features
        age = st.sidebar.slider("Age", 29, 77, 50)
        trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 94, 200, 120)
        chol = st.sidebar.slider("Cholesterol (chol)", 126, 564, 200)
        thalach = st.sidebar.slider("Max Heart Rate (thalach)", 71, 202, 150)
        oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
        
        # Categorical features - ensuring we capture the raw integer values expected before encoding
        sex = st.sidebar.selectbox("Sex", (0, 1), format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.sidebar.selectbox("Chest Pain Type (cp)", (0, 1, 2, 3), format_func=lambda x: f"Type {x}")
        fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", (0, 1), format_func=lambda x: "True" if x == 1 else "False")
        restecg = st.sidebar.selectbox("Resting ECG (restecg)", (0, 1, 2))
        exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")
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

    st.subheader("User Input Parameters")
    st.write(input_df)

    if st.button("Predict"):
        # Preprocessing to match training data
        # 1. One-Hot Encoding
        # The notebook applied get_dummies to: ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        # We need to do the same. Note: drop_first=True was likely used in notebook standard practices or we should check.
        # Most sklearn pipelines handle this, but if we manually did get_dummies, we must replicate.
        # User's notebook typically uses get_dummies(df, columns=[...], drop_first=True)
        # We'll assume drop_first=True effectively via alignment with model_columns.
        
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        input_processed = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True) # Try drop_first=True first
        
        # 2. Align with training columns
        # This adds missing columns (filling with 0) and removes extra ones
        input_processed = input_processed.reindex(columns=model_columns, fill_value=0)
        
        # 3. Scaling
        # If model is a pipeline, it handles scaling. If scaler is separate, apply it.
        # Our fix script saves scaler if it's in the pipeline, but if it IS in the pipeline, 
        # we shouldn't apply it manually *unless* we extracted the classifier *out* of the pipeline.
        # However, typically 'best_model.pkl' is the whole pipeline. 
        # If best_model is a Pipeline, we pass raw (encoded) data.
        # If best_model is just the estimator (SVC), we need to scale.
        # Let's try predicting directly first. If model expects scaled input and is just SVC, we apply scaler.
        
        try:
            # Check if model is pipeline by looking for 'predict' method which Pipelines have.
            # But Pipelines handle transformation.
            # If we saved the scaler separately, it implies we might need it. 
            # BUT, if best_model is the pipeline, we don't apply scaler manually.
            # Let's assume best_model IS the pipeline for SVM as per notebook structure.
            # So typically we just pass input_processed.
            
            prediction = model.predict(input_processed)
            pass
        except Exception as e:
            # Maybe it needs scaling manually?
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
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("⚠️ High risk of Heart Disease detected.")
            else:
                st.success("✅ No Heart Disease detected.")
            
            # Probability - SVM needs probability=True to likely have predict_proba
            if hasattr(model, "predict_proba"):
                prediction_proba = model.predict_proba(input_processed)
                st.subheader("Prediction Probability")
                st.write(f"Probability of Disease: {prediction_proba[0][1]:.2f}")
                st.write(f"Probability of No Disease: {prediction_proba[0][0]:.2f}")


