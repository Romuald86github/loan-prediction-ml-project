import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.preprocessing_pipeline import PreprocessingPipeline

def load_model_and_pipeline():
    """Load the best trained model and preprocessing pipeline."""
    try:
        # Load the best model (adjust filename as needed)
        model = joblib.load('models/Random_Forest_model.pkl')
        
        # Load the preprocessing pipeline
        preprocessing = PreprocessingPipeline()
        preprocessing.load_pipeline()
        
        return model, preprocessing
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_loan_eligibility(model, preprocessing, input_data):
    """Predict loan eligibility using the trained model and preprocessing pipeline."""
    try:
        # Preprocess the input data
        X_processed = preprocessing.transform(input_data)
        
        # Make prediction
        prediction = model.predict(X_processed)
        
        # Convert prediction back to original labels
        target_encoder = preprocessing.target_encoder
        prediction_label = target_encoder.inverse_transform(prediction.reshape(-1, 1))[0][0]
        
        return prediction_label
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    # Set page title and favicon
    st.set_page_config(page_title="Loan Eligibility Predictor", page_icon=":bank:")
    
    # Page title
    st.title("Loan Eligibility Prediction")
    
    # Load model and preprocessing pipeline
    model, preprocessing = load_model_and_pipeline()
    
    if model is None or preprocessing is None:
        st.stop()
    
    # Create input form
    with st.form("loan_application"):
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Categorical inputs
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
            credit_history = st.selectbox("Credit History", [0, 1])
        
        with col2:
            # Numerical inputs
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0, value=200)
        
        # Submit button
        submitted = st.form_submit_button("Predict Loan Eligibility")
    
    # Prediction logic
    if submitted:
        # Prepare input data as DataFrame (removed dropped features)
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [str(dependents)],
            'Education': [education],
            'LoanAmount': [loan_amount],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })
        
        # Make prediction
        prediction = predict_loan_eligibility(model, preprocessing, input_data)
        
        # Display results
        if prediction:
            st.subheader("Prediction Result")
            if prediction == 'Y':
                st.success("Congratulations! 🎉 Your loan application is likely to be approved.")
            else:
                st.warning("Unfortunately, your loan application may not be approved.")
            
            # Optional: Add probability or confidence (if model supports)
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(preprocessing.transform(input_data))[0]
                st.info(f"Prediction Confidence: {prob.max():.2%}")

if __name__ == "__main__":
    main()