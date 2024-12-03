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
        pipeline_components = preprocessing.load_pipeline()
        preprocessing.fitted_categorical_encoders = pipeline_components['categorical_encoders']
        preprocessing.target_encoder = pipeline_components['target_encoder']
        preprocessing.scaler = pipeline_components['scaler']
        preprocessing.smote = pipeline_components['smote']
        
        return model, preprocessing
    except Exception as e:
        st.error(f"Error loading model or pipeline: {e}")
        return None, None

def predict_loan_eligibility(model, preprocessing, input_data):
    """Predict loan eligibility using the trained model and preprocessing pipeline."""
    try:
        # Ensure the input data has all required columns in the correct order
        required_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Credit_History', 'Property_Area', 'LoanAmount']
        input_data = input_data.reindex(columns=required_columns)
        
        # Preprocess the input data
        X_processed = preprocessing.transform(input_data)
        
        # Make prediction
        prediction = model.predict(X_processed)
        
        # Convert prediction back to original labels
        prediction_label = preprocessing.target_encoder.inverse_transform(prediction.reshape(-1, 1))[0][0]
        
        return prediction_label
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    st.set_page_config(page_title="Loan Eligibility Predictor", page_icon=":bank:")
    st.title("Loan Eligibility Prediction")
    
    model, preprocessing = load_model_and_pipeline()
    
    if model is None or preprocessing is None:
        st.stop()
    
    with st.form("loan_application"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        
        with col2:
            credit_history = st.selectbox("Credit History", ["1", "0"])
            property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=200)
        
        submitted = st.form_submit_button("Predict Loan Eligibility")
    
    if submitted:
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Credit_History': [credit_history],
            'Property_Area': [property_area],
            'LoanAmount': [loan_amount]
        })
        
        prediction = predict_loan_eligibility(model, preprocessing, input_data)
        
        if prediction:
            st.subheader("Prediction Result")
            if prediction == 'Y':
                st.success("Congratulations! 🎉 Your loan application is likely to be approved.")
            else:
                st.warning("Unfortunately, your loan application may not be approved.")
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(preprocessing.transform(input_data))[0]
                st.info(f"Prediction Confidence: {prob.max():.2%}")

if __name__ == "__main__":
    main()