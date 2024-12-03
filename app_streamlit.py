import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.preprocessing_pipeline import PreprocessingPipeline

def load_model_and_pipeline():
    """Load the best trained model and preprocessing pipeline."""
    try:
        model = joblib.load('models/Random_Forest_model.pkl')
        
        preprocessing = PreprocessingPipeline()
        pipeline_components = preprocessing.load_pipeline()
        preprocessing.fitted_categorical_encoders = pipeline_components['categorical_encoders']
        preprocessing.target_encoder = pipeline_components['target_encoder']
        preprocessing.scaler = pipeline_components['scaler']
        preprocessing.smote = pipeline_components['smote']
        
        # Get the feature names from the scaler
        feature_names = preprocessing.scaler.feature_names_in_
        
        return model, preprocessing, feature_names
    except Exception as e:
        st.error(f"Error loading model or pipeline: {e}")
        return None, None, None

def predict_loan_eligibility(model, preprocessing, feature_names, input_data):
    """Predict loan eligibility using the trained model and preprocessing pipeline."""
    try:
        # Ensure the input data has all required columns in the correct order
        input_data = input_data.reindex(columns=feature_names)
        
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
    
    model, preprocessing, feature_names = load_model_and_pipeline()
    
    if model is None or preprocessing is None or feature_names is None:
        st.stop()
    
    st.write("Expected features:", feature_names)  # Debug information
    
    with st.form("loan_application"):
        col1, col2 = st.columns(2)
        
        input_data = {}
        for i, feature in enumerate(feature_names):
            with col1 if i % 2 == 0 else col2:
                if feature in ['Gender', 'Married', 'Education', 'Property_Area']:
                    options = {
                        'Gender': ["Male", "Female"],
                        'Married': ["Yes", "No"],
                        'Education': ["Graduate", "Not Graduate"],
                        'Property_Area': ["Urban", "Rural", "Semiurban"]
                    }
                    input_data[feature] = st.selectbox(feature, options[feature])
                elif feature == 'Dependents':
                    input_data[feature] = st.selectbox(feature, ["0", "1", "2", "3+"])
                elif feature == 'Credit_History':
                    input_data[feature] = st.selectbox(feature, ["1", "0"])
                else:  # Assume numeric for any other feature
                    input_data[feature] = st.number_input(feature, value=0)
        
        submitted = st.form_submit_button("Predict Loan Eligibility")
    
    if submitted:
        input_df = pd.DataFrame([input_data])
        
        prediction = predict_loan_eligibility(model, preprocessing, feature_names, input_df)
        
        if prediction:
            st.subheader("Prediction Result")
            if prediction == 'Y':
                st.success("Congratulations! 🎉 Your loan application is likely to be approved.")
            else:
                st.warning("Unfortunately, your loan application may not be approved.")
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(preprocessing.transform(input_df))[0]
                st.info(f"Prediction Confidence: {prob.max():.2%}")

if __name__ == "__main__":
    main()