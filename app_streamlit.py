import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

from src.preprocessing_pipeline import PreprocessingPipeline

def load_model_and_pipeline():
    """
    Load the saved preprocessing pipeline and best model
    """
    # Check if preprocessing pipeline exists
    pipeline_path = 'models/preprocessing_pipeline.pkl'
    model_path = 'models/Random_Forest_model.pkl'

    if not os.path.exists(pipeline_path):
        st.error(f"Preprocessing pipeline not found at {pipeline_path}. Please train the model first.")
        return None, None

    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None, None

    # Load preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline.load_pipeline()

    # Load the saved model
    model = joblib.load(model_path)
    return preprocessing_pipeline, model

def safe_transform(pipeline, input_data):
    """
    Safely transform input data, handling potential numerical issues
    """
    try:
        # First, ensure no infinite or extremely large values
        for col in input_data.select_dtypes(include=['float64', 'int64']).columns:
            input_data[col] = input_data[col].replace([np.inf, -np.inf], np.nan)
            input_data[col] = input_data[col].fillna(input_data[col].mean())

        # Transform the data
        preprocessed_data = pipeline.transform(input_data)

        # Ensure preprocessed_data is a 2D numpy array
        if isinstance(preprocessed_data, tuple):
            preprocessed_data = preprocessed_data[0]
        
        if preprocessed_data.ndim == 1:
            preprocessed_data = preprocessed_data.reshape(1, -1)

        # Additional safety check for transformed data
        if not np.isfinite(preprocessed_data).all():
            st.warning("Warning: Preprocessing resulted in non-finite values. Using mean imputation.")
            preprocessed_data = np.nan_to_num(preprocessed_data, nan=0.0, posinf=0.0, neginf=0.0)

        return preprocessed_data

    except Exception as e:
        st.error(f"Error in data preprocessing: {e}")
        return None

def main():
    # Set page title and icon
    st.set_page_config(page_title="Loan Approval Prediction", page_icon=":bank:")

    # Title of the application
    st.title("Loan Approval Prediction")
    st.write("Predict whether a loan application will be approved or not.")

    # Load preprocessing pipeline and model
    preprocessing_pipeline, model = load_model_and_pipeline()

    if preprocessing_pipeline is None or model is None:
        return

    # Input form for loan application details
    with st.form("loan_application"):
        # Categorical inputs
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        credit_history = st.selectbox("Credit History", ["1.0", "0.0"])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

        # Numeric input
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, max_value=500, value=150)

        # Submit button
        submitted = st.form_submit_button("Predict Loan Approval")

        if submitted:
            # Prepare input data as DataFrame
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Married': [married],
                'Dependents': [dependents],
                'Education': [education],
                'LoanAmount': [loan_amount],
                'Credit_History': [float(credit_history)],
                'Property_Area': [property_area]
            })

            # Preprocess the input data with safety checks
            preprocessed_data = safe_transform(preprocessing_pipeline, input_data)

            if preprocessed_data is not None:
                try:
                    # Make prediction
                    prediction = model.predict(preprocessed_data)

                    # Display prediction
                    if prediction[0] == 1:
                        st.success("Congratulations! 🎉 Your loan is likely to be approved.")
                    else:
                        st.warning("Sorry, your loan application might not be approved.")

                    # Optional: Show prediction probability
                    prediction_proba = model.predict_proba(preprocessed_data)
                    st.write(f"Probability of Approval: {prediction_proba[0][1]*100:.2f}%")

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.error("Could not preprocess the input data")

if __name__ == "__main__":
    main()