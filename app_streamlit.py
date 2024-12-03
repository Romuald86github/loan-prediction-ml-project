import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

from src.preprocessing_pipeline import PreprocessingPipeline

def load_model_and_pipeline():
    pipeline_path = 'models/preprocessing_pipeline.pkl'
    model_path = 'models/Random_Forest_model.pkl'

    if not os.path.exists(pipeline_path) or not os.path.exists(model_path):
        st.error("Model or preprocessing pipeline not found. Please train the model first.")
        return None, None

    preprocessing_pipeline = PreprocessingPipeline.load_pipeline()
    model = joblib.load(model_path)
    return preprocessing_pipeline, model

def main():
    st.set_page_config(page_title="Loan Approval Prediction", page_icon=":bank:")
    st.title("Loan Approval Prediction")
    st.write("Predict whether a loan application will be approved or not.")

    preprocessing_pipeline, model = load_model_and_pipeline()

    if preprocessing_pipeline is None or model is None:
        return

    with st.form("loan_application"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
        credit_history = st.selectbox("Credit History", ["1.0", "0.0"])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

        submitted = st.form_submit_button("Predict Loan Approval")

        if submitted:
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Married': [married],
                'Dependents': [dependents],
                'Education': [education],
                'LoanAmount': [loan_amount],
                'Credit_History': [float(credit_history)],
                'Property_Area': [property_area]
            })

            try:
                preprocessed_data = preprocessing_pipeline.transform(input_data)
                prediction = model.predict(preprocessed_data)
                
                # Decode the prediction
                prediction_label = preprocessing_pipeline.target_encoder.inverse_transform(prediction)
                
                if prediction_label[0] == 'Y':
                    st.success("Congratulations! 🎉 Your loan is likely to be approved.")
                else:
                    st.warning("Sorry, your loan application might not be approved.")

                prediction_proba = model.predict_proba(preprocessed_data)
                st.write(f"Probability of Approval: {prediction_proba[0][1]*100:.2f}%")

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    main()