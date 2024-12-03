import streamlit as st
import joblib
import pandas as pd
import os

from preprocessing_pipeline import PreprocessingPipeline

def load_model_and_pipeline():
    """
    Load the saved preprocessing pipeline and best model
    """
    # Load preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline()
    
    # Load the saved model
    model_path = 'models/Random_Forest_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None, None
    
    model = joblib.load(model_path)
    return preprocessing_pipeline, model

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
            
            # Preprocess the input data
            preprocessed_data, _ = preprocessing_pipeline.transform(input_data)
            
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

if __name__ == "__main__":
    main()