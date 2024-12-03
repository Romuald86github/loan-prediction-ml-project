from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from preprocessing_pipeline import PreprocessingPipeline

app = Flask(__name__)

# Load the model and preprocessing pipeline
model = joblib.load('models/Random_Forest_model.pkl')
preprocessing_pipeline = PreprocessingPipeline.load_pipeline()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        features = {
            'Gender': request.form.get('Gender'),
            'Married': request.form.get('Married'),
            'Dependents': request.form.get('Dependents'),
            'Education': request.form.get('Education'),
            'LoanAmount': float(request.form.get('LoanAmount')),
            'Credit_History': float(request.form.get('Credit_History')),
            'Property_Area': request.form.get('Property_Area')
        }

        # Create a DataFrame
        df = pd.DataFrame([features])

        # Preprocess the input
        X = preprocessing_pipeline.transform(df)

        # Make prediction
        prediction = model.predict(X)

        # Decode prediction
        result = preprocessing_pipeline.target_encoder.inverse_transform(prediction)[0]

        # Add class for color-coding
        result_class = 'approved' if result == 'Y' else 'rejected'

        return render_template('index.html', prediction_text=f'Loan Status: <span class="{result_class}">{result}</span>')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Get JSON data from request
    data = request.json

    # Create a DataFrame
    df = pd.DataFrame([data])

    # Preprocess the input
    X = preprocessing_pipeline.transform(df)

    # Make prediction
    prediction = model.predict(X)

    # Decode prediction
    result = preprocessing_pipeline.target_encoder.inverse_transform(prediction)[0]

    return jsonify({'loan_status': result})

if __name__ == '__main__':
    app.run(debug=True)