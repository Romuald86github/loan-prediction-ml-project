from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from src.preprocessing_pipeline import PreprocessingPipeline

app = Flask(__name__)

# Load the model and preprocessing pipeline
model = joblib.load('models/Random_Forest_model.pkl')
preprocessing_pipeline = PreprocessingPipeline.load_pipeline()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        
        return render_template(
            'index.html',
            result=result,
            result_class=result_class
        )
    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {str(e)}"
        )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        X = preprocessing_pipeline.transform(df)
        prediction = model.predict(X)
        result = preprocessing_pipeline.target_encoder.inverse_transform(prediction)[0]
        return jsonify({'loan_status': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)