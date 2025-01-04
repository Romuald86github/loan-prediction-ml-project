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
    """
    Render the home page where users can input their loan information.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submissions for loan status prediction.
    """
    try:
        # Extract features from form inputs
        features = {
            'Gender': request.form.get('Gender'),
            'Married': request.form.get('Married'),
            'Dependents': request.form.get('Dependents'),
            'Education': request.form.get('Education'),
            'LoanAmount': float(request.form.get('LoanAmount')),
            'Credit_History': float(request.form.get('Credit_History')),
            'Property_Area': request.form.get('Property_Area')
        }

        # Ensure no fields are left empty
        if not all(features.values()):
            raise ValueError("All fields must be filled out.")

        # Create a DataFrame for the input
        df = pd.DataFrame([features])

        # Preprocess the input data using the pipeline
        X = preprocessing_pipeline.transform(df)

        # Make prediction using the loaded model
        prediction = model.predict(X)

        # Decode the prediction result
        result = preprocessing_pipeline.target_encoder.inverse_transform(prediction)[0]

        # Assign a CSS class for styling based on the result
        result_class = 'approved' if result == 'Y' else 'rejected'

        # Render the home page with the prediction result
        return render_template(
            'index.html',
            result=result,
            result_class=result_class
        )
    except ValueError as ve:
        # Handle validation errors
        return render_template(
            'index.html',
            error=str(ve)
        )
    except Exception as e:
        # Handle any other errors gracefully
        return render_template(
            'index.html',
            error=f"An error occurred: {str(e)}"
        )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Provide a JSON API endpoint for loan status prediction.
    """
    try:
        # Extract JSON data from the request
        data = request.json

        # Validate required fields
        required_fields = [
            'Gender', 'Married', 'Dependents', 'Education',
            'LoanAmount', 'Credit_History', 'Property_Area'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Create a DataFrame for the input
        df = pd.DataFrame([data])

        # Preprocess the input data
        X = preprocessing_pipeline.transform(df)

        # Make prediction
        prediction = model.predict(X)

        # Decode the prediction result
        result = preprocessing_pipeline.target_encoder.inverse_transform(prediction)[0]

        # Return the result as JSON
        return jsonify({
            'status': 'success',
            'loan_status': result,
            'message': 'Prediction successful'
        })
    except Exception as e:
        # Handle API errors gracefully
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Prediction failed'
        }), 400

@app.errorhandler(404)
def page_not_found(e):
    """
    Handle 404 errors gracefully.
    """
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """
    Handle 500 errors gracefully.
    """
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=8000, debug=False)