from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("best_fraud_detection_model.pkl")

# Define Flask app
app = Flask(__name__)

# Define feature names based on model training
feature_columns = [
    'amt', 'city_pop', 'merch_lat', 'merch_long', 
    'category_food_dining', 'category_gas_transport', 'category_grocery_net',
    'category_grocery_pos', 'category_health_fitness', 'category_home', 'category_kids_pets',
    'category_misc_net', 'category_misc_pos', 'category_personal_care',
    'category_shopping_net', 'category_shopping_pos', 'category_travel'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        features = data.get('features', [])  # Extract feature values
        
        # Check if the correct number of features are provided
        if len(features) != len(feature_columns):
            return jsonify({'error': f'Expected {len(feature_columns)} features, but got {len(features)}'})

        # Convert input data into DataFrame with correct column names
        input_df = pd.DataFrame([features], columns=feature_columns)

        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Return response
        return jsonify({'Fraud Prediction': int(prediction)})  # Convert NumPy int to Python int
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
