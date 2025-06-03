import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import time
from apscheduler.schedulers.background import BackgroundScheduler

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration from .env
SECRET_TOKEN = os.getenv('SECRET_TOKEN')
MODEL_PATH = os.getenv('MODEL_PATH', 'stroke_model.pkl')  # Default fallback
PREDICTION_THRESHOLD = float(os.getenv('PREDICTION_THRESHOLD', 0.35))
PORT = int(os.getenv('PORT', 5000))  # Default port 5000 if not specified

# Load model with error handling
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {str(e)}")

expected_features = [
    "Disease Free (Months)",
    "Person Neoplasm Status_WITH TUMOR",
    "Person Neoplasm Status_TUMOR FREE",
    "New Neoplasm Event Post Initial Therapy Indicator_NO",
    "Diagnosis Age",
    "New Neoplasm Event Post Initial Therapy Indicator_YES",
    "Patient Smoking History Category",
    "UICC TNM Tumor Stage Code_T2a",
    "Karnofsky Performance Score",
    "UICC TNM Tumor Stage Code_T2",
    "Prior Cancer Diagnosis Occurence_No",
    "UICC TNM Tumor Stage Code_T3b"
]

@app.route('/')
def home():
     return "✅ Stroke Prediction API Ready"

@app.route('/predict', methods=['POST'])
def predict():
    # Token check
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Missing or malformed token'}), 401
    
    if auth_header.split(' ')[1] != SECRET_TOKEN:
        return jsonify({'error': 'Invalid token'}), 401

    # Input validation
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    # print(data)
    missing = [f for f in expected_features if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    # Prediction
    try:
        input_data = [data[feature] for feature in expected_features]
        input_array = np.array(input_data).reshape(1, -1)
        proba = model.predict_proba(input_array)[0][1]
        
        return jsonify({
            'prediction': int(proba >= PREDICTION_THRESHOLD),
            'probability': round(proba, 3),
            'threshold_used': PREDICTION_THRESHOLD
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Background scheduler to reload the website every 14 minutes
MAX_RETRIES = 3  # Maximum number of retries
RETRY_DELAY = 5  # Delay between retries in seconds

def reload_website():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get("https://mon-projet-flask-6.onrender.com/")
            print(f"Reloaded at {response.status_code}: {response.reason}")
            return  # Exit the function if successful
        except requests.RequestException as e:
            print(f"Error reloading (attempt {retries + 1}): {e}")
            retries += 1
            time.sleep(RETRY_DELAY)
    print("All retry attempts failed. Will try again in the next cycle.")

scheduler = BackgroundScheduler()
scheduler.add_job(reload_website, 'interval', minutes=14)
scheduler.start()

if __name__ == '__main__':
    # Only for development
    app.run(host='0.0.0.0', port=PORT, debug=os.getenv('FLASK_DEBUG') == 'True')
    # app.run(debug=os.getenv('FLASK_DEBUG', False))