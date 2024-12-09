from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Load the trained model (ensure the model is saved as a pickle file)
model_path = os.path.join(os.path.dirname(__file__), 'gbm.pkl')
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

# Serve the HTML page on the root route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route for receiving data and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.form

        # Validate and map categorical inputs
        gender_mapping = {'Male': 1, 'Female': 0}
        occupation_mapping = {
            'Others': 7, 'Doctor': 1, 'Teacher': 3, 'Nurse': 0, 
            'Engineer': 2, 'Accountant': 5, 'Lawyer': 4, 'Salesperson': 6
        }
        bmi_mapping = {'Normal Weight': 0, 'Overweight': 1, 'Obese': 2}

        gender = gender_mapping.get(data.get('gender'))
        occupation = occupation_mapping.get(data.get('occupation'))
        bmi = bmi_mapping.get(data.get('bmi_category'))

        # Validate numerical inputs
        age = float(data.get('age'))
        sleep_duration = float(data.get('sleep_duration'))
        quality_sleep = int(data.get('quality_sleep'))
        physical_activity = int(data.get('physical_activity'))
        stress_level = int(data.get('stress_level'))
        heart_rate = int(data.get('heart_rate'))
        daily_steps = int(data.get('daily_steps'))
        systolic = int(data.get('systolic'))
        diastolic = int(data.get('diastolic'))

        # Check for invalid inputs
        if gender is None or occupation is None or bmi is None:
            return jsonify({"error": "Invalid categorical input"}), 400

        # Prepare features for prediction
        features = np.array([[gender, age, occupation, sleep_duration, quality_sleep,
                              physical_activity, stress_level, bmi, heart_rate,
                              daily_steps, systolic, diastolic]])

        # Make prediction
        prediction = model.predict(features)

        # Map prediction result to sleep disorder label
        prediction_label = {0: "Insomnia", 1: "No Disorder", 2: "Sleep Apnea"}.get(prediction[0])

        return jsonify({"Sleep Health": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
