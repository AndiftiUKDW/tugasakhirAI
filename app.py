from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
app = Flask(__name__)
rf_heartattack_loaded = joblib.load('heartattack_revisifinal.pkl')
rf_stroke_loaded = joblib.load('stroke_revisifinal.pkl')
features = joblib.load('features_revisifinal.pkl')
@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = {}
        for feature in features:
            value = request.form.get(feature)
            if value is None or value == '':
                user_data[feature] = 0 
            else:
                try:
                    user_data[feature] = float(value)
                except ValueError:
                    user_data[feature] = 0 

        user_data_df = pd.DataFrame([user_data])

        heartattack_proba = rf_heartattack_loaded.predict_proba(user_data_df)[0]
        heartattack_prediction = rf_heartattack_loaded.predict(user_data_df)[0]
        heartattack_sureness = heartattack_proba[1] if heartattack_prediction == 1 else heartattack_proba[0]
        heartattack_sureness = round(heartattack_sureness * 100, 2)
        stroke_proba = rf_stroke_loaded.predict_proba(user_data_df)[0]
        stroke_prediction = rf_stroke_loaded.predict(user_data_df)[0]
        stroke_sureness = stroke_proba[1] if stroke_prediction == 1 else stroke_proba[0]
        stroke_sureness = round(stroke_sureness * 100, 2)

        return jsonify({
            'heart_attack_prediction': 'Yes' if heartattack_prediction == 1 else 'No',
            'heart_attack_sureness': f"{heartattack_sureness}%",
            'stroke_prediction': 'Yes' if stroke_prediction == 1 else 'No',
            'stroke_sureness': f"{stroke_sureness}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)