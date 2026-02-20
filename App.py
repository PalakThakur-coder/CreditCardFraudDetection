from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model = None
scaler = None

try:
    model_path = os.path.join(os.path.dirname(__file__), 'credit.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'sc.pkl')
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
except Exception as e:
    print(f"Warning: Could not load model files: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('index.html', prediction_text="Error: Model not loaded")
    
    try:
        input_features = [float(x) for x in request.form.values()]
        features = np.array(input_features).reshape(1,-1)

        # Scale only Amount (last column)
        features[0][29] = scaler.transform([[features[0][29]]])[0][0]

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Fraud Transaction"
        else:
            result = "Normal Transaction"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
