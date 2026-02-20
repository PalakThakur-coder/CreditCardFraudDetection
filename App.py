from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('credit.pkl', 'rb'))
scaler = pickle.load(open('sc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

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

if __name__ == "__main__":
    app.run(debug=True)
