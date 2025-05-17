from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')  # Load trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)