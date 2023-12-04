from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('Boosting.pkl')

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    CRIM = float(request.args.get('CRIM')) if request.args.get('CRIM') is not None else 0.0
    ZN = float(request.args.get('ZN')) if request.args.get('ZN') is not None else 0.0
    INDUS = float(request.args.get('INDUS')) if request.args.get('INDUS') is not None else 0.0
    CHAS = float(request.args.get('CHAS')) if request.args.get('CHAS') is not None else 0.0
    NOX = float(request.args.get('NOX')) if request.args.get('NOX') is not None else 0.0
    RM = float(request.args.get('RM')) if request.args.get('RM') is not None else 0.0
    AGE = float(request.args.get('AGE')) if request.args.get('AGE') is not None else 0.0
    DIS = float(request.args.get('DIS')) if request.args.get('DIS') is not None else 0.0
    RAD = float(request.args.get('RAD')) if request.args.get('RAD') is not None else 0.0
    TAX = float(request.args.get('TAX')) if request.args.get('TAX') is not None else 0.0
    PTRATIO = float(request.args.get('PTRATIO')) if request.args.get('PTRATIO') is not None else 0.0
    B = float(request.args.get('B')) if request.args.get('B') is not None else 0.0
    LSTAT = float(request.args.get('LSTAT')) if request.args.get('LSTAT') is not None else 0.0

    prediction = model.predict([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    return f'Predicted Median House Price: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)

