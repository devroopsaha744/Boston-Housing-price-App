import streamlit as st
import joblib
import numpy as np

# Load the pre-trained machine learning model
model = joblib.load('Boosting.pkl')

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

def predict_price(CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT):
    # Make a prediction using the loaded model
    prediction = model.predict([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    return prediction[0]

def main():
    st.title('Boston Housing Price Prediction')

    # Input sliders for each feature
    CRIM = st.slider('CRIM', 0.0, 100.0, 0.0)
    ZN = st.slider('ZN', 0.0, 100.0, 0.0)
    INDUS = st.slider('INDUS', 0.0, 100.0, 0.0)
    CHAS = st.slider('CHAS', 0.0, 1.0, 0.0)
    NOX = st.slider('NOX', 0.0, 1.0, 0.0)
    RM = st.slider('RM', 0.0, 10.0, 0.0)
    AGE = st.slider('AGE', 0.0, 100.0, 0.0)
    DIS = st.slider('DIS', 0.0, 10.0, 0.0)
    RAD = st.slider('RAD', 0.0, 100.0, 0.0)
    TAX = st.slider('TAX', 0.0, 1000.0, 0.0)
    PTRATIO = st.slider('PTRATIO', 0.0, 50.0, 0.0)
    B = st.slider('B', 0.0, 1000.0, 0.0)
    LSTAT = st.slider('LSTAT', 0.0, 50.0, 0.0)

    # Button to make the prediction
    if st.button('Predict'):
        prediction = predict_price(CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)
        st.success(f'Predicted Median House Price: {prediction}')

if __name__ == '__main__':
    main()
