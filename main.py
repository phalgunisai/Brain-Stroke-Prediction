import streamlit as st
from PIL import Image
import base64
import pygame
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import joblib

# Load the saved scaler and model
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/dt.sav')  # You can change the model file accordingly

# Function to preprocess input data
def preprocess_input(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [Residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })
    input_data['gender'] = input_data['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
    input_data['ever_married'] = input_data['ever_married'].map({'No': 0, 'Yes': 1})
    input_data['work_type'] = input_data['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
    input_data['Residence_type'] = input_data['Residence_type'].map({'Rural': 0, 'Urban': 1})
    input_data['smoking_status'] = input_data['smoking_status'].map({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3})
    input_data['bmi'].fillna(input_data['bmi'].mean(), inplace=True)
    return input_data

# Function to predict stroke risk
def predict_stroke(input_data):
    input_data_std = scaler.transform(input_data)
    prediction = model.predict(input_data_std)
    return prediction[0]

# add_bg_from_local function with width and height parameters
def add_bg_from_local(image_file, width, height):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{encoded_string}');
            background-size: {width} {height};
        }}
        .stSelectbox, .stSlider, .stNumberInput, .stSelectbox, .stMarkdown {{
            color: white; /* Set text color to white */
        }}
        .stButton {{
            color: black; /* Set button text color to black */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the add_bg_from_local function with the desired width and height
add_bg_from_local("brainstroke.jpg", "100%", "100%")

# Streamlit app
def main():
    st.title('Stroke Risk Prediction')
    st.write('Fill out the following information to predict stroke risk.')

    # Form inputs
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.slider('Age', min_value=0, max_value=100, value=50, step=1)
    hypertension = st.selectbox('Hypertension', [0, 1])
    heart_disease = st.selectbox('Heart Disease', [0, 1])
    ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
    work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    Residence_type = st.selectbox('Residence Type', ['Rural', 'Urban'])
    avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=100.0, step=0.1)
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    smoking_status = st.selectbox('Smoking Status', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

    if st.button('Predict'):
        input_data = preprocess_input(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status)
        prediction = predict_stroke(input_data)
        
        # Popup with results
        result_container = st.empty()
        if prediction == 1:
            result_container.error('You are at risk of stroke!')
        else:
            result_container.success('You are not at risk of stroke!')

if __name__ == '__main__':
    main()
