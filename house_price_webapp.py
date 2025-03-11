import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import subprocess

# Backend API Endpoints
BENTO_API_URL = "http://localhost:3000/predict"
MONITORING_API_URL = "http://localhost:5001/monitor"
TRAINING_TRIGGER_URL = "http://localhost:8000/"


def push_data_to_db():
    python_path = "C:\\Users\\EdwinVivekN\\AppData\\Local\\Microsoft\\WindowsApps\\python"
    script_path = os.path.join(os.getcwd(), "/airflow/update_datastore.py")
    result = subprocess.run([python_path, script_path], capture_output=True, text=True)
    
    if(result.stdout.endswith("Data pushed successfully\n")):
        st.write("Data pushed successfully")
    else:
        st.write("Error in pushing data to DB")



st.title("MLOps House Price Prediction")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "View Predictions", "Monitor Drift", "Train Model"])

if page == "Upload Data":
    st.header("Upload New Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        if st.button("Save Data"):
            df.to_csv("data/new_data.csv", index=False)
            st.success("Data uploaded successfully!")

elif page == "View Predictions":
    st.header("Predict House Prices")
    st.write("Enter house features for prediction")
    
    # User inputs
    sqft = st.number_input("Square Feet", min_value=500, max_value=10000, value=1500)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
    
    if st.button("Predict Price"):
        response = requests.post(BENTO_API_URL, json={
            "input_data": [
                [
                    sqft,
                    bedrooms,
                    bathrooms
                ]
        ]})
        if response.status_code == 200:
            prediction = response.json()[0]
            st.success(f"Estimated House Price: ${prediction}")
            st.success("Data pushed to feedback.csv")
        else:
            st.error("Error fetching prediction")

    if st.button("Push Data to Feature Store"):
        push_data_to_db()


elif page == "Monitor Drift":
    st.header("Data Drift Monitoring")
    if st.button("Check Drift"):
        response = requests.get(MONITORING_API_URL)
        if response.status_code == 200:
            drift_report = response.json()
            st.json(drift_report)
        else:
            st.error("Error fetching drift report")

elif page == "Train Model":
    st.header("Trigger Model Training")
    if st.button("Start Training"):
        response = requests.get(TRAINING_TRIGGER_URL)
        if response.status_code == 200:
            st.success("Model training started!")
        else:
            st.error("Error starting training")



