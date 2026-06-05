# Electricity_cost_prediction
Built an Electricity Cost Prediction web app using ANN, TensorFlow, Scikit-learn, and Streamlit. Performed EDA, data preprocessing, feature scaling, model training, and real-time prediction of electricity costs through an interactive dashboard.
# Electricity Consumption Prediction using ANN

## Overview

This project predicts electricity consumption costs using an Artificial Neural Network (ANN). The application combines data analysis, machine learning, and an interactive Streamlit dashboard to provide real-time predictions based on building and environmental factors.

## Features

* Interactive Streamlit web application
* Exploratory Data Analysis (EDA)
* Data preprocessing and feature scaling
* Artificial Neural Network (TensorFlow/Keras)
* Model evaluation using MAE and Loss metrics
* Real-time electricity cost prediction
* Model and scaler persistence using Joblib

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Plotly

## Input Features

* Site Area
* Structure Type
* Water Consumption
* Resident Count
* Air Quality Index
* Recycling Rate
* Utilization Rate
* Issue Resolution Time

## Model Architecture

* Dense Layer (128 neurons, ReLU)
* Dense Layer (64 neurons, ReLU)
* Dense Layer (32 neurons, ReLU)
* Output Layer (1 neuron)

## Installation

```bash
pip install -r requirements.txt
streamlit run Electricity_train.py
```

## Project Workflow

1. Load and explore the dataset
2. Perform preprocessing and encoding
3. Scale features and target values
4. Train ANN model with Early Stopping
5. Evaluate model performance
6. Save model and scalers
7. Make predictions through the Streamlit interface

## Results

The model achieves low prediction error and provides accurate electricity cost estimates based on building characteristics and environmental factors.

## Author

Harsh Jariwala
Aspiring Data Scientist | Machine Learning Enthusiast
