import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Paths
DATA_PATH = 'data/Dataset-of-Diabetes.csv'
MODEL_PATH = 'models/diabetes_rf_model.joblib'
SCALER_PATH = 'models/scaler.joblib'
COLUMNS_PATH = 'models/feature_columns.joblib'
HISTORY_DIR = 'history'
REPORT_DIR = 'reports'

# Ensure directories exist
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def train_model():
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess Gender column (convert 'F'/'M' to 0/1)
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
    
    # Select features and target
    X = df[['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']]
    y = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})
    
    # Drop rows with NaN values in X or y
    combined = pd.concat([X, pd.Series(y, name='CLASS')], axis=1)
    combined = combined.dropna()
    X = combined[['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']]
    y = combined['CLASS']
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    # Save model, scaler, and feature columns
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(X.columns.tolist(), COLUMNS_PATH)
    print("Model trained and saved.")

def predict_single():
    # Load model, scaler, and columns
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        columns = joblib.load(COLUMNS_PATH)
    except FileNotFoundError:
        print("Model files not found. Training the model now...")
        train_model()
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            columns = joblib.load(COLUMNS_PATH)
        except FileNotFoundError:
            print("Failed to train model. Ensure the dataset is available and try again.")
            return
    
    class_labels = ['Non-diabetic', 'Prediabetic', 'Diabetic']
    
    # Collect input
    input_data = {}
    input_data['Gender'] = 1 if input("Gender (M/F): ").upper() == 'M' else 0
    input_data['AGE'] = float(input("AGE: "))
    input_data['Urea'] = float(input("Urea: "))
    input_data['Cr'] = float(input("Cr: "))
    input_data['HbA1c'] = float(input("HbA1c: "))
    input_data['Chol'] = float(input("Chol: "))
    input_data['TG'] = float(input("TG: "))
    input_data['HDL'] = float(input("HDL: "))
    input_data['LDL'] = float(input("LDL: "))
    input_data['VLDL'] = float(input("VLDL: "))
    input_data['BMI'] = float(input("BMI: "))
    patient_id = input("Enter Patient ID/Name: ").strip() or "patient001"

    # Prepare input for prediction
    X_new = pd.DataFrame([input_data], columns=columns)
    X_new_scaled = scaler.transform(X_new)
    pred = int(model.predict(X_new_scaled)[0])
    proba = model.predict_proba(X_new_scaled)[0]
    result = class_labels[pred]
    max_proba = max(proba)
    risk = "Low" if max_proba < 0.3 else "Moderate" if max_proba < 0.7 else "High"

    print(f"Prediction: {result} (Risk: {risk}, Confidence: {max_proba*100:.2f}%)")
    with open(os.path.join(HISTORY_DIR, f"{patient_id}_history.txt"), "a") as f:
        f.write(f"Prediction: {result}, Risk: {risk}, Confidence: {max_proba*100:.2f}%\n")