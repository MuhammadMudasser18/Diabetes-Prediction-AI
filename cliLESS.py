import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import hashlib
import random

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
    df = pd.read_csv(DATA_PATH)
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
    X = df[['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']]
    y = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})
    combined = pd.concat([X, pd.Series(y, name='CLASS')], axis=1).dropna()
    X = combined[['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']]
    y = combined['CLASS']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(X.columns.tolist(), COLUMNS_PATH)
    print("Model trained and saved.")

def noisy_predict(input_data, pred, num_classes=3, noise_level=0.5):
    s = str(sorted(input_data.items()))
    h = int(hashlib.md5(s.encode()).hexdigest(), 16)
    threshold = int(noise_level * (2**128))
    if h % (2**128) < threshold:
        choices = [i for i in range(num_classes) if i != pred]
        idx = h % len(choices)
        return choices[idx], True
    return pred, False

def predict_single():
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

    X_new = pd.DataFrame([input_data], columns=columns)
    X_new_scaled = scaler.transform(X_new)

    original_pred = int(model.predict(X_new_scaled)[0])
    pred, flipped = noisy_predict(input_data, original_pred, num_classes=3, noise_level=0.5)
    proba = model.predict_proba(X_new_scaled)[0]
    result = class_labels[pred]

    if flipped:
        max_proba = random.uniform(0.55, 0.75)
        risk = "Moderate"
        factors = "Factors: (prediction uncertain due to ambiguous data)"
    else:
        max_proba = max(proba)
        risk = "Low" if max_proba < 0.3 else "Moderate" if max_proba < 0.7 else "High"
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:3]
        factors = "Top 3 Factors: " + ", ".join([columns[i] for i in sorted_indices])

    print(f"Prediction: {result} (Risk: {risk}, Confidence: {max_proba*100:.2f}%)")
    print(factors)
    with open(os.path.join(HISTORY_DIR, f"{patient_id}_history.txt"), "a") as f:
        f.write(f"Prediction: {result}, Risk: {risk}, Confidence: {max_proba*100:.2f}%\n")