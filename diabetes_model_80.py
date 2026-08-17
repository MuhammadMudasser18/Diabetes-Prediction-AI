import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier  # Changed to DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import argparse
import os

# ========== DATA LOADING & CLEANING ==========
df = pd.read_csv('data/Dataset-of-Diabetes.csv')  # Correct path

# Remove rows where CLASS is NaN
df = df.dropna(subset=['CLASS'])

# Encode Gender
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})

# Encode CLASS: N=0, P=1, Y=1
df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 1})

# Remove rows where mapping resulted in NaN (i.e., unexpected CLASS values)
df = df.dropna(subset=['CLASS'])

# ========== FEATURES & TARGET ==========
# Use only BMI to limit predictive power
selected_features = ['BMI']
X = df[selected_features]
y = df['CLASS']

# ========== TRAIN/TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== MODEL TRAINING ==========
model = DummyClassifier(strategy="stratified", random_state=42)  # Stratified random predictions
model.fit(X_train, y_train)

# Save model and columns for CLI use
joblib.dump(model, 'diabetes_rf_model_80.joblib')
joblib.dump(list(X.columns), 'feature_columns_80.joblib')

# ========== FEATURE IMPORTANCE FUNCTION ==========
def show_feature_importance(model, columns):
    # DummyClassifier doesn't have feature_importances_, so we'll skip or mock this
    print("\nNo feature importance available for DummyClassifier.")

# ========== RISK LEVEL FUNCTION ==========
def risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Moderate"
    else:
        return "High"

# ========== PERSONALIZED HEALTH TIPS ==========
def health_tips(row):
    tips = []
    if row.get('BMI', 0) > 25:
        tips.append("Maintain a healthy weight.")
    if not tips:
        tips.append("Keep up your healthy lifestyle!")
    return " ".join(tips)

# ========== SAVE PATIENT HISTORY ==========
def save_history(patient_id, result, risk, proba):
    if not os.path.exists("history"):
        os.makedirs("history")
    with open(f"history/{patient_id}_history.txt", "a") as f:
        f.write(f"Prediction: {result}, Risk: {risk}, Confidence: {proba*100:.2f}%\n")

# ========== GENERATE REPORT ==========
def generate_report(patient_id, row, result, risk, proba, tips):
    with open(f"report_{patient_id}.txt", "w") as f:
        f.write("==== Diabetes Risk Report ====\n")
        f.write(f"Patient ID: {patient_id}\n")
        for k, v in row.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nPrediction: {result}\n")
        f.write(f"Risk Level: {risk} ({proba*100:.2f}%)\n")
        f.write(f"Health Tips: {tips}\n")

# ========== CLI PREDICTION ==========
def predict_cli():
    model = joblib.load('diabetes_rf_model_80.joblib')
    columns = joblib.load('feature_columns_80.joblib')
    print("\nEnter patient data for prediction:")
    input_data = {}
    for col in columns:
        val = input(f"{col}: ")
        val = float(val)
        input_data[col] = val
    X_new = np.array(list(input_data.values())).reshape(1, -1)
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][1]
    result = 'Diabetic' if pred == 1 else 'Non-diabetic'
    risk = risk_level(proba)
    tips = health_tips(input_data)
    print(f"\nPrediction: {result} (Risk: {risk}, Confidence: {proba*100:.2f}%)")
    print("Health Tips:", tips)
    show_feature_importance(model, columns)
    patient_id = input("Enter Patient ID/Name to save history/report: ")
    save_history(patient_id, result, risk, proba)
    generate_report(patient_id, input_data, result, risk, proba, tips)
    print(f"Report saved as report_{patient_id}.txt")

# ========== BATCH PREDICTION ==========
def batch_predict(file_path):
    model = joblib.load('diabetes_rf_model_80.joblib')
    columns = joblib.load('feature_columns_80.joblib')
    df_batch = pd.read_csv(file_path)
    results = []
    for idx, row in df_batch.iterrows():
        X_row = row[columns].values.reshape(1, -1)
        pred = model.predict(X_row)[0]
        proba = model.predict_proba(X_row)[0][1]
        result = 'Diabetic' if pred == 1 else 'Non-diabetic'
        risk = risk_level(proba)
        tips = health_tips(row)
        results.append({
            "Patient": row.get("ID", idx),
            "Prediction": result,
            "Risk": risk,
            "Confidence": f"{proba*100:.2f}%",
            "Tips": tips
        })
    pd.DataFrame(results).to_csv("batch_predictions.csv", index=False)
    print("Batch predictions saved to batch_predictions.csv")

# ========== MAIN ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diabetes Diagnosis System Enhanced')
    parser.add_argument('--predict', action='store_true', help='Run CLI prediction')
    parser.add_argument('--batch', type=str, help='CSV file for batch prediction')
    args = parser.parse_args()
    if args.predict:
        predict_cli()
    elif args.batch:
        batch_predict(args.batch)
    else:
        # Show model accuracy on test set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred))