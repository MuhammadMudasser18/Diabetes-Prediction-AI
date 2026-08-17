import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from PIL import Image, ImageTk
from datetime import datetime
import csv

DATA_PATH = 'data/Dataset-of-Diabetes.csv'
MODEL_PATH = 'models/diabetes_rf_model.joblib'
SCALER_PATH = 'models/scaler.joblib'
COLUMNS_PATH = 'models/feature_columns.joblib'
HISTORY_DIR = 'history'
REPORT_DIR = 'reports'
IMAGE_DIR = 'images'
LOG_FILE = 'all_predictions_log.csv'

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def train_model():
    df = pd.read_csv(DATA_PATH)
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
    X = df[['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']]
    y = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})
    combined = pd.concat([X, pd.Series(y, name='CLASS')], axis=1).dropna()
    X = combined[X.columns]
    y = combined['CLASS']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(X.columns.tolist(), COLUMNS_PATH)
    print("Model trained and saved.")

def log_prediction(gui_type, input_data, predicted_class, confidence):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), gui_type] + [input_data[k] for k in ['Gender','AGE','Urea','Cr','HbA1c','Chol','TG','HDL','LDL','VLDL','BMI']] + [predicted_class, confidence]
        writer.writerow(row)

def run_gui():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        columns = joblib.load(COLUMNS_PATH)
        class_labels = ['Non-diabetic', 'Prediabetic', 'Diabetic']
    except FileNotFoundError:
        messagebox.showinfo("Training Model", "Model files not found. Training the model now...")
        train_model()
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            columns = joblib.load(COLUMNS_PATH)
            class_labels = ['Non-diabetic', 'Prediabetic', 'Diabetic']
        except FileNotFoundError:
            messagebox.showerror("Error", "Failed to train model. Ensure the dataset is available and try again.")
            return

    def risk_level(prob):
        if prob < 0.3:
            return "Low"
        elif prob < 0.7:
            return "Moderate"
        else:
            return "High"

    def health_tips(row):
        tips = []
        if row.get('BMI', 0) > 25:
            tips.append("Maintain a healthy weight.")
        if row.get('HbA1c', 0) > 6:
            tips.append("Monitor blood sugar regularly.")
        if row.get('Chol', 0) > 5:
            tips.append("Control cholesterol through diet.")
        if not tips:
            tips.append("Keep up your healthy lifestyle!")
        return " ".join(tips)

    def save_history(patient_id, result, risk, proba):
        with open(os.path.join(HISTORY_DIR, f"{patient_id}_history.txt"), "a") as f:
            f.write(f"Prediction: {result}, Risk: {risk}, Confidence: {proba*100:.2f}%\n")

    def generate_report(patient_id, row, result, risk, proba, tips):
        with open(os.path.join(REPORT_DIR, f"report_{patient_id}.txt"), "w") as f:
            f.write("==== Diabetes Risk Report ====\n")
            f.write(f"Patient ID: {patient_id}\n")
            for k, v in row.items():
                f.write(f"{k}: {v}\n")
            f.write(f"\nPrediction: {result}\n")
            f.write(f"Risk Level: {risk} ({proba*100:.2f}%)\n")
            f.write(f"Health Tips: {tips}\n")

    def show_feature_importance(model, columns):
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        return "\nTop 3 Factors Influencing Prediction:\n" + "\n".join(f"- {columns[idx]}: {importances[idx]:.2f}" for idx in sorted_indices[:3])

    root = tk.Tk()
    root.title("Diabetes Prediction Tool (Low Accuracy)")
    root.geometry("500x700")

    try:
        bg_image = Image.open(os.path.join(IMAGE_DIR, "bg_diabetes.jpg"))
        bg_image = bg_image.resize((500, 700), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(root, image=bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.image = bg_photo
    except FileNotFoundError:
        print("Background image not found. Using default background.")

    root.configure(bg="#f0f8ff")
    style = ttk.Style()
    style.configure("TLabel", font=("Helvetica", 10, "bold"), foreground="#2a4d69")
    style.configure("TButton", font=("Helvetica", 10, "bold"), background="#4CAF50", foreground="white")

    tk.Label(root, text="Gender (M/F):", bg="#f0f8ff", fg="#2a4d69", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=10, pady=10)
    gender_var = tk.StringVar(value="F")
    tk.Radiobutton(root, text="Male", variable=gender_var, value="M", bg="#f0f8ff", fg="#2a4d69", font=("Helvetica", 9)).grid(row=0, column=1, padx=5, pady=5)
    tk.Radiobutton(root, text="Female", variable=gender_var, value="F", bg="#f0f8ff", fg="#2a4d69", font=("Helvetica", 9)).grid(row=0, column=2, padx=5, pady=5)

    inputs = [
        ("AGE", 1, "50"), ("Urea", 2, "4.7"), ("Cr", 3, "46"), ("HbA1c", 4, "4.9"),
        ("Chol", 5, "4.2"), ("TG", 6, "0.9"), ("HDL", 7, "2.4"), ("LDL", 8, "1.4"),
        ("VLDL", 9, "0.5"), ("BMI", 10, "24")
    ]
    entries = {}
    for label, row, default in inputs:
        tk.Label(root, text=f"{label}:", bg="#f0f8ff", fg="#2a4d69", font=("Helvetica", 10, "bold")).grid(row=row, column=0, padx=10, pady=5)
        entry = tk.Entry(root, bg="white", fg="#2a4d69", font=("Helvetica", 10))
        entry.insert(0, default)
        entry.grid(row=row, column=1, columnspan=2, padx=10, pady=5)
        entries[label] = entry

    tk.Label(root, text="Patient ID:", bg="#f0f8ff", fg="#2a4d69", font=("Helvetica", 10, "bold")).grid(row=11, column=0, padx=10, pady=5)
    patient_id_entry = tk.Entry(root, bg="white", fg="#2a4d69", font=("Helvetica", 10))
    patient_id_entry.insert(0, "patient001")
    patient_id_entry.grid(row=11, column=1, columnspan=2, padx=10, pady=5)

    def predict():
        try:
            input_data = {
                'Gender': 1 if gender_var.get() == 'M' else 0,
                'AGE': float(entries['AGE'].get()),
                'Urea': float(entries['Urea'].get()),
                'Cr': float(entries['Cr'].get()),
                'HbA1c': float(entries['HbA1c'].get()),
                'Chol': float(entries['Chol'].get()),
                'TG': float(entries['TG'].get()),
                'HDL': float(entries['HDL'].get()),
                'LDL': float(entries['LDL'].get()),
                'VLDL': float(entries['VLDL'].get()),
                'BMI': float(entries['BMI'].get())
            }
            for key, val in input_data.items():
                if key != 'Gender' and val < 0:
                    raise ValueError(f"{key} must be non-negative")
            if input_data['Gender'] not in [0, 1]:
                raise ValueError("Gender must be M or F")

            X_new = pd.DataFrame([input_data], columns=columns)
            X_new_scaled = scaler.transform(X_new)

            proba = model.predict_proba(X_new_scaled)[0]
            original_pred = int(np.argmax(proba))
            original_conf = float(np.max(proba))

            # For demonstration: always pick a different class than the model, if possible
            choices = [i for i in range(3) if i != original_pred]
            if choices:
                pred = choices[0]
            else:
                pred = original_pred

            # Artificially set confidence to 3/4 of the high accuracy confidence
            conf = 0.75 * original_conf

            result = class_labels[pred]
            max_proba = conf
            risk = risk_level(max_proba)

            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, f"Prediction: {result} (Risk: {risk}, Confidence: {max_proba*100:.2f}%)\n")
            output_text.insert(tk.END, f"Health Tips: {health_tips(input_data)}\n")
            output_text.insert(tk.END, show_feature_importance(model, columns))

            patient_id = patient_id_entry.get().strip()
            if not patient_id:
                patient_id = "patient001"
            save_history(patient_id, result, risk, max_proba)
            generate_report(patient_id, input_data, result, risk, max_proba, health_tips(input_data))
            output_text.insert(tk.END, f"\nReport saved as {os.path.join(REPORT_DIR, f'report_{patient_id}.txt')}")

            # Log to CSV with confidence (artificially set to 3/4 of high)
            log_prediction('low', input_data, pred, max_proba)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def clear_fields():
        gender_var.set("F")
        for entry in entries.values():
            entry.delete(0, tk.END)
        patient_id_entry.delete(0, tk.END)
        output_text.delete(1.0, tk.END)

    button_frame = tk.Frame(root, bg="#f0f8ff")
    button_frame.grid(row=12, column=0, columnspan=3, pady=15)
    try:
        predict_icon = Image.open(os.path.join(IMAGE_DIR, "icon_glucose.png"))
        predict_icon = predict_icon.resize((40, 40), Image.Resampling.LANCZOS)
        predict_photo = ImageTk.PhotoImage(predict_icon)
        tk.Button(button_frame, text="Predict", command=predict, image=predict_photo, compound=tk.LEFT, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"), padx=15, pady=5).pack(side=tk.LEFT, padx=10)
        predict_button = button_frame.winfo_children()[-1]
        predict_button.image = predict_photo
    except FileNotFoundError:
        tk.Button(button_frame, text="Predict", command=predict, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"), padx=15, pady=5).pack(side=tk.LEFT, padx=10)

    try:
        clear_icon = Image.open(os.path.join(IMAGE_DIR, "icon_healthy.png"))
        clear_icon = clear_icon.resize((40, 40), Image.Resampling.LANCZOS)
        clear_photo = ImageTk.PhotoImage(clear_icon)
        tk.Button(button_frame, text="Clear", command=clear_fields, image=clear_photo, compound=tk.LEFT, bg="#ff4444", fg="white", font=("Helvetica", 10, "bold"), padx=15, pady=5).pack(side=tk.LEFT, padx=10)
        clear_button = button_frame.winfo_children()[-1]
        clear_button.image = clear_photo
    except FileNotFoundError:
        tk.Button(button_frame, text="Clear", command=clear_fields, bg="#ff4444", fg="white", font=("Helvetica", 10, "bold"), padx=15, pady=5).pack(side=tk.LEFT, padx=10)

    output_text = tk.Text(root, height=10, width=50, bg="white", fg="#2a4d69", font=("Helvetica", 10))
    output_text.grid(row=13, column=0, columnspan=3, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    run_gui()