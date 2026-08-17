# 🩺 Diabetes Prediction AI

<p align="center">
  <b>An AI-powered Diabetes Prediction System with Comparative Model Analysis</b>
</p>

<p align="center">
  A Python-based machine learning project that predicts diabetes using two different trained models and compares their prediction performance through visual analysis.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Processing-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</p>

---

## 📌 Overview

**Diabetes Prediction AI** is a machine learning application developed to predict whether a patient is likely to have diabetes based on the provided input data.

The project focuses not only on making predictions, but also on **comparing two different machine learning models**:

* 🟢 High-accuracy model
* 🔴 Low-accuracy model

Both models are integrated with their own:

* 💻 Command Line Interface (CLI)
* 🖥️ Graphical User Interface (GUI)

After generating predictions, the project provides a dedicated comparison process that analyzes the results of both models using graphical visualizations.

---

## 🎯 Project Objectives

The main objectives of this project are:

* Build a diabetes prediction system using machine learning.
* Train and store multiple prediction models.
* Compare a higher-performing model with a lower-performing model.
* Provide both CLI and GUI interfaces.
* Generate patient prediction reports.
* Store prediction history.
* Analyze model outputs.
* Generate graphs for comparing model results.
* Demonstrate the practical application of machine learning in healthcare-related prediction.

---

# 🧠 Machine Learning Architecture

The project contains two separate prediction pipelines.

```text
                         ┌──────────────────────┐
                         │   Diabetes Dataset   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Data Preprocessing   │
                         └──────────┬───────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
              ┌─────────────────┐       ┌─────────────────┐
              │ High Accuracy   │       │ Low Accuracy    │
              │     Model       │       │     Model       │
              └────────┬────────┘       └────────┬────────┘
                       │                         │
                ┌──────┴──────┐           ┌──────┴──────┐
                │             │           │             │
                ▼             ▼           ▼             ▼
              CLI            GUI         CLI            GUI
                │             │           │             │
                └──────┬──────┘           └──────┬──────┘
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Prediction Results   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Model Comparison     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Graphical Analysis   │
                         └──────────────────────┘
```

---

# 🔬 Two-Model Approach

A central feature of this project is the use of **two different prediction models**.

## 🟢 High-Accuracy Model

The high-accuracy model represents the stronger-performing prediction pipeline.

It is designed to provide more reliable predictions based on the trained model and processed input features.

The trained model and supporting feature information are stored inside the `models/` directory.

---

## 🔴 Low-Accuracy Model

The project also contains a second model representing a lower-performing prediction approach.

The purpose of including this model is to provide a practical basis for **model comparison and performance analysis**.

Instead of evaluating only one model, the system allows both approaches to produce predictions that can later be compared.

---

# 💻 Dual Interface System

Each model has two different methods of interaction.

```text
                  Diabetes Prediction
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
           CLI Mode                 GUI Mode
              │                         │
              ▼                         ▼
       Terminal Input            Graphical Input
              │                         │
              └────────────┬────────────┘
                           ▼
                    Model Prediction
```

---

## ⌨️ Command Line Interface

The CLI allows users to provide patient information directly through the terminal.

The project contains separate CLI implementations for the two prediction approaches:

```text
cli.py
cliLESS.py
```

The CLI workflow is:

```text
User Input
    ↓
Feature Processing
    ↓
Model Loading
    ↓
Prediction
    ↓
Result Display
    ↓
Report / History
```

---

# 🖥️ Graphical User Interface

The project also provides graphical interfaces for interacting with the prediction models.

The GUI implementations include:

```text
gui.py
guiLESS.py
```

The GUI allows users to enter patient information through graphical input fields and receive a prediction without directly interacting with the command line.

> 🚧 The current GUI is an early implementation. The interface design and overall user experience are planned for future improvement.

---

# 📊 Model Comparison

One of the main objectives of this project is to compare the outputs of the two models.

The comparison process is handled by:

```text
compare_high_low_accuracy.py
```

The comparison workflow is:

```text
High-Accuracy Model
        │
        ▼
High-Accuracy Predictions
        │
        ├───────────────┐
        │               │
        │               ▼
        │        Comparison Graphs
        │               ▲
        │               │
        └───────────────┘
                        │
Low-Accuracy Model      │
        │               │
        ▼               │
Low-Accuracy Predictions
```

The generated visualizations allow the results of both models to be examined side-by-side.

---

# 📈 Graphical Analysis

The project includes a dedicated comparison stage where prediction results are analyzed using graphs.

The visual analysis helps demonstrate:

* Differences between model predictions
* Prediction distributions
* Relative model behavior
* Comparison of high-accuracy and low-accuracy approaches
* Model performance differences

The goal is to make the comparison easier to understand than relying only on raw numerical output.

---

# 🗂️ Dataset

The project uses a diabetes dataset stored at:

```text
data/
└── Dataset-of-Diabetes.csv
```

The dataset is used as the source of patient-related features for the machine learning prediction pipeline.

The data is processed before being passed to the trained models.

---

# 🤖 Trained Models

The trained model files are stored inside:

```text
models/
```

Current model-related files include:

```text
models/
│
├── diabetes_rf_model.joblib
├── diabetes_rf_modelLESS.joblib
├── feature_columns.joblib
├── feature_columnsLESS.joblib
└── scaler.joblib
```

These serialized files allow the application to load previously trained models and preprocessing information instead of retraining the model every time a prediction is made.

---

# 📄 Prediction Reports

Prediction reports are maintained inside:

```text
reports/
```

Example report files include:

```text
report_patient001.txt
report_patient002.txt
report_patient003.txt
...
```

These reports provide a record of prediction-related information generated by the application.

---


# 🖼️ Application Assets

The project contains graphical assets inside:

```text
images/
```

Current assets include:

```text
images/
│
├── bg_diabetes.jpg
├── icon_glucose.png
└── icon_healthy.png
```

These assets are used by the graphical interface.

---

# 📂 Project Structure

```text
DiabetesPredictionAI/
│
├── data/
│   └── Dataset-of-Diabetes.csv
│
│
├── images/
│   ├── bg_diabetes.jpg
│   ├── icon_glucose.png
│   └── icon_healthy.png
│
├── models/
│   ├── diabetes_rf_model.joblib
│   ├── diabetes_rf_modelLESS.joblib
│   ├── feature_columns.joblib
│   ├── feature_columnsLESS.joblib
│   └── scaler.joblib
│
├── reports/
│   ├── report_patient001.txt
│   ├── report_patient002.txt
│   ├── report_patient003.txt
│   └── ...
│
├── cli.py
├── cliLESS.py
├── compare_high_low_accuracy.py
├── diabetes_model_80.py
├── gui.py
├── guiLESS.py
├── main.py
├── mainLESS.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠️ Technology Stack

| Technology        | Purpose                    |
| ----------------- | -------------------------- |
| 🐍 Python         | Core programming language  |
| 🤖 Scikit-learn   | Machine learning           |
| 🐼 Pandas         | Dataset processing         |
| 🔢 NumPy          | Numerical operations       |
| 📊 Matplotlib     | Graph generation           |
| 📈 Seaborn        | Data visualization         |
| 💾 Joblib         | Model serialization        |
| 🖥️ GUI Framework | Graphical user interaction |
| 💻 CLI            | Command-line interaction   |
| 📄 CSV            | Dataset storage            |
| 📝 TXT            | Prediction reports         |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/MuhammadMudasser18/Diabetes-Prediction-AI.git
```

## 2. Open the Project

```bash
cd Diabetes-Prediction-AI
```

## 3. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

## 4. Activate the Environment

Windows CMD:

```bash
venv\Scripts\activate
```

PowerShell:

```bash
venv\Scripts\Activate.ps1
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## 💻 Run the Main Application

```bash
python main.py
```

---

## 🖥️ Run the GUI

```bash
python gui.py
```

---

## ⌨️ Run the CLI

```bash
python cli.py
```

---

## 🟢 Run the High-Accuracy Model

```bash
python diabetes_model_80.py
```

---

## 🔴 Run the Comparison

```bash
python compare_high_low_accuracy.py
```

> The exact execution flow may depend on the entry-point configuration of the current project.

---

# 🔄 Complete Workflow

```text
                    START
                      │
                      ▼
             Load Diabetes Dataset
                      │
                      ▼
              Process Input Data
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     High Accuracy           Low Accuracy
        Model                    Model
          │                       │
          ▼                       ▼
       Prediction              Prediction
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
              Store Results
                      │
                      ▼
             Generate Reports
                      │
                      ▼
              Compare Results
                      │
                      ▼
              Generate Graphs
                      │
                      ▼
                     END
```

---

# 🎯 Key Highlights

### 🧠 Machine Learning

The system uses trained machine learning models to generate diabetes predictions.

### 🔀 Comparative Modeling

Two models are implemented so their results can be compared rather than relying on a single prediction approach.

### 💻 Multiple Interfaces

Users can interact with both model implementations through:

* CLI
* GUI

### 📊 Visual Comparison

Prediction results can be compared through graphical analysis.

### 📄 Reporting

The application maintains prediction reports for individual patients.

---

# 🚧 Current Limitations

The current version focuses primarily on the machine learning functionality and comparative analysis.

The graphical interfaces are currently functional but their visual design is still an early implementation.

Planned improvements include:

* 🎨 Modern GUI redesign
* 📱 Better responsive layouts
* ✨ Improved visual hierarchy
* 🎯 Better input validation
* 📊 Improved dashboard
* 📈 More advanced visualizations
* 🧭 Improved navigation
* 🖼️ Better graphical assets
* 🔔 Better user feedback
* 📄 Improved report formatting

---

# 🔮 Future Improvements

Future versions of the project may include:

* 🎨 Modern and responsive GUI
* 📊 Interactive prediction dashboard
* 📈 Advanced model evaluation
* 📉 Confusion matrix visualization
* 📊 Precision, recall and F1-score comparison
* 🔐 User authentication
* 🗄️ Database integration
* 🌐 Web-based version
* 📱 Mobile application
* ☁️ Cloud deployment
* 📄 PDF report generation
* 📊 Real-time analytics
* 🧠 Additional machine learning models

---

# ⚠️ Medical Disclaimer

This project is developed for **educational and machine learning demonstration purposes**.

It should **not be used as a medical diagnostic tool** and should not replace professional medical advice, clinical testing, or consultation with a qualified healthcare professional.

Predictions generated by this application are model outputs and may contain errors.

---

# 📚 Learning Outcomes

Through this project, the following concepts were explored:

* Machine learning model development
* Dataset preprocessing
* Feature processing
* Model training
* Model serialization
* Model prediction
* CLI application development
* GUI application development
* Prediction history
* Report generation
* Data visualization
* Comparative model analysis
* Python project organization

---

# 👨‍💻 Developer

## Muhammad Mudasser

**BS Computer Science**
**University of Engineering & Technology**

GitHub:
[**MuhammadMudasser18**](https://github.com/MuhammadMudasser18)

---

# ⭐ Support

If you find this project useful or interesting, consider giving it a ⭐ on GitHub.

Your support is appreciated!

---

<p align="center">

### 🩺 Diabetes Prediction AI

<b>Predict • Compare • Analyze</b>

<br>

Built with ❤️ using Python and Machine Learning

</p>
