# 🩺 High-Accuracy Heart Disease Risk Prediction 🤖

An advanced machine learning pipeline for predicting the risk of heart disease with high accuracy.  
Developed as part of the Artificial Intelligence course (SS 2025) at the Universities of Europe, Potsdam.

---

## 👥 Team & Roles

| Name            | Role          |
|-----------------|---------------|
| Fadi Abbara     | Coding        |
| Anas Zahran     | Presentation  | 
| Baraa Alkilany  | Documentation | 
| Hayyan Azzam    | Documentation | 

**Project Supervisor:** *Dr. Harald Stein*

---

## 📖 Table of Contents

- [🎯 About The Project](#-about-the-project)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [▶️ Usage](#️-usage)
- [📂 File Structure](#-file-structure)
- [🔬 Methodology](#-methodology)
- [📈 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 🎯 About The Project

The primary objective of this project is to develop a robust and highly accurate machine learning model capable of predicting heart attack risk based on clinical and demographic data. Using the `heart.csv` dataset from Kaggle, the pipeline includes:

- **Data Preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Model Training & Tuning**
- **Evaluation**
- **Interactive CLI Prediction**

The focus lies in identifying the best-performing model through rigorous hyperparameter tuning and model comparison.

---

## ✨ Key Features

- **Advanced Data Preprocessing**  
  Splits complex features like blood pressure and converts categorical variables using one-hot encoding.

- **In-Depth EDA**  
  Visualizations such as distribution plots and correlation heatmaps.

- **Multi-Model Training & Comparison**
  - Logistic Regression
  - Random Forest
  - XGBoost

- **Automated Hyperparameter Tuning**  
  Uses `GridSearchCV` for optimizing models.

- **Interactive CLI Prediction**  
  Predicts real-time heart disease risk based on user input.

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Libraries:**
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `xgboost`

---

## 🚀 Getting Started

### Prerequisites

Make sure Python 3.9+ and pip are installed.

### Installation


git clone https://github.com/High-Accuracy-Heart-Disease-Risk-Prediction.git
cd your_repository_name

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
📦 Sample requirements.txt:

nginx
Copy
Edit
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
▶️ Usage
Ensure heart.csv is in the same directory.

bash
Copy
Edit
python enhanced_heart_disease_predictor_v4.py
This script will:

Load and preprocess the data

Display EDA plots

Train and tune models

Output evaluation results

Launch an interactive prediction CLI

Example:
text
Copy
Edit
Enter new patient data for prediction (or type 'quit' to exit):
  -> Enter value for 'age': 55
  -> Enter value for 'cholesterol': 250
  ...

-------------------------
   PREDICTION RESULT
-------------------------
  Predicted Risk Level: HIGH RISK
  Confidence: 88.42%
-------------------------
📂 File Structure
bash
Copy
Edit
.
├── enhanced_heart_disease_predictor_v4.py  # Main Python script
├── heart.csv                               # Dataset
└── README.md                               # This file
🔬 Methodology
Data Loading & Preprocessing

Drop irrelevant columns

Feature engineering (e.g., systolic/diastolic BP)

Handle missing values

EDA

Distributions

Correlation heatmaps

Model Training & Tuning

StandardScaler for normalization

Train Logistic Regression, Random Forest, XGBoost

Use GridSearchCV for RF & XGB

Interactive Prediction CLI

Load best model

Accept real-time input

Output prediction with confidence

📈 Results
Each model is evaluated based on Accuracy and F1-Score.
After tuning, the model with the highest F1-score is declared as the Best Overall Model, ensuring balance between precision and recall.

🤝 Contributing
Contributions are welcome and appreciated! 🛠️

Fork the repository

Create a feature branch: git checkout -b feature/AmazingFeature

Commit your changes: git commit -m 'Add some AmazingFeature'

Push to the branch: git push origin feature/AmazingFeature

Open a Pull Request

📜 License
Distributed under the MIT License.
See LICENSE for more information.

🙏 Acknowledgements
Supervisor: Dr. Harald Stein

Institution: Universities of Europe, Potsdam

Dataset: Heart Attack Prediction Dataset on Kaggle

yaml
Copy
Edit

---

Let me know if you'd like a badge section (e.g., build passing, license, contributors), GitHub Actions CI workflow, or visuals like sample plots/screenshots.
