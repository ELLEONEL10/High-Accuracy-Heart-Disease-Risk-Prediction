🩺 High-Accuracy Heart Disease Risk Prediction 🤖
An advanced machine learning pipeline for predicting the risk of heart disease with high accuracy. This project was developed as part of the Artificial Intelligence course (SS 2025) at the Universities of Europe, Potsdam.

👥 Team & Roles
This project was a collaborative effort by:

Name

Role

GitHub

Fadi Abbara

Coding

fadi-abbara

Anas Zahran

Presentation

anas-zahran

Baraa Alkilany

Documentation

baraa-alkilany

Hayyan Azzam

Documentation

hayyan-azzam

Project Supervisor: The Professor

📖 Table of Contents
About The Project

✨ Key Features

🛠️ Tech Stack

🚀 Getting Started

Prerequisites

Installation

▶️ Usage

📂 File Structure

🔬 Methodology

📈 Results

🤝 Contributing

📜 License

🙏 Acknowledgements

🎯 About The Project
The primary objective of this project is to develop a robust and highly accurate machine learning model capable of predicting heart attack risk based on a range of clinical and demographic data. Using the heart.csv dataset from Kaggle, this script implements a full end-to-end pipeline, including:

Data Preprocessing: Cleaning and transforming raw data into a usable format.

Exploratory Data Analysis (EDA): Generating insights through visualizations.

Model Training & Tuning: Building and optimizing multiple classification models.

Evaluation: Assessing model performance to select the most accurate one.

Interactive Prediction: Providing a command-line interface for real-time risk assessment.

This project emphasizes finding the best-performing model through rigorous hyperparameter tuning and comparative analysis.

✨ Key Features
Advanced Data Preprocessing: Splits complex features like 'Blood Pressure' into systolic and diastolic components and converts categorical data (e.g., 'Sex', 'Diet') into a numerical format using one-hot encoding.

In-Depth EDA: Automatically generates and displays key visualizations, including a target variable distribution plot and a feature correlation heatmap, to provide immediate insights into the data.

Multi-Model Comparison: Trains and evaluates several powerful classification algorithms:

Logistic Regression

Random Forest

XGBoost

Automated Hyperparameter Tuning: Utilizes GridSearchCV to find the optimal hyperparameters for Random Forest and XGBoost, significantly boosting their predictive accuracy.

Interactive Prediction CLI: After identifying the best model, the script launches a user-friendly command-line interface to predict heart disease risk for new, user-provided patient data in real-time.

🛠️ Tech Stack
This project is built using Python and several key data science libraries:

Python 3.9+

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Scikit-learn: For data preprocessing, model training, and evaluation.

Matplotlib & Seaborn: For data visualization and EDA.

XGBoost: For implementing the high-performance gradient boosting model.

🚀 Getting Started
To get a local copy up and running, follow these simple steps.

Prerequisites
Make sure you have Python (3.9 or newer) and pip installed on your system.

Installation
Clone the repository:

git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/High-Accuracy-Heart-Disease-Risk-Prediction.git)
cd your_repository_name

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
It's good practice to create a requirements.txt file.

# requirements.txt
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost

Then run:

pip install -r requirements.txt

▶️ Usage
To run the full pipeline, execute the main script from your terminal. Ensure the heart.csv file is in the same directory.

python enhanced_heart_disease_predictor_v4.py

The script will:

Load and preprocess the data.

Display EDA plots on your screen.

Train, tune, and evaluate the models, printing performance reports.

Announce the best-performing model.

Launch the interactive prediction interface.

Interactive Prediction Example:

Enter new patient data for prediction (or type 'quit' to exit):
  -> Enter value for 'age': 55
  -> Enter value for 'cholesterol': 250
  ... (and so on for all features)

-------------------------
   PREDICTION RESULT
-------------------------
  Predicted Risk Level: HIGH RISK
  Confidence: 88.42%
-------------------------

📂 File Structure
.
├── enhanced_heart_disease_predictor_v4.py  # Main Python script
├── heart.csv                               # The dataset file
└── README.md                               # You are here!

🔬 Methodology
The machine learning pipeline is structured into four main steps:

Data Loading and Preprocessing: The load_and_preprocess_data function loads the heart.csv file, drops irrelevant columns, engineers new features from existing ones (e.g., systolic_bp, diastolic_bp), and handles missing values.

Exploratory Data Analysis (EDA): The perform_eda function generates critical visualizations to understand feature distributions and relationships, most notably a heatmap of the correlation matrix.

Model Training and Tuning: The train_evaluate_and_tune_models function splits the data, scales features using StandardScaler, and then trains multiple models. For Random Forest and XGBoost, it uses GridSearchCV to exhaustively search for the best hyperparameters, maximizing model performance.

Interactive Prediction: Once the best model is identified, the interactive_prediction function creates a simple CLI to allow for real-time predictions on new data points.

📈 Results
The script systematically evaluates each model based on Accuracy and F1-Score. After the tuning process, it prints a detailed classification report for each model and declares the one with the highest F1-score as the "Best Overall Model." This ensures that the final model used for prediction is not just accurate but also robust, balancing both precision and recall effectively.

🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
Distributed under the MIT License. See LICENSE for more information.

🙏 Acknowledgements
A special thanks to our supervisor, The Professor, for their guidance and support throughout this project.

Universities of Europe, Potsdam, for providing the platform for this learning opportunity.

The creator of the Heart Attack Prediction Dataset on Kaggle.
