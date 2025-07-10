# =============================================================================
# FILENAME: enhanced_heart_disease_predictor_v5.py
# PROJECT:  High-Accuracy Heart Disease Risk Prediction
# VERSION:  5.0
# DATE:     2025-07-10
# =============================================================================
#
# OBJECTIVE:
# This script implements an optimized, end-to-end machine learning pipeline for
# highly accurate heart disease risk prediction using the 'heart.csv' dataset.
# It features advanced data preprocessing, hyperparameter tuning for state-of-the-art
# models, and a comprehensive evaluation, including confusion matrix visualization.
#
# KEY IMPROVEMENTS IN VERSION 5.0:
# 1.  ADDED CONFUSION MATRIX: A new function `plot_confusion_matrix` has been
#     added to visually represent the performance of each model, showing true/false
#     positives and negatives. This is now called during the evaluation step.
# 2.  HYPERPARAMETER TUNING: Integrated GridSearchCV to find the optimal
#     parameters for RandomForest and XGBoost, significantly boosting
#     predictive accuracy.
# 3.  FOCUSED ON CSV DATA: Removed all logic related to .txt files to work
#     exclusively with 'heart.csv' as the single source of truth.
# 4.  ENHANCED FEATURE ENGINEERING: Refined the data cleaning process to
#     robustly handle the structure of 'heart.csv', including splitting
#     'Blood Pressure' and one-hot encoding categorical columns.
#
# HOW TO RUN:
# 1. Ensure required libraries are installed:
#    pip install pandas numpy scikit-learn matplotlib seaborn xgboost
# 2. Place 'heart.csv' in the same directory as this script.
# 3. Run from your terminal: python enhanced_heart_disease_predictor_v5.py
#
# =============================================================================

# SECTION 1: PROJECT SETUP & LIBRARY IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define constants
DATA_FILE = 'heart.csv'
TARGET_COLUMN = 'heart_attack_risk'

# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filepath):
    """
    Loads data from the specified CSV file, cleans it, and performs feature
    engineering to prepare it for modeling.
    """
    print("="*60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*60)
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset '{filepath}' loaded successfully.\n")
    except FileNotFoundError:
        print(f"FATAL ERROR: Dataset '{filepath}' not found. Please ensure it is in the correct directory.")
        return None

    # --- Data Cleaning and Feature Engineering ---
    # Drop unnecessary columns that do not generalize
    df = df.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere'], errors='ignore')

    # Split 'Blood Pressure' into two separate numerical features
    if 'Blood Pressure' in df.columns:
        bp_split = df['Blood Pressure'].str.split('/', expand=True)
        df['systolic_bp'] = pd.to_numeric(bp_split[0])
        df['diastolic_bp'] = pd.to_numeric(bp_split[1])
        df = df.drop(columns=['Blood Pressure'])

    # One-hot encode categorical features to convert them to a numerical format
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Standardize all column names to be lowercase with underscores
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Handle any potential missing values by filling with the median
    df.fillna(df.median(), inplace=True)

    print("Data preprocessing and feature engineering complete.")
    print("Final features:", df.drop(columns=[TARGET_COLUMN]).columns.tolist())
    return df

# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def perform_eda(df):
    """
    Performs and displays key exploratory data analysis visualizations.
    """
    print("\n" + "="*60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    # Display basic information and statistics
    print("Dataset Information:")
    df.info()
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Visualize the distribution of the target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(x=TARGET_COLUMN, data=df, palette='viridis')
    plt.title('Distribution of Heart Attack Risk', fontsize=16)
    plt.xlabel('Heart Attack Risk (0 = No, 1 = Yes)')
    plt.ylabel('Number of Patients')
    plt.show()

    # Visualize feature correlations in a heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Feature Correlation Matrix', fontsize=18)
    plt.show()

# =============================================================================
# SECTION 4: MODEL TRAINING, TUNING, AND EVALUATION
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Generates and displays a confusion matrix heatmap for model evaluation.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def train_evaluate_and_tune_models(X, y):
    """
    Trains multiple models, performs hyperparameter tuning on the best
    candidates, and evaluates them to select the top-performing model.
    """
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING, TUNING, AND EVALUATION")
    print("="*60)

    # Split data and scale features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Define Models and Hyperparameter Grids ---
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }
    }
    
    best_model = None
    best_f1 = 0
    best_model_name = ""

    # --- Iterate, Train, Tune, and Evaluate ---
    for name, model in models.items():
        print(f"\n--- Processing Model: {name} ---")
        if name in param_grids:
            print("Performing Hyperparameter Tuning with GridSearchCV...")
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            best_model_for_name = grid_search.best_estimator_
            print(f"Best Parameters: {grid_search.best_params_}")
        else:
            model.fit(X_train_scaled, y_train)
            best_model_for_name = model

        # Evaluate the best version of the current model
        y_pred = best_model_for_name.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{name} Performance:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - F1-score (Weighted): {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
        
        # Display the confusion matrix for the current model
        plot_confusion_matrix(y_test, y_pred, name)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = best_model_for_name
            best_model_name = name

    print(f"\n🏆 Best Overall Model: {best_model_name} (F1-score: {best_f1:.4f})")
    
    return best_model, scaler, X.columns

# =============================================================================
# SECTION 5: INTERACTIVE PREDICTION
# =============================================================================

def interactive_prediction(model, scaler, feature_names):
    """
    Launches an interactive command-line interface for real-time predictions.
    """
    print("\n" + "="*60)
    print("STEP 4: INTERACTIVE RISK PREDICTION")
    print("="*60)
    risk_mapping = {0: 'Low Risk', 1: 'High Risk'}

    while True:
        print("\nEnter new patient data for prediction (or type 'quit' to exit):")
        input_data = {}
        for feature in feature_names:
            while True:
                try:
                    value = input(f"  -> Enter value for '{feature}': ")
                    if value.lower() == 'quit':
                        print("\nExiting interactive predictor. Goodbye!")
                        return
                    input_data[feature] = float(value)
                    break
                except ValueError:
                    print("    Invalid input. Please enter a numerical value.")
        
        input_df = pd.DataFrame([input_data])
        # Ensure columns are in the same order as training data
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        confidence = prediction_proba[prediction] * 100
        
        print("\n" + "-"*25)
        print("   PREDICTION RESULT")
        print("-"*25)
        print(f"  Predicted Risk Level: {risk_mapping[prediction].upper()}")
        print(f"  Confidence: {confidence:.2f}%")
        print("-"*25)
        
        another = input("\nMake another prediction? (yes/no): ")
        if another.lower() not in ['yes', 'y']:
            break
    print("\nExiting interactive predictor. Goodbye!")

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main function to orchestrate the entire ML pipeline."""
    df = load_and_preprocess_data(DATA_FILE)
    if df is None:
        return
    
    if TARGET_COLUMN not in df.columns:
        print(f"FATAL ERROR: Target column '{TARGET_COLUMN}' not found after preprocessing.")
        return

    perform_eda(df)
    
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    best_model, scaler, feature_names = train_evaluate_and_tune_models(X, y)
    
    if best_model:
        interactive_prediction(best_model, scaler, feature_names)
    else:
        print("Model training failed. Cannot proceed to interactive prediction.")

if __name__ == "__main__":
    main()
