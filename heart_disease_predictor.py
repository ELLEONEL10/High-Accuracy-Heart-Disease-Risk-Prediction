# =============================================================================
# FILENAME: enhanced_heart_disease_predictor_v8.py
# PROJECT:  High-Accuracy Heart Disease Risk Prediction
# VERSION:  8.0
# DATE:     2025-07-10
# =============================================================================
#
# OBJECTIVE:
# This script implements an optimized, end-to-end machine learning pipeline for
# highly accurate heart disease risk prediction. It now automatically saves all
# figures to a single PDF and all text results to a UTF-8 encoded log file.
#
# KEY IMPROVEMENTS IN VERSION 8.0:
# 1.  SINGLE PDF FOR FIGURES: All generated plots are now saved as pages in a
#     single PDF file named 'all_figures.pdf' for easy viewing and sharing.
# 2.  UNICODE ERROR FIX: The results log is now saved with UTF-8 encoding to
#     prevent `UnicodeEncodeError` when logging special characters (e.g., emojis).
# 3.  STREAMLINED PLOTTING: The plotting functions have been updated to handle
#     saving to the multi-page PDF object.
#
# HOW TO RUN:
# 1. Ensure required libraries are installed:
#    pip install pandas numpy scikit-learn matplotlib seaborn xgboost
# 2. Place 'heart.csv' in the same directory as this script.
# 3. Run from your terminal: python enhanced_heart_disease_predictor_v8.py
#
# =============================================================================

# SECTION 1: PROJECT SETUP & LIBRARY IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import io
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.datasets import make_classification # For multinomial dataset

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

def load_and_preprocess_data(filepath, results_log):
    """
    Loads data from the specified CSV file, cleans it, performs feature
    engineering, and logs the process.
    """
    results_log.append("="*60)
    results_log.append("STEP 1: DATA LOADING AND PREPROCESSING")
    results_log.append("="*60)
    try:
        df = pd.read_csv(filepath)
        results_log.append(f"Dataset '{filepath}' loaded successfully.\n")
    except FileNotFoundError:
        error_msg = f"FATAL ERROR: Dataset '{filepath}' not found. Please ensure it is in the correct directory."
        print(error_msg)
        results_log.append(error_msg)
        return None

    # --- Data Cleaning and Feature Engineering ---
    df = df.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere'], errors='ignore')

    if 'Blood Pressure' in df.columns:
        bp_split = df['Blood Pressure'].str.split('/', expand=True)
        df['systolic_bp'] = pd.to_numeric(bp_split[0])
        df['diastolic_bp'] = pd.to_numeric(bp_split[1])
        df = df.drop(columns=['Blood Pressure'])

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    df.fillna(df.median(), inplace=True)

    results_log.append("Data preprocessing and feature engineering complete.")
    results_log.append("Final features: " + ", ".join(df.drop(columns=[TARGET_COLUMN]).columns.tolist()))
    return df

# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def perform_eda(df, pdf_pages, results_log):
    """
    Performs EDA, saves visualizations to a PDF object, and logs summaries.
    """
    results_log.append("\n" + "="*60)
    results_log.append("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    results_log.append("="*60)
    
    # Log basic information and statistics
    results_log.append("\nDataset Information:")
    with io.StringIO() as buffer:
        df.info(buf=buffer)
        results_log.append(buffer.getvalue())
    
    results_log.append("\nDescriptive Statistics:")
    results_log.append(df.describe().to_string())
    
    # Visualize and save the distribution of the target variable to the PDF
    plt.figure(figsize=(8, 6))
    sns.countplot(x=TARGET_COLUMN, data=df, palette='viridis')
    plt.title('Distribution of Heart Attack Risk', fontsize=16)
    plt.xlabel('Heart Attack Risk (0 = No, 1 = Yes)')
    plt.ylabel('Number of Patients')
    pdf_pages.savefig()
    plt.close()

    # Visualize and save the feature correlation heatmap to the PDF
    plt.figure(figsize=(20, 16))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Feature Correlation Matrix', fontsize=18)
    pdf_pages.savefig()
    plt.close()
    
    results_log.append("\nEDA plots saved to PDF file.")

# =============================================================================
# SECTION 4: MODEL TRAINING, TUNING, AND EVALUATION
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name, pdf_pages, labels=['Low Risk', 'High Risk']):
    """
    Generates and saves a confusion matrix heatmap to a PDF object.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    pdf_pages.savefig()
    plt.close()

def train_evaluate_and_tune_models(X, y, pdf_pages, results_log):
    """
    Trains models, performs hyperparameter tuning, evaluates them, and logs results.
    """
    results_log.append("\n" + "="*60)
    results_log.append("STEP 3: MODEL TRAINING, TUNING, AND EVALUATION")
    results_log.append("="*60)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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

    for name, model in models.items():
        results_log.append(f"\n--- Processing Model: {name} ---")
        if name in param_grids:
            results_log.append("Performing Hyperparameter Tuning with GridSearchCV...")
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            best_model_for_name = grid_search.best_estimator_
            results_log.append(f"Best Parameters: {grid_search.best_params_}")
        else:
            model.fit(X_train_scaled, y_train)
            best_model_for_name = model

        y_pred = best_model_for_name.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results_log.append(f"\n{name} Performance:")
        results_log.append(f"  - Accuracy: {accuracy:.4f}")
        results_log.append(f"  - F1-score (Weighted): {f1:.4f}")
        results_log.append("\nClassification Report:")
        results_log.append(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
        
        plot_confusion_matrix(y_test, y_pred, name, pdf_pages)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = best_model_for_name
            best_model_name = name

    results_log.append(f"\n🏆 Best Overall Model: {best_model_name} (F1-score: {best_f1:.4f})")
    
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
# SECTION 6: ADD-ON - MULTINOMIAL (SOFTMAX) REGRESSION
# =============================================================================

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, n_classes):
        one_hot_y = np.zeros((len(y), n_classes))
        one_hot_y[np.arange(len(y)), y] = 1
        return one_hot_y

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))
        y_one_hot = self._one_hot(y, n_classes)
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._softmax(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_one_hot))
            db = (1 / n_samples) * np.sum(y_predicted - y_one_hot, axis=0)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._softmax(linear_model)
        return np.argmax(y_predicted, axis=1)

def run_multinomial_regression_addon(pdf_pages, results_log):
    """
    Orchestrates the multinomial regression add-on and logs its results.
    """
    results_log.append("\n" + "="*80)
    results_log.append("||         ADD-ON: MULTINOMIAL (SOFTMAX) REGRESSION FROM SCRATCH         ||")
    results_log.append("="*80)

    results_log.append("\n[STEP 1] Generating a synthetic multi-class dataset...")
    X_multi, y_multi = make_classification(
        n_samples=1000, n_features=10, n_informative=5, n_redundant=2,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    results_log.append("Dataset generated with 1000 samples, 10 features, and 3 classes.")

    X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results_log.append("\n[STEP 2] Training the from-scratch Softmax Regression model...")
    model = SoftmaxRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X_train_scaled, y_train)
    results_log.append("Model training complete.")

    results_log.append("\n[STEP 3] Evaluating the model performance...")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    results_log.append("\n--- Multinomial Regression Performance ---")
    results_log.append(f"  - Accuracy: {accuracy:.4f}")
    results_log.append(f"  - Precision (Macro): {precision:.4f}")
    results_log.append(f"  - Recall (Macro): {recall:.4f}")
    results_log.append(f"  - F1-score (Macro): {f1:.4f}")
    
    results_log.append("\nClassification Report:")
    results_log.append(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1', 'Class 2']))

    plot_confusion_matrix(y_test, y_pred, "Softmax_Regression", pdf_pages, labels=['Class 0', 'Class 1', 'Class 2'])
    results_log.append("\n" + "="*80)
    results_log.append("||                  END OF MULTINOMIAL REGRESSION ADD-ON                   ||")
    results_log.append("="*80)

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main function to orchestrate the entire ML pipeline."""
    # --- Setup Results Directory and Files ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    results_log = []
    
    pdf_filepath = os.path.join(results_dir, 'all_figures.pdf')
    with PdfPages(pdf_filepath) as pdf_pages:
        # --- Run Main Pipeline ---
        df = load_and_preprocess_data(DATA_FILE, results_log)
        if df is None:
            return
        
        if TARGET_COLUMN not in df.columns:
            error_msg = f"FATAL ERROR: Target column '{TARGET_COLUMN}' not found after preprocessing."
            print(error_msg)
            results_log.append(error_msg)
            return

        perform_eda(df, pdf_pages, results_log)
        
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
        
        best_model, scaler, feature_names = train_evaluate_and_tune_models(X, y, pdf_pages, results_log)
        
        # --- Run Add-on ---
        run_multinomial_regression_addon(pdf_pages, results_log)

        # --- Save Log File ---
        log_filepath = os.path.join(results_dir, 'results_summary.txt')
        # FIX: Specify encoding='utf-8' to prevent UnicodeEncodeError
        with open(log_filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(results_log))
        
        print(f"\n✅ Pipeline complete. All results and figures saved to '{results_dir}'")
        
        # --- Start Interactive Prediction ---
        if best_model:
            interactive_prediction(best_model, scaler, feature_names)
        else:
            results_log.append("\nModel training failed. Cannot proceed to interactive prediction.")


if __name__ == "__main__":
    main()
