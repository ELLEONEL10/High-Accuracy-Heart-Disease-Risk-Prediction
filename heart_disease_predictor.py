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
# --- Core Libraries ---
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import os  # For interacting with the operating system (e.g., creating directories)
import warnings  # To control warning messages
import io  # To manage in-memory text streams (for logging df.info())
from datetime import datetime  # To timestamp the results folder

# --- Visualization Libraries ---
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns  # A high-level interface for drawing attractive and informative statistical graphics
from matplotlib.backends.backend_pdf import PdfPages  # To save multiple plots to a single PDF

# --- Scikit-learn Utilities ---
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.metrics import (  # For model evaluation
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.datasets import make_classification  # To generate a synthetic dataset for the multinomial add-on

# --- Machine Learning Models ---
from sklearn.linear_model import LogisticRegression  # Linear model for classification
from sklearn.svm import SVC  # Support Vector Classifier (commented out, but available)
from sklearn.ensemble import RandomForestClassifier  # Ensemble model (bagging)
from xgboost import XGBClassifier  # Gradient boosting model, known for high performance

# --- Configuration ---
# Suppress warnings for a cleaner, more readable output console.
warnings.filterwarnings('ignore')

# Define global constants for file paths and key column names to avoid magic strings.
DATA_FILE = 'heart.csv'
TARGET_COLUMN = 'heart_attack_risk'

# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filepath, results_log):
    """
    Loads data from a CSV, cleans it, performs feature engineering, and logs the process.

    Args:
        filepath (str): The path to the heart disease dataset (CSV).
        results_log (list): A list to append log messages to.

    Returns:
        pd.DataFrame or None: The preprocessed DataFrame, or None if the file is not found.
    """
    # Start of the data preprocessing section in the log.
    results_log.append("="*60)
    results_log.append("STEP 1: DATA LOADING AND PREPROCESSING")
    results_log.append("="*60)
    
    # Safely load the dataset with error handling for missing files.
    try:
        df = pd.read_csv(filepath)
        results_log.append(f"Dataset '{filepath}' loaded successfully.\n")
    except FileNotFoundError:
        error_msg = f"FATAL ERROR: Dataset '{filepath}' not found. Please ensure it is in the correct directory."
        print(error_msg)
        results_log.append(error_msg)
        return None # Exit the function if the file doesn't exist.

    # --- Data Cleaning and Feature Engineering ---
    # Drop columns that are identifiers or have too many unique values to be useful.
    df = df.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere'], errors='ignore')

    # Engineer new features from existing ones. Here, split 'Blood Pressure' into systolic and diastolic.
    if 'Blood Pressure' in df.columns:
        bp_split = df['Blood Pressure'].str.split('/', expand=True)
        df['systolic_bp'] = pd.to_numeric(bp_split[0])
        df['diastolic_bp'] = pd.to_numeric(bp_split[1])
        df = df.drop(columns=['Blood Pressure']) # Drop the original column.

    # One-Hot Encode categorical features to convert them into a numerical format.
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Standardize column names to a consistent format (lowercase with underscores).
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Handle missing values by filling them with the median of their respective columns.
    # Median is chosen as it is robust to outliers.
    df.fillna(df.median(), inplace=True)

    results_log.append("Data preprocessing and feature engineering complete.")
    results_log.append("Final features: " + ", ".join(df.drop(columns=[TARGET_COLUMN]).columns.tolist()))
    return df

# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def perform_eda(df, pdf_pages, results_log):
    """
    Performs EDA, generates key visualizations, saves them to a PDF, and logs summaries.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        pdf_pages (PdfPages): The PDF object to save figures to.
        results_log (list): A list to append log messages to.
    """
    results_log.append("\n" + "="*60)
    results_log.append("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    results_log.append("="*60)
    
    # Log basic DataFrame info (column types, non-null counts) to the log file.
    # Use io.StringIO to capture the output of df.info() as a string.
    results_log.append("\nDataset Information:")
    with io.StringIO() as buffer:
        df.info(buf=buffer)
        results_log.append(buffer.getvalue())
    
    # Log descriptive statistics (mean, std, quartiles) for a numerical overview.
    results_log.append("\nDescriptive Statistics:")
    results_log.append(df.describe().to_string())
    
    # --- Visualizations ---
    # 1. Distribution of the Target Variable (Heart Attack Risk)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=TARGET_COLUMN, data=df, palette='viridis')
    plt.title('Distribution of Heart Attack Risk', fontsize=16)
    plt.xlabel('Heart Attack Risk (0 = No, 1 = Yes)')
    plt.ylabel('Number of Patients')
    pdf_pages.savefig() # Save the current figure to the PDF.
    plt.close() # Close the plot to free up memory.

    # 2. Feature Correlation Heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Feature Correlation Matrix', fontsize=18)
    pdf_pages.savefig() # Save the heatmap to the PDF.
    plt.close() # Close the plot.
    
    results_log.append("\nEDA plots (Target Distribution, Correlation Heatmap) saved to PDF file.")

# =============================================================================
# SECTION 4: MODEL TRAINING, TUNING, AND EVALUATION
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name, pdf_pages, labels=['Low Risk', 'High Risk']):
    """
    Generates and saves a styled confusion matrix heatmap to the PDF object.

    Args:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels from the model.
        model_name (str): The name of the model for the plot title.
        pdf_pages (PdfPages): The PDF object to save the figure to.
        labels (list): The labels for the axes.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    pdf_pages.savefig() # Save the confusion matrix plot to the PDF.
    plt.close()

def train_evaluate_and_tune_models(X, y, pdf_pages, results_log):
    """
    Splits data, scales features, trains multiple models, performs hyperparameter tuning,
    evaluates them, and logs all results.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        pdf_pages (PdfPages): The PDF object to save figures to.
        results_log (list): A list to append log messages to.

    Returns:
        tuple: A tuple containing the best model object, the fitted scaler, and feature names.
    """
    results_log.append("\n" + "="*60)
    results_log.append("STEP 3: MODEL TRAINING, TUNING, AND EVALUATION")
    results_log.append("="*60)

    # Split data into training and testing sets (80/20 split).
    # `stratify=y` ensures the same proportion of target classes in both sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features using StandardScaler to normalize the data (mean=0, variance=1).
    # This is crucial for models like Logistic Regression and SVMs.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define the models to be evaluated.
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    # Define hyperparameter grids for models that will be tuned.
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
    
    # Variables to track the best-performing model based on F1-score.
    best_model = None
    best_f1 = 0
    best_model_name = ""

    # Loop through each model to train, tune, and evaluate.
    for name, model in models.items():
        results_log.append(f"\n--- Processing Model: {name} ---")
        
        # If a parameter grid is defined for the model, perform hyperparameter tuning.
        if name in param_grids:
            results_log.append("Performing Hyperparameter Tuning with GridSearchCV...")
            # Use GridSearchCV to find the best hyperparameters via 5-fold cross-validation.
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            best_model_for_name = grid_search.best_estimator_ # The model with the best found parameters.
            results_log.append(f"Best Parameters: {grid_search.best_params_}")
        else:
            # If no grid is defined, train the model with its default parameters.
            model.fit(X_train_scaled, y_train)
            best_model_for_name = model

        # Evaluate the tuned (or default) model on the test set.
        y_pred = best_model_for_name.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log the performance metrics.
        results_log.append(f"\n{name} Performance:")
        results_log.append(f"  - Accuracy: {accuracy:.4f}")
        results_log.append(f"  - F1-score (Weighted): {f1:.4f}")
        results_log.append("\nClassification Report:")
        results_log.append(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
        
        # Plot and save the confusion matrix for this model.
        plot_confusion_matrix(y_test, y_pred, name, pdf_pages)
        
        # Check if this model is the best one seen so far.
        if f1 > best_f1:
            best_f1 = f1
            best_model = best_model_for_name
            best_model_name = name

    # Announce the best overall model.
    results_log.append(f"\n🏆 Best Overall Model: {best_model_name} (F1-score: {best_f1:.4f})")
    
    return best_model, scaler, X.columns

# =============================================================================
# SECTION 5: INTERACTIVE PREDICTION
# =============================================================================

def interactive_prediction(model, scaler, feature_names):
    """
    Launches an interactive command-line interface for real-time predictions.

    Args:
        model: The trained machine learning model.
        scaler: The fitted StandardScaler object.
        feature_names (list): The list of feature names the model expects.
    """
    print("\n" + "="*60)
    print("STEP 4: INTERACTIVE RISK PREDICTION")
    print("="*60)
    risk_mapping = {0: 'Low Risk', 1: 'High Risk'} # For user-friendly output.

    while True:
        print("\nEnter new patient data for prediction (or type 'quit' to exit):")
        input_data = {}
        # Loop through each feature and ask the user for input.
        for feature in feature_names:
            while True:
                try:
                    value = input(f"  -> Enter value for '{feature}': ")
                    if value.lower() == 'quit':
                        print("\nExiting interactive predictor. Goodbye!")
                        return # Exit the function and the script.
                    # Convert input to a number.
                    input_data[feature] = float(value)
                    break # Exit the inner loop if input is valid.
                except ValueError:
                    print("    Invalid input. Please enter a numerical value.")
        
        # Convert the collected data into a DataFrame, ensuring column order is correct.
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names] 
        
        # Scale the user's input using the same scaler fitted on the training data.
        input_scaled = scaler.transform(input_df)
        
        # Make the prediction and get the probabilities.
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        confidence = prediction_proba[prediction] * 100
        
        # Display the results in a formatted block.
        print("\n" + "-"*25)
        print("   PREDICTION RESULT")
        print("-"*25)
        print(f"  Predicted Risk Level: {risk_mapping[prediction].upper()}")
        print(f"  Confidence: {confidence:.2f}%")
        print("-"*25)
        
        # Ask the user if they want to make another prediction.
        another = input("\nMake another prediction? (yes/no): ")
        if another.lower() not in ['yes', 'y']:
            break # Exit the main loop if the answer isn't 'yes'.
            
    print("\nExiting interactive predictor. Goodbye!")

# =============================================================================
# SECTION 6: ADD-ON - MULTINOMIAL (SOFTMAX) REGRESSION FROM SCRATCH
# =============================================================================

class SoftmaxRegression:
    """
    A from-scratch implementation of Multinomial (Softmax) Regression using NumPy.
    This demonstrates the underlying math of the algorithm.
    """
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """Initializes the model with hyperparameters."""
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _softmax(self, z):
        """
        Computes the softmax function for a set of scores `z`.
        Subtracting max(z) is a trick for numerical stability to prevent overflow.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, n_classes):
        """Converts a 1D array of labels into a 2D one-hot encoded matrix."""
        one_hot_y = np.zeros((len(y), n_classes))
        one_hot_y[np.arange(len(y)), y] = 1 # Set the appropriate index to 1 for each sample.
        return one_hot_y

    def fit(self, X, y):
        """
        Trains the model using gradient descent.
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Initialize weights and bias to zeros.
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))
        
        # Convert target vector to one-hot encoding for gradient calculation.
        y_one_hot = self._one_hot(y, n_classes)
        
        # Gradient Descent loop.
        for _ in range(self.n_iters):
            # Calculate the linear model output (scores).
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply softmax to get probabilities.
            y_predicted = self._softmax(linear_model)
            
            # Calculate the gradients of the cross-entropy loss.
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_one_hot))
            db = (1 / n_samples) * np.sum(y_predicted - y_one_hot, axis=0)
            
            # Update weights and bias.
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Makes predictions on new data using the trained weights.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._softmax(linear_model)
        # Return the class with the highest probability for each sample.
        return np.argmax(y_predicted, axis=1)

def run_multinomial_regression_addon(pdf_pages, results_log):
    """
    Orchestrates the creation, training, and evaluation of the from-scratch
    Softmax Regression model on a synthetic multi-class dataset.
    """
    results_log.append("\n" + "="*80)
    results_log.append("||             ADD-ON: MULTINOMIAL (SOFTMAX) REGRESSION FROM SCRATCH             ||")
    results_log.append("="*80)

    # [STEP 1] Generate a synthetic dataset suitable for multi-class classification.
    results_log.append("\n[STEP 1] Generating a synthetic multi-class dataset...")
    X_multi, y_multi = make_classification(
        n_samples=1000, n_features=10, n_informative=5, n_redundant=2,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    results_log.append("Dataset generated with 1000 samples, 10 features, and 3 classes.")

    # Standard train-test split and scaling process.
    X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # [STEP 2] Train the custom Softmax Regression model.
    results_log.append("\n[STEP 2] Training the from-scratch Softmax Regression model...")
    model = SoftmaxRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X_train_scaled, y_train)
    results_log.append("Model training complete.")

    # [STEP 3] Evaluate the model's performance.
    results_log.append("\n[STEP 3] Evaluating the model performance...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate standard classification metrics. 'macro' average is used for multi-class problems.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Log the performance metrics.
    results_log.append("\n--- Multinomial Regression Performance ---")
    results_log.append(f"  - Accuracy: {accuracy:.4f}")
    results_log.append(f"  - Precision (Macro): {precision:.4f}")
    results_log.append(f"  - Recall (Macro): {recall:.4f}")
    results_log.append(f"  - F1-score (Macro): {f1:.4f}")
    
    results_log.append("\nClassification Report:")
    results_log.append(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1', 'Class 2']))

    # Plot and save the confusion matrix.
    plot_confusion_matrix(y_test, y_pred, "Softmax_Regression", pdf_pages, labels=['Class 0', 'Class 1', 'Class 2'])
    results_log.append("\n" + "="*80)
    results_log.append("||                         END OF MULTINOMIAL REGRESSION ADD-ON                      ||")
    results_log.append("="*80)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main function to orchestrate the entire ML pipeline."""
    # --- Setup Results Directory and Files ---
    # Create a unique directory for this run using a timestamp.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize a list to hold all log messages.
    results_log = []
    
    # Use PdfPages context manager to automatically handle the creation and saving of a single PDF file.
    pdf_filepath = os.path.join(results_dir, 'all_figures.pdf')
    with PdfPages(pdf_filepath) as pdf_pages:
        # --- Run Main Pipeline ---
        # Step 1: Load and preprocess data.
        df = load_and_preprocess_data(DATA_FILE, results_log)
        if df is None:
            # If data loading fails, write the log and exit.
            log_filepath = os.path.join(results_dir, 'error_log.txt')
            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(results_log))
            return

        # Defensive check to ensure the target column exists after preprocessing.
        if TARGET_COLUMN not in df.columns:
            error_msg = f"FATAL ERROR: Target column '{TARGET_COLUMN}' not found after preprocessing."
            print(error_msg)
            results_log.append(error_msg)
            return

        # Step 2: Perform EDA and save plots.
        perform_eda(df, pdf_pages, results_log)
        
        # Separate features (X) and target (y).
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
        
        # Step 3: Train, tune, and evaluate models.
        best_model, scaler, feature_names = train_evaluate_and_tune_models(X, y, pdf_pages, results_log)
        
        # --- Run Add-on ---
        # Run the separate from-scratch multinomial regression example.
        run_multinomial_regression_addon(pdf_pages, results_log)

    # --- Save Log File ---
    # After all processes are complete (including PDF generation), write the log list to a text file.
    log_filepath = os.path.join(results_dir, 'results_summary.txt')
    # FIX: Specify encoding='utf-8' to prevent UnicodeEncodeError, especially with emojis like 🏆.
    with open(log_filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(results_log))
    
    print(f"\n✅ Pipeline complete. All results and figures saved to '{results_dir}'")
    
    # --- Start Interactive Prediction ---
    # If a model was successfully trained, launch the interactive prediction module.
    if best_model:
        interactive_prediction(best_model, scaler, feature_names)
    else:
        # This case would be rare but is good practice to handle.
        results_log.append("\nModel training failed. Cannot proceed to interactive prediction.")
        print("Model training failed. Cannot proceed to interactive prediction.")


# This ensures the `main()` function is called only when the script is executed directly.
if __name__ == "__main__":
    main()
