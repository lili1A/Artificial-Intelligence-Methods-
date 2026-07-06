import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 60)
print("HEART DISEASE PREDICTION USING SVM")
print("=" * 60)

# Load Dataset
print("\n1. LOADING DATASET")
print("-" * 40)

try:
    data = pd.read_csv("heart.csv")
    print(f"Dataset loaded successfully")
    print(f"Total records: {len(data)}")
    print(f"Features: {len(data.columns) - 1}")
except FileNotFoundError:
    print("Error: 'heart.csv' file not found.")
    print("Please ensure 'heart.csv' is in your working directory")
    exit()

# Prepare Data
print("\n2. PREPARING DATA")
print("-" * 40)

X = data.drop("target", axis=1)
y = data["target"]
feature_names = X.columns.tolist()  # Store feature names for later use

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train SVM Model
print("\n3. TRAINING SVM MODEL")
print("-" * 40)

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=42))
])

# Hyperparameter tuning
param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.01, 0.1]
}

print("Training model with hyperparameter tuning...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Model training completed")
print(f"Best parameters: {grid_search.best_params_}")

# Get feature importance for SVM with linear kernel
if grid_search.best_params_.get('svm__kernel') == 'linear':
    # For linear SVM, we can get coefficients as feature importance
    svm_coef = best_model.named_steps['svm'].coef_[0]
    importances = np.abs(svm_coef)
    indices = np.argsort(importances)[::-1]
else:
    # For non-linear kernels, we'll use permutation importance or default to equal weights
    importances = np.ones(len(feature_names)) / len(feature_names)
    indices = np.arange(len(feature_names))

# Predict All Data
print("\n4. PREDICTING ON ALL DATA")
print("-" * 40)

# Scale and predict on ALL data
scaler = best_model.named_steps['scaler']
X_scaled = scaler.transform(X)
y_pred_all = best_model.predict(X_scaled)
y_prob_all = best_model.predict_proba(X_scaled)[:, 1]

print(f"Predictions made for ALL {len(data)} records")
print(f"Predicted heart disease cases: {y_pred_all.sum()} ({y_pred_all.sum() / len(y_pred_all) * 100:.1f}%)")

# Model Evaluation (Test Data)
print("\n5. MODEL EVALUATION (on test set)")
print("-" * 40)

# Evaluate on test set
y_pred_test = best_model.predict(X_test)
y_prob_test = best_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_prob_test)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Save File
print("\n6. SAVING REQUIRED CSV FILES")
print("-" * 40)

# Create output directory
output_dir = "svm_predictions_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# File 1 All Predict Result
print("\nCreating: ALL PREDICTIONS FILE...")

results_all = data.copy()
results_all["Predicted"] = y_pred_all
results_all["Predicted_Probability"] = y_prob_all
results_all["Risk_Probability (%)"] = (y_prob_all * 100).round(2)


# Add risk categories
def risk_category(risk):
    if risk < 30:
        return "Low Risk"
    elif risk < 70:
        return "Moderate Risk"
    else:
        return "High Risk"


results_all["Risk_Category"] = results_all["Risk_Probability (%)"].apply(risk_category)

# Add prediction status
results_all["Prediction_Correct"] = results_all["target"] == results_all["Predicted"]
results_all["Prediction_Status"] = results_all.apply(
    lambda row: "True Positive" if row["target"] == 1 and row["Predicted"] == 1 else
    "True Negative" if row["target"] == 0 and row["Predicted"] == 0 else
    "False Positive" if row["target"] == 0 and row["Predicted"] == 1 else
    "False Negative", axis=1
)

# Save to CSV
all_predictions_file = f"{output_dir}/all_predictions.csv"
results_all.to_csv(all_predictions_file, index=False)
print(f"✓ Saved: '{all_predictions_file}'")
print(f"  Contains: {len(results_all)} records with predictions")

# Summary Statistic File
print("\nCreating: SUMMARY STATISTICS FILE...")

# Calculate risk category counts
high_risk = results_all[results_all["Risk_Probability (%)"] >= 70]
moderate_risk = results_all[results_all["Risk_Probability (%)"].between(30, 69)]
low_risk = results_all[results_all["Risk_Probability (%)"] < 30]

# Create summary dataframe
summary_data = {
    'Category': [
        'Dataset Information',
        'Total Patients',
        'Actual Heart Disease Cases',
        'Actual No Disease Cases',
        '',
        'Prediction Results (All Data)',
        'Predicted Heart Disease Cases',
        'Predicted No Disease Cases',
        'Correct Predictions',
        'Incorrect Predictions',
        '',
        'Model Performance (Test Set)',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score',
        'ROC-AUC Score',
        '',
        'Confusion Matrix (Test Set)',
        'True Positives',
        'True Negatives',
        'False Positives',
        'False Negatives',
        '',
        'Risk Category Distribution',
        'High Risk Patients (≥70%)',
        'Moderate Risk Patients (30-69%)',
        'Low Risk Patients (<30%)',
        '',
        'Model Configuration',
        'Best Kernel',
        'Best C Parameter',
        'Best Gamma',
        'Cross-Validation Folds'
    ],
    'Value': [
        '',
        str(len(results_all)),
        str(results_all['target'].sum()),
        str(len(results_all) - results_all['target'].sum()),
        '',
        '',
        str(results_all['Predicted'].sum()),
        str(len(results_all) - results_all['Predicted'].sum()),
        str(results_all['Prediction_Correct'].sum()),
        str(len(results_all) - results_all['Prediction_Correct'].sum()),
        '',
        '',
        f"{accuracy:.4f} ({accuracy * 100:.1f}%)",
        f"{precision:.4f} ({precision * 100:.1f}%)",
        f"{recall:.4f} ({recall * 100:.1f}%)",
        f"{f1:.4f} ({f1 * 100:.1f}%)",
        f"{roc_auc:.4f}",
        '',
        '',
        str(tp),
        str(tn),
        str(fp),
        str(fn),
        '',
        '',
        f"{len(high_risk)} ({len(high_risk) / len(results_all) * 100:.1f}%)",
        f"{len(moderate_risk)} ({len(moderate_risk) / len(results_all) * 100:.1f}%)",
        f"{len(low_risk)} ({len(low_risk) / len(results_all) * 100:.1f}%)",
        '',
        '',
        str(grid_search.best_params_.get('svm__kernel')),
        str(grid_search.best_params_.get('svm__C')),
        str(grid_search.best_params_.get('svm__gamma')),
        '5'
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_file = f"{output_dir}/summary_statistics.csv"
summary_df.to_csv(summary_file, index=False)
print(f"✓ Saved: '{summary_file}'")
print(f"  Contains: Comprehensive model and prediction statistics")

# File 3 Performance Metrics
print("\nCreating: PERFORMANCE METRICS FILE (CSV)...")

# Detailed performance metrics
performance_metrics = {
    'Metric': [
        'Accuracy',
        'Precision',
        'Recall (Sensitivity)',
        'Specificity',
        'F1 Score',
        'ROC-AUC Score',
        'False Positive Rate',
        'False Negative Rate',
        'Positive Predictive Value',
        'Negative Predictive Value'
    ],
    'Value': [
        f"{accuracy:.6f}",
        f"{precision:.6f}",
        f"{recall:.6f}",
        f"{tn / (tn + fp):.6f}",  # Specificity
        f"{f1:.6f}",
        f"{roc_auc:.6f}",
        f"{fp / (fp + tn):.6f}",  # FPR
        f"{fn / (fn + tp):.6f}",  # FNR
        f"{precision:.6f}",  # PPV = Precision
        f"{tn / (tn + fn):.6f}"  # NPV
    ],
    'Percentage': [
        f"{accuracy * 100:.2f}%",
        f"{precision * 100:.2f}%",
        f"{recall * 100:.2f}%",
        f"{tn / (tn + fp) * 100:.2f}%",
        f"{f1 * 100:.2f}%",
        f"{roc_auc * 100:.2f}%",
        f"{fp / (fp + tn) * 100:.2f}%",
        f"{fn / (fn + tp) * 100:.2f}%",
        f"{precision * 100:.2f}%",
        f"{tn / (tn + fn) * 100:.2f}%"
    ],
    'Description': [
        'Overall correctness of predictions',
        'Correct positive predictions among all positive predictions',
        'Ability to correctly identify actual positives',
        'Ability to correctly identify actual negatives',
        'Harmonic mean of precision and recall',
        'Ability to distinguish between classes',
        'Proportion of negatives incorrectly classified as positives',
        'Proportion of positives incorrectly classified as negatives',
        'Proportion of true positives among positive predictions',
        'Proportion of true negatives among negative predictions'
    ]
}

performance_df = pd.DataFrame(performance_metrics)
performance_file = f"{output_dir}/performance_metrics.csv"
performance_df.to_csv(performance_file, index=False)
print(f"✓ Saved: '{performance_file}'")

# Also save confusion matrix details as separate CSV
print("\nCreating: CONFUSION MATRIX DETAILS FILE...")

confusion_details = {
    'Matrix_Element': ['True Positives (TP)', 'True Negatives (TN)',
                       'False Positives (FP)', 'False Negatives (FN)'],
    'Count': [str(tp), str(tn), str(fp), str(fn)],
    'Percentage_of_Test_Set': [
        f"{tp / len(y_test) * 100:.2f}%",
        f"{tn / len(y_test) * 100:.2f}%",
        f"{fp / len(y_test) * 100:.2f}%",
        f"{fn / len(y_test) * 100:.2f}%"
    ],
    'Description': [
        'Correctly predicted heart disease cases',
        'Correctly predicted healthy cases',
        'Healthy cases incorrectly predicted as heart disease',
        'Heart disease cases incorrectly predicted as healthy'
    ]
}

confusion_df = pd.DataFrame(confusion_details)
confusion_file = f"{output_dir}/confusion_matrix_details.csv"
confusion_df.to_csv(confusion_file, index=False)
print(f"✓ Saved: '{confusion_file}'")

# Also save model configuration as separate CSV
print("\nCreating: MODEL CONFIGURATION FILE...")

model_config = {
    'Parameter': [
        'Model Type',
        'Best Kernel',
        'Best C Parameter',
        'Best Gamma',
        'Cross-Validation Folds',
        'Scoring Metric',
        'Test Set Size',
        'Training Set Size',
        'Total Features',
        'Total Patients'
    ],
    'Value': [
        'Support Vector Machine (SVC)',
        str(grid_search.best_params_.get('svm__kernel')),
        str(grid_search.best_params_.get('svm__C')),
        str(grid_search.best_params_.get('svm__gamma')),
        '5',
        'F1 Score',
        str(len(X_test)),
        str(len(X_train)),
        str(X.shape[1]),
        str(len(results_all))
    ],
    'Description': [
        'Type of machine learning algorithm used',
        'Kernel function used for SVM',
        'Regularization parameter',
        'Kernel coefficient',
        'Number of cross-validation folds used',
        'Metric optimized during training',
        'Number of samples in test set',
        'Number of samples in training set',
        'Number of input features',
        'Total number of patients analyzed'
    ]
}

model_config_df = pd.DataFrame(model_config)
model_config_file = f"{output_dir}/model_configuration.csv"
model_config_df.to_csv(model_config_file, index=False)
print(f"✓ Saved: '{model_config_file}'")

# Display All Summary
print("\n" + "=" * 60)
print("PROGRAM COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"\n✓ ALL CSV FILES CREATED in '{output_dir}' directory:")
print(f"  1. all_predictions.csv - All patient predictions")
print(f"  2. summary_statistics.csv - Summary statistics")
print(f"  3. performance_metrics.csv - Performance metrics")
print(f"  4. confusion_matrix_details.csv - Confusion matrix breakdown")
print(f"  5. model_configuration.csv - Model configuration")

print(f"\n SUMMARY OF RESULTS:")
print(f"• Total patients analyzed: {len(results_all)}")
print(f"• Model accuracy: {accuracy:.1%}")
print(f"• High-risk patients identified: {len(high_risk)}")

print(f"\n PREDICTION ACCURACY (All Data):")
print(f"  Correct predictions: {results_all['Prediction_Correct'].sum()}")
print(f"  Accuracy rate: {results_all['Prediction_Correct'].mean() * 100:.1f}%")

print(f"\n️  RISK DISTRIBUTION:")
print(f"  High Risk (≥70%): {len(high_risk)} patients ({len(high_risk)/len(results_all)*100:.1f}%)")
print(f"  Moderate Risk (30-69%): {len(moderate_risk)} patients ({len(moderate_risk)/len(results_all)*100:.1f}%)")
print(f"  Low Risk (<30%): {len(low_risk)} patients ({len(low_risk)/len(results_all)*100:.1f}%)")

print(f"\n MODEL CONFIGURATION:")
print(f"  Kernel: {grid_search.best_params_.get('svm__kernel')}")
print(f"  C Parameter: {grid_search.best_params_.get('svm__C')}")
print(f"  Gamma: {grid_search.best_params_.get('svm__gamma')}")

print(f"\n FIRST 5 PREDICTIONS:")
print("-" * 50)
sample_df = results_all[['age', 'sex', 'target', 'Predicted',
                         'Risk_Probability (%)', 'Risk_Category', 'Prediction_Status']].head(5)
print(sample_df.to_string(index=False))

# Display top 5 most important features for SVM (if linear kernel)
print(f"\n TOP 5 MOST IMPORTANT FEATURES:")
print("-" * 50)
if grid_search.best_params_.get('svm__kernel') == 'linear':
    for i in range(min(5, len(feature_names))):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
else:
    print("Feature importance is not available for non-linear SVM kernels")
    print("Using linear kernel provides interpretable feature weights")

print(f"\n All required files have been saved successfully!")
