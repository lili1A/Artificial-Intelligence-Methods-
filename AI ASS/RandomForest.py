import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    classification_report
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 60)
print("HEART DISEASE PREDICTION USING RANDOM FOREST")
print("=" * 60)


# LOAD DATASET
print("\n1. LOADING DATASET")
print("-" * 40)

try:
    data = pd.read_csv("heart.csv")
    print(f"✓ Dataset loaded successfully")
    print(f"✓ Total records: {len(data)}")
    print(f"✓ Features: {len(data.columns) - 1}")
except FileNotFoundError:
    print("✗ Error: 'heart.csv' file not found.")
    print("Please ensure 'heart.csv' is in your working directory")
    exit()


# PREPARE DATA
print("\n2. PREPARING DATA")
print("-" * 40)

X = data.drop("target", axis=1)
y = data["target"]
feature_names = X.columns.tolist()

print(f"✓ Features: {len(feature_names)}")
print(f"✓ Samples: {len(data)}")
print(f"✓ Patients with heart disease: {(y == 1).sum()} ({((y == 1).sum() / len(y)) * 100:.1f}%)")
print(f"✓ Patients without heart disease: {(y == 0).sum()} ({((y == 0).sum() / len(y)) * 100:.1f}%)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")


# TRAIN RANDOM FOREST MODEL
print("\n3. TRAINING RANDOM FOREST MODEL")
print("-" * 40)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Random Forest classifier
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ['sqrt', 'log2']
}

print("Training model with hyperparameter tuning...")
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print(f"✓ Model training completed")
print(f"✓ Best parameters: {grid_search.best_params_}")


# PREDICT ON ALL DATA

print("\n4. PREDICTING ON ALL DATA")
print("-" * 40)

# Scale ALL data for prediction
X_scaled = scaler.transform(X)

# Predict on ALL data
y_pred_all = best_model.predict(X_scaled)
y_prob_all = best_model.predict_proba(X_scaled)[:, 1]

print(f"✓ Predictions made for ALL {len(data)} records")
print(f"✓ Predicted heart disease cases: {y_pred_all.sum()} ({y_pred_all.sum() / len(y_pred_all) * 100:.1f}%)")


# MODEL EVALUATION (on test set)
print("\n5. MODEL EVALUATION (on test set)")
print("-" * 40)

# Evaluate on test set
y_pred_test = best_model.predict(X_test_scaled)
y_prob_test = best_model.predict_proba(X_test_scaled)[:, 1]

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


# SAVE THE REQUIRED CSV FILES
print("\n6. SAVING REQUIRED CSV FILES")
print("-" * 40)

# Create output directory
output_dir = "rf_predictions_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")


# ALL PREDICTIONS FILE
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


# FILE 2 SUMMARY STATISTICS FILE
print("\nCreating: SUMMARY STATISTICS FILE...")

# Calculate risk category counts
high_risk = results_all[results_all["Risk_Probability (%)"] >= 70]
moderate_risk = results_all[results_all["Risk_Probability (%)"].between(30, 69)]
low_risk = results_all[results_all["Risk_Probability (%)"] < 30]

# Get feature importances
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_feature = feature_names[indices[0]] if len(feature_names) > 0 else "N/A"

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
        'Best n_estimators',
        'Best max_depth',
        'Best min_samples_split',
        'Best min_samples_leaf',
        'Best max_features',
        'Top Feature',
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
        f"{accuracy:.4f} ({accuracy*100:.1f}%)",
        f"{precision:.4f} ({precision*100:.1f}%)",
        f"{recall:.4f} ({recall*100:.1f}%)",
        f"{f1:.4f} ({f1*100:.1f}%)",
        f"{roc_auc:.4f}",
        '',
        '',
        str(tp),
        str(tn),
        str(fp),
        str(fn),
        '',
        '',
        f"{len(high_risk)} ({len(high_risk)/len(results_all)*100:.1f}%)",
        f"{len(moderate_risk)} ({len(moderate_risk)/len(results_all)*100:.1f}%)",
        f"{len(low_risk)} ({len(low_risk)/len(results_all)*100:.1f}%)",
        '',
        '',
        str(grid_search.best_params_.get('n_estimators')),
        str(grid_search.best_params_.get('max_depth')),
        str(grid_search.best_params_.get('min_samples_split')),
        str(grid_search.best_params_.get('min_samples_leaf')),
        str(grid_search.best_params_.get('max_features')),
        top_feature,
        '5'
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_file = f"{output_dir}/summary_statistics.csv"
summary_df.to_csv(summary_file, index=False)
print(f"✓ Saved: '{summary_file}'")
print(f"  Contains: Comprehensive model and prediction statistics")


# FILE 3 PERFORMANCE METRICS FILE
print("\nCreating: PERFORMANCE METRICS FILE...")

# Main performance metrics
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
        f"{tn/(tn+fp):.6f}",  # Specificity
        f"{f1:.6f}",
        f"{roc_auc:.6f}",
        f"{fp/(fp+tn):.6f}",  # FPR
        f"{fn/(fn+tp):.6f}",  # FNR
        f"{precision:.6f}",  # PPV = Precision
        f"{tn/(tn+fn):.6f}"   # NPV
    ],
    'Percentage': [
        f"{accuracy*100:.2f}%",
        f"{precision*100:.2f}%",
        f"{recall*100:.2f}%",
        f"{tn/(tn+fp)*100:.2f}%",
        f"{f1*100:.2f}%",
        f"{roc_auc*100:.2f}%",
        f"{fp/(fp+tn)*100:.2f}%",
        f"{fn/(fn+tp)*100:.2f}%",
        f"{precision*100:.2f}%",
        f"{tn/(tn+fn)*100:.2f}%"
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


print("\nFEATURE IMPORTANCE FILE")

feature_importance_data = []
for i in range(len(feature_names)):
    feature_importance_data.append({
        'Feature': feature_names[indices[i]],
        'Importance': importances[indices[i]],
        'Rank': i + 1
    })

feature_importance_df = pd.DataFrame(feature_importance_data)
feature_importance_file = f"{output_dir}/feature_importance.csv"
feature_importance_df.to_csv(feature_importance_file, index=False)
print(f"✓ Saved: '{feature_importance_file}' (Bonus file)")

# ===============================
# 7. FINAL SUMMARY
# ===============================
print("\n" + "=" * 60)
print("PROGRAM COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"\n REQUIRED CSV FILES CREATED in '{output_dir}' directory:")
print(f"  1. all_predictions.csv - All patient predictions")
print(f"  2. summary_statistics.csv - Summary statistics")
print(f"  3. performance_metrics.csv - Performance metrics")
print(f"  4. feature_importance.csv - Feature importance rankings")

print(f"\n SUMMARY OF RESULTS:")
print(f"• Total patients analyzed: {len(results_all)}")
print(f"• Model accuracy: {accuracy:.1%}")
print(f"• High-risk patients identified: {len(high_risk)}")
print(f"• Most important feature: {top_feature}")

print(f"\n PREDICTION ACCURACY (All Data):")
print(f"  Correct predictions: {results_all['Prediction_Correct'].sum()}")
print(f"  Accuracy rate: {results_all['Prediction_Correct'].mean()*100:.1f}%")

print(f"\n  RISK DISTRIBUTION:")
print(f"  High Risk (≥70%): {len(high_risk)} patients ({len(high_risk)/len(results_all)*100:.1f}%)")
print(f"  Moderate Risk (30-69%): {len(moderate_risk)} patients ({len(moderate_risk)/len(results_all)*100:.1f}%)")
print(f"  Low Risk (<30%): {len(low_risk)} patients ({len(low_risk)/len(results_all)*100:.1f}%)")

print(f"\n FIRST 5 PREDICTIONS:")
print("-" * 50)
sample_df = results_all[['age', 'sex', 'target', 'Predicted',
                         'Risk_Probability (%)', 'Risk_Category', 'Prediction_Status']].head(5)
print(sample_df.to_string(index=False))

print(f"\n TOP 5 MOST IMPORTANT FEATURES:")
print("-" * 50)
for i in range(min(5, len(feature_names))):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

print(f"\n All 3 required files have been saved successfully!")