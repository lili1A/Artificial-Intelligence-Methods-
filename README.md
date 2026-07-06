# Artificial-Intelligence-Methods-
This is the code part for the AIM group assignment. This assignment implements multiple machine learning algorithms testing on identifying and predicting the diagnosis of heart disease 

# Heart Disease Risk Prediction — Run Guide

Three standalone scripts predict heart disease risk from `heart.csv` using different ML algorithms: **Random Forest**, **SVM**, and **Gradient Boosting**. Each trains a model, tunes hyperparameters, evaluates performance, and exports risk predictions to CSV.

## 1. Requirements

- Python 3.8+
- Packages: `pandas`, `numpy`, `scikit-learn`

Install:
```bash
pip install pandas numpy scikit-learn
```

## 2. Folder setup

All three scripts expect `heart.csv` in the **same directory** they're run from. Your current layout already satisfies this:

```
AI ASS/
├── heart.csv
├── RandomForest.py
├── SVM.py
└── GradientBoosting.py
```

`heart.csv` columns: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target` (`target` = 1 has heart disease, 0 does not).

## 3. Running each script

From inside the `AI ASS` folder, run one at a time:

```bash
python RandomForest.py
python SVM.py
python GradientBoosting.py
```

Each run:
1. Loads `heart.csv` and splits 80/20 train/test (stratified).
2. Scales features with `StandardScaler`.
3. Tunes hyperparameters via `GridSearchCV` (5-fold CV, optimizing F1 score).
4. Predicts on the full dataset and on the held-out test set.
5. Prints accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix to the terminal.
6. Saves results as CSVs in a new output folder.

Runtime: SVM and Gradient Boosting grid searches can take a few minutes depending on your machine; Random Forest's grid is larger and may take longest.

## 4. What each algorithm tunes

| Script | Algorithm | Key hyperparameters searched |
|---|---|---|
| `RandomForest.py` | Random Forest (`class_weight='balanced'`) | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| `SVM.py` | Support Vector Machine | `kernel` (linear/rbf), `C`, `gamma` |
| `GradientBoosting.py` | Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split` |

## 5. Output files

Each script creates its own output folder (`rf_predictions_output/`, `svm_predictions_output/`, `gb_predictions_output/`) containing:

- **all_predictions.csv** — every patient's original data plus predicted label, predicted probability, risk % (0–100), risk category (Low <30%, Moderate 30–69%, High ≥70%), and whether the prediction was correct.
- **summary_statistics.csv** — dataset totals, prediction counts, test-set performance metrics, confusion matrix, risk distribution, and best hyperparameters found.
- **performance_metrics.csv** — accuracy, precision, recall, specificity, F1, ROC-AUC, false positive/negative rates, PPV, NPV.
- **feature_importance.csv** (RF and GB only) — features ranked by importance to the model.
- SVM additionally saves **confusion_matrix_details.csv** and **model_configuration.csv**.

Note: SVM only produces feature importance/coefficients if the grid search picks a **linear** kernel; with an RBF kernel it prints a message that importances aren't available.

## 6. Interpreting results

- Compare `performance_metrics.csv` across the three output folders to see which algorithm generalizes best on the test set (look at F1 and ROC-AUC, not just accuracy, since the two classes may be imbalanced).
- Use `feature_importance.csv` to identify which clinical measurements (e.g., `cp`, `thalach`, `oldpeak`) drive risk the most.
- `all_predictions.csv` lets you flag "High Risk" patients (≥70% probability) for further review.

## 7. Common issues

- **`FileNotFoundError: heart.csv`** — run the script from inside the folder containing `heart.csv`, or move the script there.
- **Long runtime** — reduce the `param_grid` values in the script if you just need a quick check.
- **Re-running** — each script overwrites its own output folder's CSVs; it doesn't touch the other two algorithms' outputs.
