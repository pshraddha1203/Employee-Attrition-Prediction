# ============================================================
# EMPLOYEE ATTRITION PREDICTION USING CLASSIFICATION MODELS
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


print("=" * 70)
print("EMPLOYEE ATTRITION PREDICTION USING CLASSIFICATION MODELS")
print("=" * 70)

print("\nLibraries imported successfully!")


# ============================================================
# 2. CREATE OUTPUT AND MODEL FOLDERS
# ============================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

train_path = "dataset/train.csv"
test_path = "dataset/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nTraining Dataset Shape:", train_df.shape)
print("Testing Dataset Shape :", test_df.shape)


# ============================================================
# 4. FIRST FIVE ROWS
# ============================================================

print("\nFirst 5 Rows:")
print(train_df.head())


# ============================================================
# 5. COLUMN NAMES
# ============================================================

print("\nDataset Columns:")
for column in train_df.columns:
    print("-", column)


# ============================================================
# 6. DATA TYPES
# ============================================================

print("\nData Types:")
print(train_df.dtypes)


# ============================================================
# 7. STATISTICAL SUMMARY
# ============================================================

print("\nStatistical Summary:")
print(train_df.describe())


# ============================================================
# 8. MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(train_df.isnull().sum())


# ============================================================
# 9. DUPLICATE VALUES
# ============================================================

duplicate_count = train_df.duplicated().sum()

print("\nDuplicate Rows:", duplicate_count)


# ============================================================
# 10. ATTRITION DISTRIBUTION
# ============================================================

print("\nAttrition Distribution:")
print(train_df["Attrition"].value_counts())

print("\nAttrition Percentage:")
print(
    train_df["Attrition"]
    .value_counts(normalize=True) * 100
)


# ============================================================
# 11. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ============================================================
# GRAPH 1 - ATTRITION DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=train_df,
    x="Attrition"
)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "outputs/attrition_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 2 - JOB SATISFACTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=train_df,
    x="Job Satisfaction",
    hue="Attrition"
)

plt.title("Employee Attrition vs Job Satisfaction")
plt.xlabel("Job Satisfaction")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "outputs/attrition_vs_job_satisfaction.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 3 - OVERTIME
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=train_df,
    x="Overtime",
    hue="Attrition"
)

plt.title("Employee Attrition vs Overtime")
plt.xlabel("Overtime")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "outputs/attrition_vs_overtime.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 4 - WORK LIFE BALANCE
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=train_df,
    x="Work-Life Balance",
    hue="Attrition"
)

plt.title("Employee Attrition vs Work-Life Balance")
plt.xlabel("Work-Life Balance")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "outputs/attrition_vs_worklife_balance.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 5 - JOB ROLE
# ============================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=train_df,
    x="Job Role",
    hue="Attrition"
)

plt.title("Employee Attrition vs Job Role")
plt.xlabel("Job Role")
plt.ylabel("Number of Employees")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/attrition_vs_job_role.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 6 - AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=train_df,
    x="Age",
    hue="Attrition",
    kde=True
)

plt.title("Age Distribution by Employee Attrition")
plt.xlabel("Age")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "outputs/age_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 7 - MONTHLY INCOME
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=train_df,
    x="Monthly Income",
    hue="Attrition",
    kde=True
)

plt.title("Monthly Income Distribution by Attrition")
plt.xlabel("Monthly Income")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "outputs/monthly_income_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 8 - YEARS AT COMPANY
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=train_df,
    x="Attrition",
    y="Years at Company"
)

plt.title("Years at Company vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Years at Company")

plt.tight_layout()

plt.savefig(
    "outputs/years_at_company.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)


# Make copies
train_data = train_df.copy()
test_data = test_df.copy()


# ============================================================
# REMOVE EMPLOYEE ID
# ============================================================

if "Employee ID" in train_data.columns:
    train_data.drop(
        "Employee ID",
        axis=1,
        inplace=True
    )

if "Employee ID" in test_data.columns:
    test_data.drop(
        "Employee ID",
        axis=1,
        inplace=True
    )


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = train_data.drop(
    "Attrition",
    axis=1
)

y = train_data["Attrition"]

X_test = test_data.drop(
    "Attrition",
    axis=1
)

y_test = test_data["Attrition"]


print("\nOriginal Feature Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# 13. CATEGORICAL ENCODING
# ============================================================

print("\nEncoding categorical variables...")

# Convert categorical columns into numerical dummy variables
X_encoded = pd.get_dummies(
    X,
    drop_first=True
)

X_test_encoded = pd.get_dummies(
    X_test,
    drop_first=True
)


# Make sure training and testing columns are identical
X_encoded, X_test_encoded = X_encoded.align(
    X_test_encoded,
    join="left",
    axis=1,
    fill_value=0
)


print(
    "Encoded Training Shape:",
    X_encoded.shape
)

print(
    "Encoded Testing Shape:",
    X_test_encoded.shape
)


# ============================================================
# 14. CONVERT BOOLEAN VALUES TO INTEGER
# ============================================================

X_encoded = X_encoded.astype(float)
X_test_encoded = X_test_encoded.astype(float)


# ============================================================
# 15. TRAIN-VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining Data Shape:", X_train.shape)
print("Validation Data Shape:", X_val.shape)


# ============================================================
# 16. RANDOM FOREST CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFIER")
print("=" * 60)


random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


print("\nTraining Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest training completed!")


# Prediction
y_pred_rf = random_forest_model.predict(
    X_val
)


# ============================================================
# 17. RANDOM FOREST EVALUATION
# ============================================================

rf_accuracy = accuracy_score(
    y_val,
    y_pred_rf
)

rf_precision = precision_score(
    y_val,
    y_pred_rf,
    pos_label="Left"
)

rf_recall = recall_score(
    y_val,
    y_pred_rf,
    pos_label="Left"
)

rf_f1 = f1_score(
    y_val,
    y_pred_rf,
    pos_label="Left"
)


print("\nRandom Forest Results:")

print(
    "Accuracy :",
    round(rf_accuracy, 4)
)

print(
    "Precision:",
    round(rf_precision, 4)
)

print(
    "Recall   :",
    round(rf_recall, 4)
)

print(
    "F1 Score :",
    round(rf_f1, 4)
)


print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred_rf
    )
)


# ============================================================
# 18. RANDOM FOREST CONFUSION MATRIX
# ============================================================

cm_rf = confusion_matrix(
    y_val,
    y_pred_rf,
    labels=["Stayed", "Left"]
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    xticklabels=["Stayed", "Left"],
    yticklabels=["Stayed", "Left"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 19. DECISION TREE CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE CLASSIFIER")
print("=" * 60)


decision_tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=15
)


print("\nTraining Decision Tree...")

decision_tree_model.fit(
    X_train,
    y_train
)

print("Decision Tree training completed!")


# Prediction
y_pred_dt = decision_tree_model.predict(
    X_val
)


# ============================================================
# 20. DECISION TREE EVALUATION
# ============================================================

dt_accuracy = accuracy_score(
    y_val,
    y_pred_dt
)

dt_precision = precision_score(
    y_val,
    y_pred_dt,
    pos_label="Left"
)

dt_recall = recall_score(
    y_val,
    y_pred_dt,
    pos_label="Left"
)

dt_f1 = f1_score(
    y_val,
    y_pred_dt,
    pos_label="Left"
)


print("\nDecision Tree Results:")

print(
    "Accuracy :",
    round(dt_accuracy, 4)
)

print(
    "Precision:",
    round(dt_precision, 4)
)

print(
    "Recall   :",
    round(dt_recall, 4)
)

print(
    "F1 Score :",
    round(dt_f1, 4)
)


print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred_dt
    )
)


# ============================================================
# 21. DECISION TREE CONFUSION MATRIX
# ============================================================

cm_dt = confusion_matrix(
    y_val,
    y_pred_dt,
    labels=["Stayed", "Left"]
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    xticklabels=["Stayed", "Left"],
    yticklabels=["Stayed", "Left"]
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/decision_tree_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 22. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


comparison_df = pd.DataFrame({

    "Model": [
        "Random Forest",
        "Decision Tree"
    ],

    "Accuracy": [
        rf_accuracy,
        dt_accuracy
    ],

    "Precision": [
        rf_precision,
        dt_precision
    ],

    "Recall": [
        rf_recall,
        dt_recall
    ],

    "F1 Score": [
        rf_f1,
        dt_f1
    ]
})


print("\n")
print(comparison_df)


# Save comparison
comparison_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# ============================================================
# 23. MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison_df,
    x="Model",
    y="Accuracy"
)

plt.title(
    "Classification Model Accuracy Comparison"
)

plt.xlabel("Classification Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "outputs/model_accuracy_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 24. RANDOM FOREST TEST DATA PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("TEST DATA PREDICTION")
print("=" * 60)


test_predictions = random_forest_model.predict(
    X_test_encoded
)


# ============================================================
# 25. TEST DATA EVALUATION
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    pos_label="Left"
)

test_recall = recall_score(
    y_test,
    test_predictions,
    pos_label="Left"
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    pos_label="Left"
)


print("\nRandom Forest Test Results:")

print(
    "Accuracy :",
    round(test_accuracy, 4)
)

print(
    "Precision:",
    round(test_precision, 4)
)

print(
    "Recall   :",
    round(test_recall, 4)
)

print(
    "F1 Score :",
    round(test_f1, 4)
)


print("\nTest Classification Report:")

print(
    classification_report(
        y_test,
        test_predictions
    )
)


# ============================================================
# 26. SAVE TEST PREDICTIONS
# ============================================================

prediction_output = test_df.copy()

prediction_output[
    "Predicted_Attrition"
] = test_predictions


prediction_output.to_csv(
    "outputs/test_predictions.csv",
    index=False
)


print(
    "\nTest predictions saved successfully!"
)


# ============================================================
# 27. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


feature_importance_df = pd.DataFrame({

    "Feature": X_encoded.columns,

    "Importance":
        random_forest_model.feature_importances_

})


feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nTop 15 Important Features:")

print(
    feature_importance_df.head(15)
)


# Save feature importance
feature_importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)


# ============================================================
# 28. FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = (
    feature_importance_df
    .head(15)
)


plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Features Influencing Employee Attrition"
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "outputs/top_15_feature_importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 29. SAVE MODELS
# ============================================================

joblib.dump(
    random_forest_model,
    "models/random_forest_model.pkl"
)

joblib.dump(
    decision_tree_model,
    "models/decision_tree_model.pkl"
)


# Save feature columns
joblib.dump(
    list(X_encoded.columns),
    "models/feature_columns.pkl"
)


print("\nModels saved successfully!")


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)


print("\nDataset:")
print("Training Records :", len(train_df))
print("Testing Records  :", len(test_df))
print("Total Features   :", X.shape[1])


print("\nRandom Forest Validation Accuracy:",
      round(rf_accuracy * 100, 2), "%")


print("Decision Tree Validation Accuracy:",
      round(dt_accuracy * 100, 2), "%")


print("\nRandom Forest Test Accuracy:",
      round(test_accuracy * 100, 2), "%")


print("\nGenerated Output Files:")
print("1. EDA graphs")
print("2. Confusion matrices")
print("3. Model comparison")
print("4. Test predictions")
print("5. Feature importance")


print("\nSaved Models:")
print("1. random_forest_model.pkl")
print("2. decision_tree_model.pkl")
print("3. feature_columns.pkl")


print("\nProject execution completed!")