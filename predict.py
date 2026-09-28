# ============================================================
# EMPLOYEE ATTRITION PREDICTION SYSTEM
# ============================================================

import pandas as pd
import joblib


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    "models/random_forest_model.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

data = pd.read_csv(
    "dataset/train.csv"
)


# ============================================================
# 3. PREPARE INPUT COLUMN LIST
# ============================================================

input_columns = [
    column
    for column in data.columns
    if column not in ["Attrition", "Employee ID"]
]


# ============================================================
# 4. DISPLAY HEADER
# ============================================================

print("\n")
print("=" * 70)
print("          EMPLOYEE ATTRITION PREDICTION SYSTEM")
print("=" * 70)

print(
    "\nEnter employee details below."
)

print(
    "The system will predict whether the employee is likely "
    "to Leave or Stay.\n"
)


# ============================================================
# 5. COLLECT EMPLOYEE DETAILS
# ============================================================

employee_data = {}


for column in input_columns:

    print("\n" + "-" * 60)
    print(column)

    # --------------------------------------------------------
    # CATEGORICAL FEATURE
    # --------------------------------------------------------

    if data[column].dtype == "object":

        options = (
            data[column]
            .dropna()
            .unique()
            .tolist()
        )

        print("\nAvailable options:")

        for i, option in enumerate(
            options,
            start=1
        ):
            print(
                f"{i}. {option}"
            )

        while True:

            try:

                choice = int(
                    input(
                        "Enter option number: "
                    )
                )

                if 1 <= choice <= len(options):

                    employee_data[column] = (
                        options[choice - 1]
                    )

                    break

                print(
                    "Invalid option. Please try again."
                )

            except ValueError:

                print(
                    "Please enter a valid number."
                )

    # --------------------------------------------------------
    # NUMERICAL FEATURE
    # --------------------------------------------------------

    else:

        while True:

            try:

                value = float(
                    input(
                        f"Enter {column}: "
                    )
                )

                employee_data[column] = value

                break

            except ValueError:

                print(
                    "Please enter a valid numeric value."
                )


# ============================================================
# 6. CREATE EMPLOYEE DATAFRAME
# ============================================================

employee_df = pd.DataFrame(
    [employee_data]
)


# ============================================================
# 7. COMBINE TRAINING DATA + NEW EMPLOYEE
#
# This ensures categorical encoding is exactly the same
# as the encoding used during model training.
# ============================================================

training_features = data[
    input_columns
].copy()

combined_data = pd.concat(
    [
        training_features,
        employee_df
    ],
    ignore_index=True
)


# ============================================================
# 8. ONE-HOT ENCODING
# ============================================================

combined_encoded = pd.get_dummies(
    combined_data,
    drop_first=True
)


# Take only the last row
employee_encoded = combined_encoded.iloc[
    [-1]
].copy()


# ============================================================
# 9. MATCH TRAINING FEATURES
# ============================================================

employee_encoded = employee_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)


# Convert to numeric
employee_encoded = employee_encoded.astype(float)


# ============================================================
# 10. MAKE PREDICTION
# ============================================================

prediction = model.predict(
    employee_encoded
)


# ============================================================
# 11. PREDICTION PROBABILITY
# ============================================================

probability = model.predict_proba(
    employee_encoded
)

classes = model.classes_


# ============================================================
# 12. DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 70)
print("                    PREDICTION RESULT")
print("=" * 70)

print(
    "\nPredicted Employee Attrition:",
    prediction[0]
)


print("\nPrediction Probability:")

for i, class_name in enumerate(classes):

    print(
        f"{class_name}: "
        f"{probability[0][i] * 100:.2f}%"
    )


print("\n" + "=" * 70)


if prediction[0] == "Left":

    print(
        "⚠️ Prediction: Employee is likely to LEAVE the company."
    )

else:

    print(
        "✅ Prediction: Employee is likely to STAY in the company."
    )


print("=" * 70)