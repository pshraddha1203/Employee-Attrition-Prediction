# ============================================================
# EMPLOYEE ATTRITION PREDICTION WEB APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/random_forest_model.pkl"
    )

    feature_columns = joblib.load(
        "models/feature_columns.pkl"
    )

    return model, feature_columns


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        "dataset/train.csv"
    )


model, feature_columns = load_model()
data = load_dataset()


# ============================================================
# HEADER
# ============================================================

st.title(
    "👨‍💼 Employee Attrition Prediction System"
)

st.write(
    "A Machine Learning based system for predicting "
    "whether an employee is likely to leave or stay "
    "in the organization."
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Project Information"
)

st.sidebar.write(
    """
    **Project:**  
    Employee Attrition Prediction

    **Models:**  
    • Random Forest  
    • Decision Tree

    **Programming:**  
    Python

    **Dataset:**  
    Employee Attrition Dataset

    **Target:**  
    Attrition
    """
)


# ============================================================
# INPUT COLUMNS
# ============================================================

input_columns = [
    column
    for column in data.columns
    if column not in [
        "Attrition",
        "Employee ID"
    ]
]


# ============================================================
# EMPLOYEE INPUT
# ============================================================

st.subheader(
    "Enter Employee Details"
)

input_data = {}


# Two-column layout
col1, col2 = st.columns(2)


for index, column in enumerate(
    input_columns
):

    current_col = (
        col1
        if index % 2 == 0
        else col2
    )


    # ========================================================
    # CATEGORICAL INPUT
    # ========================================================

    if data[column].dtype == "object":

        options = (
            data[column]
            .dropna()
            .unique()
            .tolist()
        )

        input_data[column] = (
            current_col.selectbox(
                column,
                options
            )
        )


    # ========================================================
    # NUMERICAL INPUT
    # ========================================================

    else:

        min_value = float(
            data[column].min()
        )

        max_value = float(
            data[column].max()
        )

        default_value = float(
            data[column].median()
        )

        input_data[column] = (
            current_col.number_input(
                column,
                min_value=min_value,
                max_value=max_value,
                value=default_value
            )
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()


predict_button = st.button(
    "🔍 Predict Employee Attrition",
    use_container_width=True
)


if predict_button:

    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    employee_df = pd.DataFrame(
        [input_data]
    )


    # ========================================================
    # COMBINE TRAINING + NEW EMPLOYEE
    # ========================================================

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


    # ========================================================
    # ONE-HOT ENCODING
    # ========================================================

    combined_encoded = pd.get_dummies(
        combined_data,
        drop_first=True
    )


    # Last row = new employee
    employee_encoded = (
        combined_encoded
        .iloc[[-1]]
        .copy()
    )


    # ========================================================
    # MATCH FEATURES
    # ========================================================

    employee_encoded = (
        employee_encoded
        .reindex(
            columns=feature_columns,
            fill_value=0
        )
    )


    employee_encoded = (
        employee_encoded
        .astype(float)
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        employee_encoded
    )


    # ========================================================
    # PROBABILITY
    # ========================================================

    probability = model.predict_proba(
        employee_encoded
    )

    classes = model.classes_


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Prediction Result"
    )


    result = prediction[0]


    if result == "Left":

        st.error(
            "⚠️ Employee is predicted to LEAVE the company."
        )

    else:

        st.success(
            "✅ Employee is predicted to STAY in the company."
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.subheader(
        "Prediction Probability"
    )


    probability_df = pd.DataFrame({

        "Attrition Status":
            classes,

        "Probability (%)":
            [
                round(
                    value * 100,
                    2
                )
                for value in probability[0]
            ]

    })


    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CHART
    # ========================================================

    st.subheader(
        "Probability Chart"
    )


    chart_data = (
        probability_df
        .set_index(
            "Attrition Status"
        )
    )


    st.bar_chart(
        chart_data
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Employee Attrition Prediction using Machine Learning "
    "| Random Forest Classification"
)