import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder


# Load model (cached)
@st.cache_resource
def load_model():
    return joblib.load("tuned_model/best_model_lr_with_best_features.pkl")


model = load_model()

# Categorical and numerical fields
categorical_fields = {
    "Gender": ["Male", "Female"],
    "Married": ["Yes", "No"],
    "Dependents": ["0", "1", "2", "3+"],
    "Education": ["Graduate", "Not Graduate"],
    "Self_Employed": ["Yes", "No"],
    "Property_Area": ["Urban", "Semiurban", "Rural"],
}

numerical_fields = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Streamlit UI
st.title("🏦 Loan Approval Prediction App")

user_input = {}

# Collect categorical inputs with defaults
st.header("Categorical Inputs")
default_categorical = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "Property_Area": "Urban",
}
for field, options in categorical_fields.items():
    default_index = options.index(default_categorical[field])
    user_input[field] = st.selectbox(field, options, index=default_index)

# Collect numerical inputs with default values
st.header("Numerical Inputs")
default_numerical = {
    "ApplicantIncome": 5000.0,
    "CoapplicantIncome": 2000.0,
    "LoanAmount": 150.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
}

for field in numerical_fields:
    user_input[field] = st.number_input(field, value=default_numerical[field], step=1.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# --------- Feature Engineering ---------

input_df["Credit_History"] = input_df["Credit_History"].astype("category")

# Total Income
input_df["TotalIncome"] = input_df["ApplicantIncome"] + input_df["CoapplicantIncome"]

# Log Transforms
input_df["LoanAmount_log"] = np.log1p(input_df["LoanAmount"])
input_df["CoapplicantIncome_log"] = np.log1p(input_df["CoapplicantIncome"])
input_df["ApplicantIncome_log"] = np.log1p(input_df["ApplicantIncome"])
input_df["TotalIncome_log"] = np.log1p(input_df["TotalIncome"])

# Loan-to-Income Ratio
input_df["LoanToIncomeRatio"] = input_df["LoanAmount_log"] / (
    input_df["ApplicantIncome_log"] + input_df["CoapplicantIncome_log"]
)

# Binning LoanAmount_log
input_df["LoanAmount_Bin"] = pd.cut(
    input_df["LoanAmount_log"],
    bins=[0, 100, 200, 700],
    labels=["Low", "Medium", "High"],
)

# Binning Loan_Amount_Term
input_df["Loan_Amount_Term_Bin"] = pd.cut(
    input_df["Loan_Amount_Term"],
    bins=[0, 60, 180, 300, 480],  # hardcoded max bin to avoid ValueError
    labels=["Very Short", "Short", "Medium", "Long"],
)

# EMI and EMI log
input_df["EMI"] = input_df["LoanAmount"] * 1000 / input_df["Loan_Amount_Term"]
input_df["EMI_log"] = np.log1p(input_df["EMI"])

# --------- Encoding ---------


def encode_categorical(df, method="label"):
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=["object", "category"]).columns

    if method == "label":
        le = LabelEncoder()
        for col in cat_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    elif method == "onehot":
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
    else:
        raise ValueError("Encoding method must be either 'label' or 'onehot'")

    return df_encoded


encoded_df = encode_categorical(input_df, method="label")

# --------- Prediction ---------

# Use all required features that were used during model training
X_best =encoded_df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Credit_History', 'Property_Area', 'LoanToIncomeRatio',
       'LoanAmount_Bin', 'LoanAmount_log', 'CoapplicantIncome_log',
       'ApplicantIncome_log', 'TotalIncome_log', 'Loan_Amount_Term_Bin',
       'EMI_log']]

if st.button("Predict Loan Approval"):
    try:
        prediction = model.predict(X_best)[0]
        result = "Yes ✅" if prediction == 1 else "No ❌"
        st.subheader(f"Loan Approved: **{result}**")

        # Add probability scores
        proba = model.predict_proba(X_best)[0]
        st.write(f"Probability of approval: {proba[1]:.2%}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Please check that all input values are valid.")
